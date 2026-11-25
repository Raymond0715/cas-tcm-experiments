"""Collect total and global-load warp instructions for a completed paired GPU timing run."""
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
METRIC_INFO = {
    'smsp__inst_executed.sum': ('dynamic_total_warp_instructions', {'inst'}),
    'smsp__sass_inst_executed_op_global_ld.sum': ('dynamic_global_load_warp_instructions', {'inst'}),
    'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum': ('l1_global_load_requests', {'', 'request'}),
    'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum': ('l1_global_load_sectors', {'sector'}),
    'l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum': ('l1_global_load_hit_sectors', {'sector'}),
    'lts__t_sectors_srcunit_tex_op_read.sum': ('l2_from_l1tex_read_sectors', {'sector'}),
    'lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum': ('l2_from_l1tex_read_hit_sectors', {'sector'}),
}
METRICS = tuple(METRIC_INFO)

def fingerprint(ptx):
    code = re.split(r'\.section\s+\.debug', ptx, maxsplit=1)[0]
    lines = [line.split('//', 1)[0].strip() for line in code.splitlines()]
    return hashlib.sha256('\n'.join(' '.join(line.split()) for line in lines
        if line and not line.startswith(('.file', '.loc'))).encode()).hexdigest()

def worker(args):
    archive = args.archive.resolve()
    os.environ['TRITON_CACHE_DIR'] = str(Path(args.output) / 'cache/triton')
    os.environ['CUDA_CACHE_PATH'] = str(Path(args.output) / 'cache/cuda')
    sys.path.insert(0, str(archive / 'sources'))
    import torch
    import torch.nn.functional as F
    import triton
    from kernels import launch
    env = json.loads((archive / 'environment.json').read_text())
    assert torch.__version__ == env['torch'] and triton.__version__ == env['triton']
    assert torch.version.cuda == env['cuda_build']
    assert torch.cuda.get_device_name(0) == env['device'], 'GPU differs from the timing run'
    for name, sha in env['sources'].items():
        assert hashlib.sha256((archive/'sources'/name).read_bytes()).hexdigest() == sha, name
    record = json.loads((archive / 'results.json').read_text())[args.pair]
    c, q = record['case'], record['config']
    torch.manual_seed(20261115)
    torch.set_num_threads(4)
    x = torch.randn(c['n'], c['ci'], c['h'], c['w'], device='cuda') * .2
    w = torch.randn(c['co'], c['ci']//c['groups'], c['kernel'], c['kernel'], device='cuda') * .2
    ref = F.conv2d(x.cpu().double(), w.cpu().double(), stride=c['stride'],
                   padding=c['padding'], groups=c['groups']).float().cuda()
    outputs = {v: torch.empty_like(ref) for v in ('canonical', 'reuse')}
    compiled = {}
    for v in outputs:
        compiled[v] = launch(x, w, outputs[v], c, v, q)
        assert torch.allclose(outputs[v], ref, atol=2e-5, rtol=2e-4), v
    assert torch.equal(outputs['canonical'], outputs['reuse'])
    torch.cuda.synchronize()
    for v, k in compiled.items():
        resources = dict(registers=k.n_regs, spills=k.n_spills, shared_bytes=k.metadata.shared)
        assert resources == record['resources'][v], 'Compiler resource mismatch'
        old = archive/'compiler'/c['id']/q['id']/(v+'.ptx')
        assert fingerprint(k.asm['ptx']) == fingerprint(old.read_text()), 'Executable PTX mismatch'
        target = Path(args.output)/'compiler'/c['id']/q['id']
        target.mkdir(parents=True, exist_ok=True)
        (target/(v+'.ptx')).write_text(k.asm['ptx'])
    for _ in range(args.samples + args.settle):
        for _ in range(20):
            launch(x, w, outputs[args.variant], c, args.variant, q)
        torch.cuda.synchronize()
        assert torch.cuda.cudart().cudaProfilerStart() == 0
        launch(x, w, outputs[args.variant], c, args.variant, q)
        torch.cuda.synchronize()
        assert torch.cuda.cudart().cudaProfilerStop() == 0
    k = compiled[args.variant]
    print('WORKER_JSON=' + json.dumps(dict(case=c['id'], config=q['id'], variant=args.variant,
          correctness=True, bitwise_equal=True, matches_timed_kernel=True, torch=torch.__version__, triton=triton.__version__,
          device=torch.cuda.get_device_name(0), registers=k.n_regs, spills=k.n_spills,
          shared_bytes=k.metadata.shared, ptx_sha256=hashlib.sha256(k.asm['ptx'].encode()).hexdigest())))

def parse_csv(text, expected):
    lines = text.splitlines()
    headers = [i for i,line in enumerate(lines) if line.startswith('"ID",') and
               ('"Metric Name"' in line or all(f'"{m}"' in line for m in METRICS))]
    if not headers:
        raise ValueError('No metric CSV header; inspect saved log for permission/metric errors.')
    rows = csv.DictReader(io.StringIO('\n'.join(lines[headers[0]:])))
    values = {}
    def number(value):
        result = float(value.replace(',', ''))
        assert math.isfinite(result) and result >= 0, value
        return result
    wide = all(m in rows.fieldnames for m in METRICS)
    units = None
    for row in rows:
        if wide:
            if row.get('ID') == '':
                assert units is None, 'Duplicate units row'
                units = {m: row[m] for m in METRICS}
                assert all(units[m] in METRIC_INFO[m][1] for m in METRICS), units
                continue
            if not (row.get('ID') or '').isdigit(): continue
            assert units is not None, 'Missing metric units row'
            assert row['ID'] not in values, 'Duplicate launch ID'
            values[row['ID']] = {m: number(row[m]) for m in METRICS}
            continue
        metric = row.get('Metric Name')
        if metric not in METRICS: continue
        assert row['Metric Unit'] in METRIC_INFO[metric][1], row
        key = row['ID']
        assert metric not in values.setdefault(key, {}), row
        values[key][metric] = number(row['Metric Value'])
    assert len(values) == expected, (len(values), expected)
    assert all(set(v) == set(METRICS) for v in values.values()), values
    return list(values.values())

def replay_passes(text):
    """Preserve replay diagnostics from raw wide CSV where NCU provides them."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('"ID",') and '"profiler__replayer_passes"' in line:
            rows = csv.DictReader(io.StringIO('\n'.join(lines[i:])))
            return {r['ID']: r['profiler__replayer_passes'] for r in rows
                    if (r.get('ID') or '').isdigit()}
    return {}

def derived(sample):
    d = {METRIC_INFO[m][0]: sample[m] for m in METRICS}
    def ratio(a, b, scale=1):
        return scale*a/b if b else None
    d['l1_sectors_per_request'] = ratio(d['l1_global_load_sectors'], d['l1_global_load_requests'])
    for level, total, hits in (
        ('l1_global_load', 'l1_global_load_sectors', 'l1_global_load_hit_sectors'),
        ('l2_from_l1tex_read', 'l2_from_l1tex_read_sectors', 'l2_from_l1tex_read_hit_sectors')):
        d[level + '_hit_rate_percent'] = ratio(d[hits], d[total], 100)
        # Do not clip inconsistent counter values; surface them in warnings.
    return d

def summarize(out):
    data = json.loads((out / 'measurements.json').read_text())
    manifest = json.loads((out / 'manifest.json').read_text())
    settle = manifest['settle']
    pairs, raw, warnings = {}, [], []
    for d in data:
        pairs.setdefault((d['case'], d['config']), {})[d['variant']] = d
        for i, sample in enumerate(d['samples']):
            vals = derived(sample)
            raw.append(dict(case=d['case'], config=d['config'], variant=d['variant'],
                            sample_index=i, phase='diagnostic' if i < settle else 'measurement', **vals))
            for k, value in vals.items():
                if k.endswith('hit_rate_percent') and value is not None and value > 100:
                    warnings.append(f'{d["case"]}/{d["config"]}/{d["variant"]} sample {i}: {k} > 100; inspect replay/cache effects')
    rows = []
    for (case, config), variants in pairs.items():
        if set(variants) != {'canonical', 'reuse'}: continue
        for scope, start in (('all_samples', 0), ('after_settle', settle)):
            samples = {v: [derived(s) for s in variants[v]['samples'][start:]] for v in variants}
            for metric in samples['canonical'][0]:
                row = dict(case=case, config=config, scope=scope, metric=metric)
                medians = {}
                for v in ('canonical', 'reuse'):
                    vals = [s[metric] for s in samples[v] if s[metric] is not None]
                    row[v + '_valid_samples'] = len(vals)
                    medians[v] = statistics.median(vals) if vals else None
                    row[v + '_median'] = medians[v]
                    row[v + '_min'] = min(vals) if vals else None
                    row[v + '_max'] = max(vals) if vals else None
                    cv = statistics.stdev(vals)/statistics.mean(vals) if len(vals)>1 and statistics.mean(vals) else None
                    row[v + '_cv'] = cv
                    if scope == 'after_settle' and cv is not None and cv > .1:
                        warnings.append(f'{case}/{config}/{v}: {metric} CV > 10%; descriptive flag, not significance test')
                c, r = medians['canonical'], medians['reuse']
                row['reuse_minus_canonical'] = r-c if c is not None and r is not None else None
                # Hit rates use percentage-point differences, not relative reductions.
                row['reduction_percent'] = 100*(1-r/c) if c and r is not None and not metric.endswith('hit_rate_percent') else None
                rows.append(row)
    for name, entries in [('samples.csv', raw), ('comparison.csv', rows)]:
        if entries:
            with (out / name).open('w') as f:
                w = csv.DictWriter(f, fieldnames=list(entries[0])); w.writeheader(); w.writerows(entries)
    # Export the two proposed figure panels, retaining load counts as a component.
    figure = []
    for d in data:
        vals = [derived(sample) for sample in d['samples'][settle:]]
        figure.append(dict(case=d['case'], config=d['config'], variant=d['variant'],
            display_name='Canonical' if d['variant']=='canonical' else 'CAS-GPU',
            dynamic_total_warp_instructions=statistics.median(v['dynamic_total_warp_instructions'] for v in vals),
            dynamic_global_load_warp_instructions=statistics.median(v['dynamic_global_load_warp_instructions'] for v in vals)))
    with (out/'figure_instruction_values.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(figure[0]));w.writeheader();w.writerows(figure)
    for d in data:
        if any(s['smsp__inst_executed.sum'] < s['smsp__sass_inst_executed_op_global_ld.sum'] for s in d['samples']):
            warnings.append(f"{d['case']}/{d['config']}/{d['variant']}: total < global-load; inspect counter semantics")
    (out / 'quality_flags.txt').write_text('\n'.join(warnings) + '\n')
    print(f'{len(rows)} comparison rows; results: {out}', flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, help='Completed timing run written by run.py --mode full')
    p.add_argument('--sudo-ncu', action='store_true', help='Run only NCU/its target through sudo')
    p.add_argument('--samples', type=int, default=7)
    p.add_argument('--settle', type=int, default=2, help='Profiled diagnostic samples retained before measurement samples')
    p.add_argument('--smoke', action='store_true', help='One dense and one depthwise pair at the first tile')
    p.add_argument('--limit-pairs', type=int, default=22, help='Use 1 for a permission smoke test')
    p.add_argument('--summarize', type=Path)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--pair', type=int)
    p.add_argument('--variant', choices=('canonical', 'reuse'))
    p.add_argument('--output')
    args = p.parse_args()
    if args.summarize: return summarize(args.summarize.resolve())
    if args.archive is None:
        p.error('--archive is required for counter collection')
    archive = args.archive.resolve()
    if not (archive / 'complete.json').is_file() or not (archive / 'results.json').is_file():
        p.error(f'Expected a completed timing run in {archive}')
    assert args.samples >= 2 and args.settle >= 0 and 1 <= args.limit_pairs <= 22
    if args.worker: return worker(args)
    ncu = shutil.which('ncu')
    if not ncu: raise SystemExit('ncu not found')
    if args.sudo_ncu: subprocess.run(['sudo', '-v'], check=True)
    all_records = json.loads((archive / 'results.json').read_text())
    if args.smoke:
        indices = []
        for family in ('dense', 'depthwise'):
            match = next((i for i, r in enumerate(all_records) if r['case']['family'] == family), None)
            if match is not None:
                indices.append(match)
        if not indices:
            p.error('--smoke requires a timing run with a dense or depthwise case')
    else:
        indices = list(range(min(args.limit_pairs, len(all_records))))
    records = [all_records[i] for i in indices]
    out = ROOT / 'instruction_collections' / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True)
    shutil.copy2(__file__, out / 'collect_instructions.py')
    manifest = dict(archive=str(archive), metrics=METRICS, samples=args.samples, settle=args.settle, indices=indices,
        protocol='20 eager warmups before every profiled launch; application replay; no cache flush; clocks unchanged; first settle samples retained as diagnostics; later samples summarized separately; not original CUDA Graph timing protocol',
        interpretation='Total executed warp instructions and global-load warp instructions per launch (not static PTX counts or thread-level counts); no derivation of control-only instructions by subtraction; global-load L1 requests/sectors; L2 read sectors originating from L1TEX (not global-load-only). Hit rates are ratios of sector counts. Sectors/request depends on active lanes and access width, not solely coalescing. No DRAM, timing or decode attribution claim.',
        source_sha256={f:hashlib.sha256((archive / 'sources' / f).read_bytes()).hexdigest()
                       for f in ('kernels.py',)},
        ncu_version=subprocess.run([ncu, '--version'], capture_output=True, text=True).stdout,
        gpu=subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout,
        cases=records)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Output: {out}', flush=True)
    data = []; rng = random.Random(20261115)
    for progress, (i, record) in enumerate(zip(indices, records)):
        variants = ['canonical', 'reuse']; rng.shuffle(variants)
        for variant in variants:
            stem = f'{i:02d}_{variant}'
            cmd = (['sudo', '--'] if args.sudo_ncu else []) + [ncu, '--profile-from-start', 'off',
                '--cache-control', 'none', '--clock-control', 'none', '--replay-mode', 'application',
                '--metrics', ','.join(METRICS), '--csv', '--page', 'raw', '--print-units', 'base',
                sys.executable, str(Path(__file__).resolve()), '--worker', '--archive', str(archive), '--pair', str(i),
                '--variant', variant, '--samples', str(args.samples), '--settle', str(args.settle), '--output', str(out)]
            (out / (stem + '.command.json')).write_text(json.dumps(cmd, indent=2))
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (out / (stem + '.log')).write_text(proc.stdout)
            if proc.returncode or 'ERR_NVGPUCTRPERM' in proc.stdout or '==ERROR==' in proc.stdout:
                (out / 'failed.json').write_text(json.dumps(dict(pair=i, variant=variant, returncode=proc.returncode, log=stem+'.log')))
                raise SystemExit(f'Collection failed; see {out / (stem + ".log")}. No missing metric is treated as zero.')
            samples = parse_csv(proc.stdout, args.samples + args.settle)
            worker_lines = [line for line in proc.stdout.splitlines() if line.startswith('WORKER_JSON=')]
            assert worker_lines, 'Missing worker validation metadata'
            assert len(set(worker_lines)) == 1, 'Application replay worker metadata differs between passes'
            metadata = json.loads(worker_lines[0].split('=', 1)[1])
            assert metadata['correctness'] and metadata['bitwise_equal']
            data.append(dict(**metadata, samples=samples, replay_passes=replay_passes(proc.stdout)))
            (out / 'measurements.json').write_text(json.dumps(data, indent=2))
            print(f'{progress+1}/{len(records)} {metadata["case"]} {metadata["config"]} {variant}: OK', flush=True)
    summarize(out)
    (out / 'complete.json').write_text(json.dumps(dict(pairs=len(records), samples=args.samples, settle=args.settle)))

if __name__ == '__main__':
    main()
