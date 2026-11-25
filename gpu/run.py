#!/usr/bin/env python3
"""Check schedule correspondence first, then run the predeclared paired matrix."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('TRITON_CACHE_DIR',str(ROOT/'.cache'/'triton'))
os.environ.setdefault('CUDA_CACHE_PATH',str(ROOT/'.cache'/'cuda'))
import argparse,datetime,hashlib,json,random,shutil,subprocess,sys,traceback
import numpy as np
import torch
import torch.nn.functional as F
import triton
from kernels import launch
from model import geometry,expected_capture,oracle_from_capture,measure,SMALL_CASES

VARIANTS=('canonical','reuse')

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dependencies():return {n:digest(ROOT/n) for n in ('kernels.py','model.py','cases.json','configs.json')}
def make_tensors(c):
    return (torch.randn(c['n'],c['ci'],c['h'],c['w'],device='cuda')*.2,
            torch.randn(c['co'],c['ci']//c['groups'],c['kernel'],c['kernel'],device='cuda')*.2)
def reference(x,w,c):
    return F.conv2d(x.cpu().double(),w.cpu().double(),stride=c['stride'],padding=c['padding'],groups=c['groups']).float().cuda()
def check_output(y,ref):
    error=(y-ref).abs()
    row=dict(max_abs_error=error.max().item(),passed=torch.allclose(y,ref,atol=2e-5,rtol=2e-4))
    assert row['passed'],row
    return row
def metadata(k):return dict(registers=k.n_regs,spills=k.n_spills,shared_bytes=k.metadata.shared)
def artifacts(out,c,q,v,k):
    dest=out/'compiler'/c['id']/q['id'];dest.mkdir(parents=True,exist_ok=True)
    for kind in ('ptx','ttir','ttgir'):(dest/(v+'.'+kind)).write_text(k.asm[kind])
    assert 'atom.global' not in k.asm['ptx']

def check_gate(out,configs,say):
    checks=[]
    for c in SMALL_CASES:
      x,w=make_tensors(c);ref=reference(x,w,c)
      for q in configs:
        sets=[]
        for v in VARIANTS:
          ei,eo,seen=expected_capture(c,q,v);sets.append(seen)
          expected=oracle_from_capture(ei,eo)
          m=measure(c,q,v)
          assert expected==[m['input_tcm'],m['output_tcm'],m['useful_macs']]
          ti=torch.full(ei.shape,-2,dtype=torch.int64,device='cuda');to=torch.full_like(ti,-2)
          y=torch.full_like(ref,float('nan'))
          launch(x,w,y,c,v,q,(ti,to));torch.cuda.synchronize()
          ai,ao=ti.cpu().numpy(),to.cpu().numpy()
          if not np.array_equal(ai,ei) or not np.array_equal(ao,eo):
            np.savez_compressed(out/'capture_failure.npz',actual_i=ai,expected_i=ei,actual_o=ao,expected_o=eo)
            raise AssertionError(f'capture mismatch {c["id"]} {q["id"]} {v}')
          debug_check=check_output(y,ref)
          y.fill_(float('nan'));k=launch(x,w,y,c,v,q);torch.cuda.synchronize()
          prod_check=check_output(y,ref)
          artifacts(out,c,q,v,k)
          dest=out/'captures';dest.mkdir(exist_ok=True)
          np.savez_compressed(dest/f'{c["id"]}_{q["id"]}_{v}.npz',input=ai,output=ao)
          row=dict(case=c,config=q,variant=v,debug_output=debug_check,production_output=prod_check,
                   tcm=m,gpu_logical_address_capture_matches_domain=True,resources=metadata(k))
          checks.append(row);say(f'PASS {c["id"]} {q["id"]} {v} {m}')
          (out/'checks.json').write_text(json.dumps(checks,indent=2))
        assert sets[0]==sets[1]
    marker=dict(passed=True,checks=len(checks),sources=dependencies(),run=str(out.relative_to(ROOT)),
                interpretation='captured logical operand addresses and MAC domain, NOT physical warp issue or cache trace')
    (out/'gate.json').write_text(json.dumps(marker,indent=2))
    (ROOT/'gate_latest.json').write_text(json.dumps(marker,indent=2))

def elapsed_graph(g,count):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record();g.replay();end.record();end.synchronize()
    return start.elapsed_time(end)*1000/count

def benchmark(out,cases,configs,args,say):
    gate=json.loads((ROOT/'gate_latest.json').read_text())
    assert gate['passed'] and gate['sources']==dependencies(),'First pass --mode check for these exact sources/configurations'
    records=[]
    for c in cases:
      x,w=make_tensors(c);ref=reference(x,w,c)
      for q in configs:
        functions={};compiled={};numeric={};ys={}
        for v in VARIANTS:
          ys[v]=torch.full_like(ref,float('nan'))
          k=launch(x,w,ys[v],c,v,q);torch.cuda.synchronize()
          numeric[v]=check_output(ys[v],ref);compiled[v]=k
          artifacts(out,c,q,v,k)
          functions[v]=lambda v=v:launch(x,w,ys[v],c,v,q)
        numeric['variants_bitwise_equal']=torch.equal(ys['canonical'],ys['reuse'])
        assert numeric['variants_bitwise_equal'], 'Per-output FMA order differs unexpectedly'
        for fn in functions.values():
          for _ in range(20):fn()
        torch.cuda.synchronize()
        graphs={}
        for v,fn in functions.items():
          g=torch.cuda.CUDAGraph()
          with torch.cuda.graph(g):
            for _ in range(args.repeat):fn()
          graphs[v]=g
          for _ in range(3):g.replay()
        torch.cuda.synchronize()
        samples={v:[] for v in VARIANTS};orders=[]
        for _ in range(args.rounds):
          order=list(VARIANTS);random.shuffle(order);orders.append(order)
          for v in order:samples[v].append(elapsed_graph(graphs[v],args.repeat))
        record=dict(case=c,config=q,numerical=numeric,samples_us=samples,round_orders=orders,
             stats={v:dict(median_us=float(np.median(a)),q25_us=float(np.percentile(a,25)),q75_us=float(np.percentile(a,75)),min_us=min(a),max_us=max(a)) for v,a in samples.items()},
             resources={v:metadata(k) for v,k in compiled.items()},tcm={v:measure(c,q,v) for v in VARIANTS})
        assert record['tcm']['canonical']['useful_macs']==record['tcm']['reuse']['useful_macs']
        records.append(record);(out/'results.json').write_text(json.dumps(records,indent=2))
        say(c['id']+' '+q['id']+' '+str(record['stats']))
    (out/'complete.json').write_text(json.dumps(dict(pairs=len(records),gate=gate,completed=True),indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['check','full','profile'],default='check')
    p.add_argument('--case');p.add_argument('--config');p.add_argument('--variant',default='canonical',choices=VARIANTS)
    p.add_argument('--rounds',type=int,default=15);p.add_argument('--repeat',type=int,default=50)
    args=p.parse_args();stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out=ROOT/'runs'/stamp;out.mkdir(parents=True)
    snapshots=out/'sources';snapshots.mkdir()
    for source in list(ROOT.glob('*.py'))+list(ROOT.glob('*.json')):
      if source.name!='gate_latest.json':shutil.copy2(source,snapshots/source.name)
    log=open(out/'run.log','w',buffering=1)
    def say(s):print(s,flush=True);print(s,file=log,flush=True)
    env=dict(args=vars(args),utc=stamp,python=sys.version,torch=torch.__version__,triton=triton.__version__,
             cuda_build=torch.version.cuda,sources=dependencies(),seed=20261112,
             timing='warm CUDA Graph replay; CUDA events; 20 eager and 3 graph warmups; no allocation/JIT/H2D/D2H; clocks not locked',
             arithmetic='FP32 explicit FMA on both sides; same per-output ic,ky,kx order; fixed register accumulators',
             tolerance=dict(atol=2e-5,rtol=2e-4),
             dependency_caveat='torch 2.5 pins triton 3.1; tested direct JIT/tensor/graph use with local triton 3.5; no torch.compile')
    try:
      env['nvidia_smi']=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout
      assert torch.cuda.is_available(),'CUDA unavailable'
      env['device']=torch.cuda.get_device_name(0);env['properties']=str(torch.cuda.get_device_properties(0))
      (out/'environment.json').write_text(json.dumps(env,indent=2))
      torch.manual_seed(20261112);random.seed(20261112);torch.set_num_threads(4)
      torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
      configs=json.loads((ROOT/'configs.json').read_text());cases=json.loads((ROOT/'cases.json').read_text())
      if args.case:cases=[c for c in cases if c['id']==args.case]
      if args.config:configs=[q for q in configs if q['id']==args.config]
      assert configs and cases
      if args.mode=='check':
        assert not args.config,'The gate must check both configurations'
        check_gate(out,configs,say)
      elif args.mode=='profile':
        c,q=cases[0],configs[0];x,w=make_tensors(c);y=reference(x,w,c)
        for _ in range(3):launch(x,w,y,c,args.variant,q)
        torch.cuda.synchronize()
      else:benchmark(out,cases,configs,args,say)
      say('COMPLETE '+str(out))
    except Exception:
      msg=traceback.format_exc();say(msg);(out/'failure.txt').write_text(msg);raise
    finally:(out/'environment.json').write_text(json.dumps(env,indent=2));log.close()

if __name__=='__main__':main()
