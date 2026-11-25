"""TCM for identical valid-convolution domains, with clipped tiles.
Run from any directory: python recompute.py. No external project writes.
"""
from pathlib import Path
import json,csv,time,hashlib
import numpy as np
from numba import njit
ROOT=Path(__file__).resolve().parent

@njit(cache=True,inline='always')
def event(a,p,R,C,last):
    resident=C[a]>0
    delta=0
    if resident:
        if R[p]>=0: delta=abs(R[p]-a)
    else:
        if last>=0: delta=abs(last-a)
        last=a
    old=R[p]
    if old>=0:C[old]-=1
    R[p]=a;C[a]+=1
    return last,int(delta>1)

@njit(cache=True)
def evaluate(cas,IF,OF,S,K,OT,YT,XT,IT,debug=False):
    H=S+K-1
    n=OT*IT*K*K if cas else OT*YT*XT
    ir=np.full(n,-1,np.int64);orr=np.full(n,-1,np.int64)
    ic=np.zeros(IF*H*H,np.int64);oc=np.zeros(OF*S*S,np.int64)
    il=-1;ol=-1;ci=0;co=0;macs=0
    visits=np.zeros(OF*IF*S*S*K*K if debug else 0,np.int64)
    output=np.zeros(OF*S*S if debug else 0,np.int64)
    if cas:
      # Tiled/unrolled form of (iy, of, if, ix, ky, kx).
      for iy in range(H):
       for ob in range(0,OF,OT):
        for ib in range(0,IF,IT):
         for ix in range(H):
          for ot in range(min(OT,OF-ob)):
           of=ob+ot
           for it in range(min(IT,IF-ib)):
            inf=ib+it
            for ky in range(K):
             oy=iy-ky
             if oy<0 or oy>=S:continue
             for kx in range(K):
              ox=ix-kx
              if ox<0 or ox>=S:continue
              p=((ot*IT+it)*K+ky)*K+kx
              ia=(inf*H+iy)*H+ix;oa=(of*S+oy)*S+ox
              il,c=event(ia,p,ir,ic,il);ci+=c
              ol,c=event(oa,p,orr,oc,ol);co+=c
              macs+=1
              if debug:
               op=((of*IF+inf)*S*S+oy*S+ox)*K*K+ky*K+kx
               visits[op]+=1
               wi=((of*IF+inf)*K+ky)*K+kx
               output[oa]+=((ia%13)-6)*((wi%7)-3)
    else:
      # Canonical affine order with output/channel tiling and PE unrolling.
      for ob in range(0,OF,OT):
       for inf in range(IF):
        for yb in range(0,S,YT):
         for xb in range(0,S,XT):
          for ky in range(K):
           for kx in range(K):
            for ot in range(min(OT,OF-ob)):
             of=ob+ot
             for yt in range(min(YT,S-yb)):
              oy=yb+yt
              for xt in range(min(XT,S-xb)):
               ox=xb+xt
               p=(ot*YT+yt)*XT+xt
               ia=(inf*H+oy+ky)*H+ox+kx;oa=(of*S+oy)*S+ox
               il,c=event(ia,p,ir,ic,il);ci+=c
               ol,c=event(oa,p,orr,oc,ol);co+=c
               macs+=1
               if debug:
                op=((of*IF+inf)*S*S+oy*S+ox)*K*K+ky*K+kx
                visits[op]+=1
                wi=((of*IF+inf)*K+ky)*K+kx
                output[oa]+=((ia%13)-6)*((wi%7)-3)
    return ci,co,macs,n,visits,output

def naive_convolution(IF,OF,S,K):
    H=S+K-1
    inp=(np.arange(IF*H*H)%13-6).reshape(IF,H,H)
    w=(np.arange(OF*IF*K*K)%7-3).reshape(OF,IF,K,K)
    out=np.zeros((OF,S,S),dtype=np.int64)
    for o in range(OF):
      for y in range(S):
       for x in range(S):out[o,y,x]=np.sum(inp[:,y:y+K,x:x+K]*w[o])
    return out.ravel()

def reference_counts(cas,IF,OF,S,K,OT,YT,XT,IT):
    # Enumerate the mathematical domain first, then sort by the tiled schedule.
    H=S+K-1; events=[]
    for o in range(OF):
      for i in range(IF):
       for y in range(S):
        for x in range(S):
         for ky in range(K):
          for kx in range(K):
           if cas:
            key=(y+ky,o//OT,i//IT,x+kx,o%OT,i%IT,ky,kx)
            p=((o%OT*IT+i%IT)*K+ky)*K+kx
           else:
            key=(o//OT,i,y//YT,x//XT,ky,kx,o%OT,y%YT,x%XT)
            p=(o%OT*YT+y%YT)*XT+x%XT
           events.append((key,p,(i*H+y+ky)*H+x+kx,(o*S+y)*S+x))
    events.sort();n=OT*IT*K*K if cas else OT*YT*XT
    resident=[[-1]*n,[-1]*n];last=[-1,-1];counts=[0,0]
    for _,p,ia,oa in events:
      for j,a in enumerate((ia,oa)):
       R=resident[j]
       if a in R: previous=R[p]
       else: previous=last[j];last[j]=a
       if previous>=0 and abs(previous-a)>1:counts[j]+=1
       R[p]=a
    return tuple(counts)

def validate():
    cases=[(1,1,1,1,1,1,1,1),(2,3,3,2,2,2,2,2),(3,5,4,3,2,3,3,2),(1,2,2,5,4,5,5,8),(3,4,5,1,3,2,3,2)]
    for IF,OF,S,K,OT,YT,XT,IT in cases:
      ref=naive_convolution(IF,OF,S,K)
      for cas in [False,True]:
        r=evaluate(cas,IF,OF,S,K,OT,YT,XT,IT,True)
        assert (r[0],r[1])==reference_counts(cas,IF,OF,S,K,OT,YT,XT,IT), 'TCM reference mismatch'
        assert r[2]==OF*IF*S*S*K*K
        assert np.all(r[4]==1),'missing or duplicate MAC'
        assert np.array_equal(r[5],ref),'numerical convolution mismatch'
    # Startup, global residency, reverse-adjacent motion, and unconditional R update.
    R=np.full(2,-1,np.int64);C=np.zeros(12,np.int64);L=-1;observed=[]
    for a,p in [(0,0),(1,1),(0,1),(4,0),(4,1),(3,0)]:
      L,c=event(a,p,R,C,L); observed.append(c)
      assert np.array_equal(C,np.bincount(R[R>=0],minlength=12))
    assert observed==[0,0,0,1,1,0],observed
    return {'small_workloads':len(cases),'schedule_checks':2*len(cases),'exact_mac_coverage':True,'numerical_convolution':True,'state_transition_check':True,'independent_sorted_trace_tcm':True}

def main():
    tests=validate();(ROOT/'validation.json').write_text(json.dumps(tests,indent=2));print('VALIDATED',tests,flush=True)
    rows=[]
    for group in ['sample','in_f','of']:
      cfg=[(8,64,S,K) for K in [3,5,7,11] for S in [14,28,42,56]] if group=='sample' else ([(i,64,56,3) for i in range(32,257,32)] if group=='in_f' else [(8,o,56,3) for o in range(32,257,32)])
      for idx,(IF,OF,S,K) in enumerate(cfg):
        t=time.time();tile=K if group=='sample' else 14
        c=evaluate(False,IF,OF,S,K,16,tile,tile,8)
        o=evaluate(True,IF,OF,S,K,64,1,1,8)
        # Sensitivity: exactly match nominal resident entries in both operand arrays.
        m=evaluate(False,IF,OF,S,K,64,8*K,K,8)
        expected=OF*IF*S*S*K*K
        assert c[2]==o[2]==m[2]==expected
        assert m[3]==o[3]
        row=dict(group=group,index=idx,IF=IF,OF=OF,S=S,K=K,input_side=S+K-1,macs=expected,canonical_input=int(c[0]),canonical_output=int(c[1]),canonical=int(c[0]+c[1]),cas_input=int(o[0]),cas_output=int(o[1]),cas=int(o[0]+o[1]),canonical_entries=int(c[3]),cas_entries=int(o[3]),matched_input=int(m[0]),matched_output=int(m[1]),matched=int(m[0]+m[1]),matched_entries=int(m[3]),canonical_OT=16,canonical_YT=tile,canonical_XT=tile,cas_OT=64,cas_IT=8,matched_OT=64,matched_YT=8*K,matched_XT=K)
        row['ratio']=row['canonical']/row['cas'];row['matched_ratio']=row['matched']/row['cas'];rows.append(row)
        (ROOT/'results.json').write_text(json.dumps(rows,indent=2));print(group,idx,'seconds',round(time.time()-t,2),'TCM',row['canonical'],row['cas'],'ratios',row['ratio'],row['matched_ratio'],flush=True)
    with (ROOT/'results.csv').open('w') as f:
      w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    for group in ['sample','in_f','of']:
      rr=[r for r in rows if r['group']==group];shape=(4,4) if group=='sample' else (1,8)
      d=ROOT/'data'/('calculate_indicator_'+group);d.mkdir(parents=True,exist_ok=True)
      for name,key in [('canonical_res','canonical'),('our_res','cas'),('matched_res','matched')]:np.save(d/(name+'.npy'),np.array([r[key] for r in rr],dtype=np.int64).reshape(shape))
    (ROOT/'run_manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':len(rows),'domain':'valid convolution, stride=1, dilation=1, no padding','S':'output spatial side','initial_state':'empty R=-1; no previous load L=-1; undefined predecessor contributes zero','unit_weights':True,'input_output_separate':True,'mac_counts_equal':True,'matched_capacity':'equal nominal resident entries per operand; not equal measured hardware resources'},indent=2))
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
