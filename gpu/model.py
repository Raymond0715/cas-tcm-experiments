"""One fixed consumer-slot model for both direct-FMA schedules.
Slot=(output row in tile, output channel in tile, output x in tile).
Both operands use the paper R/last event rule; no slot/capacity retuning by variant.
R is an explicit logical residency assumption, not physical GPU cache state.
"""
from pathlib import Path
import json,math
import numpy as np
from numba import njit
ROOT=Path(__file__).resolve().parent

def geometry(c,q,v):
    ci,co,h,w,k,s,p=[c[t] for t in ('ci','co','h','w','kernel','stride','padding')]
    oh=(h+2*p-k)//s+1;ow=(w+2*p-k)//s+1
    by,bx,bc=[q[t] for t in ('by','bx','bc')];dw=c['groups']!=1
    lanes=bc*by*bx
    steps=(1 if dw else ci)*(k if v=='canonical' else (by-1)*s+k)*k
    ctas=c['n']*math.ceil(co/bc)*math.ceil(oh/by)*math.ceil(ow/bx)
    return oh,ow,0,0,ctas,steps,lanes

@njit(inline='always')
def event(a,p,R,C,last):
    prev=R[p] if C[a]>0 else last
    if C[a]==0:last=a
    if R[p]>=0:C[R[p]]-=1
    R[p]=a;C[a]+=1
    return last,int(prev>=0 and abs(a-prev)>1)

@njit(cache=True)
def evaluate(n,ci,co,h,w,k,s,p,groups,by,bx,bc,reuse):
    oh=(h+2*p-k)//s+1;ow=(w+2*p-k)//s+1;dw=groups!=1
    size=bc*by*bx
    ir=np.full(size,-1,np.int64);orr=np.full(size,-1,np.int64)
    ins=np.zeros(n*ci*h*w,np.int64);outs=np.zeros(n*co*oh*ow,np.int64)
    counts=np.zeros(4,np.int64)
    for batch in range(n):
     for cb in range(0,co,bc):
      for y0 in range(0,oh,by):
       for x0 in range(0,ow,bx):
        il=-1;ol=-1
        for ic0 in range(1 if dw else ci):
         for sy in range((by-1)*s+k if reuse else k):
          for kx in range(k):
           loaded_row=False
           for r in range(min(by,oh-y0)):
            ky=sy-r*s if reuse else sy
            if ky<0 or ky>=k:continue
            oy=y0+r;iy=oy*s-p+ky
            if iy<0 or iy>=h:continue
            for oc in range(cb,min(cb+bc,co)):
             ic=oc if dw else ic0
             for dx in range(min(bx,ow-x0)):
              ox=x0+dx;ix=ox*s-p+kx
              if ix<0 or ix>=w:continue
              slot=(r*bc+oc-cb)*bx+dx
              ia=((batch*ci+ic)*h+iy)*w+ix
              oa=((batch*co+oc)*oh+oy)*ow+ox
              il,a=event(ia,slot,ir,ins,il);ol,b=event(oa,slot,orr,outs,ol)
              counts[0]+=a;counts[1]+=b;counts[2]+=1
              if (dw or oc==cb) and (not reuse or not loaded_row):counts[3]+=1
            loaded_row=True
        for j in range(size):
         if ir[j]>=0:ins[ir[j]]-=1
         if orr[j]>=0:outs[orr[j]]-=1
         ir[j]=-1;orr[j]=-1
    return counts,size

def measure(c,q,v):
    counts,size=evaluate(*[c[t] for t in ('n','ci','co','h','w','kernel','stride','padding','groups')],
                         *[q[t] for t in ('by','bx','bc')],v=='reuse')
    return dict(input_tcm=int(counts[0]),output_tcm=int(counts[1]),total_tcm=int(counts[0]+counts[1]),
                useful_macs=int(counts[2]),nominal_entries_per_operand_per_CTA=int(size),source_input_requested_bytes=int(counts[3]*4))

def expected_capture(c,q,v):
    """Independent domain enumeration assigning useful MACs to time and slot."""
    oh,ow,_,_,ctas,steps,lanes=geometry(c,q,v)
    ti=np.full((ctas,steps,lanes),-1,dtype=np.int64);to=ti.copy()
    ci,co,h,w,k,s,p=[c[t] for t in ('ci','co','h','w','kernel','stride','padding')]
    by,bx,bc=[q[t] for t in ('by','bx','bc')];dw=c['groups']!=1
    ny=math.ceil(oh/by);nx=math.ceil(ow/bx);nc=math.ceil(co/bc)
    seen=set()
    for batch in range(c['n']):
     for oc in range(co):
      for ic0 in range(1 if dw else ci):
       ic=oc if dw else ic0
       for oy in range(oh):
        for ox in range(ow):
         for ky in range(k):
          for kx in range(k):
           iy=oy*s-p+ky;ix=ox*s-p+kx
           if not (0<=iy<h and 0<=ix<w):continue
           cta=((batch*nc+oc//bc)*ny+oy//by)*nx+ox//bx
           lane=((oy%by)*bc+oc%bc)*bx+ox%bx
           if v=='canonical':step=(ic0*k+ky)*k+kx
           else:step=(ic0*((by-1)*s+k)+(oy%by)*s+ky)*k+kx
           assert ti[cta,step,lane]==-1
           ti[cta,step,lane]=((batch*ci+ic)*h+iy)*w+ix
           to[cta,step,lane]=((batch*co+oc)*oh+oy)*ow+ox
           key=(batch,oc,ic0,oy,ox,ky,kx)
           assert key not in seen;seen.add(key)
    return ti,to,seen

def oracle_from_capture(ti,to):
    """Slow linear membership oracle, no shared optimized counter code."""
    counts=[0,0,0]
    for cta in range(ti.shape[0]):
     R=[[-1]*ti.shape[2],[-1]*ti.shape[2]];last=[-1,-1]
     for step in range(ti.shape[1]):
      for slot in range(ti.shape[2]):
       if ti[cta,step,slot]<0:continue
       for j,a in enumerate((int(ti[cta,step,slot]),int(to[cta,step,slot]))):
        old=R[j][slot] if a in R[j] else last[j]
        if a not in R[j]:last[j]=a
        counts[j]+=int(old>=0 and abs(a-old)>1);R[j][slot]=a
       counts[2]+=1
    return counts

SMALL_CASES=[
 dict(id='check_dense_valid',n=1,ci=2,co=3,h=7,w=9,kernel=3,stride=1,padding=0,groups=1),
 dict(id='check_dense_padded',n=2,ci=3,co=5,h=7,w=9,kernel=3,stride=2,padding=1,groups=1),
 dict(id='check_dw_padded',n=2,ci=3,co=3,h=7,w=9,kernel=5,stride=2,padding=2,groups=3),
 dict(id='check_pointwise',n=1,ci=3,co=5,h=5,w=7,kernel=1,stride=1,padding=0,groups=1)]


SMALL_CASES.append(dict(id='check_stencil7',n=1,ci=1,co=1,h=9,w=11,kernel=7,stride=1,padding=3,groups=1))
if __name__=='__main__':
    rows=[]
    for q in json.loads((ROOT/'configs.json').read_text()):
     for c in json.loads((ROOT/'cases.json').read_text()):
      pair=[]
      for v in ('canonical','reuse'):
       row=dict(case=c['id'],config=q['id'],variant=v,**measure(c,q,v));rows.append(row);pair.append(row)
       print(row,flush=True)
      assert pair[0]['useful_macs']==pair[1]['useful_macs']
      assert pair[0]['nominal_entries_per_operand_per_CTA']==pair[1]['nominal_entries_per_operand_per_CTA']
    (ROOT/'model_results.json').write_text(json.dumps(rows,indent=2))
