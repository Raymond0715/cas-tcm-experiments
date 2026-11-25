"""Render the two Figure 6 panels from a complete paired timing run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, default=ROOT / "reference" / "results.json")
parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "figure6")
args = parser.parse_args()

rows = []
for pair in json.loads(args.results.read_text()):
    canonical = pair["tcm"]["canonical"]
    reuse = pair["tcm"]["reuse"]
    timings = pair["stats"]
    rows.append(dict(family=pair["case"]["family"], config=pair["config"]["id"],
                     tcm_ratio=canonical["total_tcm"] / reuse["total_tcm"],
                     speedup=timings["canonical"]["median_us"] / timings["reuse"]["median_us"]))
if len(rows) != 22:
    parser.error("Figure 6 requires the complete set of 22 paired results")
args.output_dir.mkdir(parents=True, exist_ok=True)

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'sans-serif','font.size':7,'axes.labelsize':7.5,
                     'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.6,
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
colors={'dense':'#2672b5','depthwise':'#d27120','pointwise':'#77549b'}
for config in ['y4x8c4','y4x16c4']:
 fig,ax=plt.subplots(figsize=(3.48,1.8))
 fig.subplots_adjust(left=.16,right=.98,bottom=.26,top=.80)
 ax.add_patch(Rectangle((.5,.45),.5,.55,facecolor='#f0f6f0',edgecolor='none',zorder=0))
 ax.axvline(1,color='#8a8a8a',ls=(0,(3,3)),lw=.65)
 ax.axhline(1,color='#8a8a8a',ls=(0,(3,3)),lw=.65)
 for family,color in colors.items():
  sub=[r for r in rows if r['family']==family and r['config']==config]
  assert len(sub)=={'dense':6,'depthwise':4,'pointwise':1}[family]
  ax.scatter([1/r['tcm_ratio'] for r in sub],[1/r['speedup'] for r in sub],s=19,c=color,marker='o',edgecolors='white',linewidths=.4,zorder=3)
 ax.plot(1,1,'*',color='#222222',ms=6.5,zorder=4)
 ax.text(.99,1.035,'Canonical (1, 1)',ha='right',va='bottom',fontsize=6.5,color='#333333')
 ax.set_xlim(.5,1.04);ax.set_ylim(.45,1.15)
 ax.set_xticks([.6,.8,1],['0.6','0.8','1.0'])
 ax.set_yticks([.5,.75,1],['0.50','0.75','1.00'])
 ax.set_xlabel('Normalized TCM',labelpad=2)
 ax.set_ylabel('Normalized time',labelpad=3)
 ax.spines[['top','right']].set_visible(False)
 ax.tick_params(length=2.5,width=.6,pad=2)
 handles=[Line2D([],[],marker='o',ls='',color=c,label=f.capitalize(),markersize=3.5) for f,c in colors.items()]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.56,1.01),ncol=3,frameon=False,columnspacing=1.1,handletextpad=.35)
 fig.savefig(args.output_dir/('paired_'+config+'.png'),dpi=240)
 fig.savefig(args.output_dir/('paired_'+config+'.pdf'))
 fig.savefig(args.output_dir/('paired_'+config+'.svg'))
 plt.close(fig)
