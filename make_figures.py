import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0,'/mnt/data'); import qaoa_fast as q
out=Path('/mnt/data/paper_figs')
R=json.load(open('/mnt/data/final_small_results.json'))
# Fig 1 pipeline schematic
fig,ax=plt.subplots(figsize=(10,3.4)); ax.axis('off')
boxes=[('Near-field\nXL-MIMO channel',0.0),('QPSK\nuplink',0.17),('Real-equivalent\nQUBO / Ising',0.34),('Top-K coupling\npruning',0.51),('p-level\nQAOA',0.68),('NISQ noise\n+ shots',0.84)]
for txt,x in boxes:
    ax.text(x,0.5,txt,ha='center',va='center',fontsize=10,
            bbox=dict(boxstyle='round,pad=0.55',fc='white',ec='0.2',lw=1.2),transform=ax.transAxes)
for x in [0.085,0.255,0.425,0.595,0.76]:
    ax.annotate('',xy=(x+0.07,0.5),xytext=(x,0.5),xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle='->',lw=1.2))
ax.text(0.5,0.08,'Classical post-processing: select the lowest full-QUBO energy among measured bit strings',ha='center',fontsize=9)
fig.tight_layout(); fig.savefig(out/'fig1_pipeline.png',dpi=300,bbox_inches='tight'); plt.close(fig)
# Fig 2 BER curve
fig,ax=plt.subplots(figsize=(7.2,4.2))
for key,label,marker in [('ML','Exact ML','o'),('MMSE_SIC','MMSE-SIC','s'),('Full_QAOA','Dense QAOA, p=2','^'),('Local_QAOA','Coupling-pruned QAOA, p=2, K=2','D')]:
    ax.semilogy(R['snrs'],np.maximum(R[key],1e-4),marker=marker,lw=1.8,ms=5,label=label)
ax.set_xlabel('SNR (dB)'); ax.set_ylabel('BER'); ax.set_title('Exact 8-qubit statevector validation (128 x 4 XL-MIMO)')
ax.grid(True,which='both',ls=':',alpha=.45); ax.legend(fontsize=8); ax.set_ylim(1e-3,5e-2)
fig.tight_layout(); fig.savefig(out/'fig2_ber.png',dpi=300,bbox_inches='tight'); plt.close(fig)
# Fig 3 coupling retention
rng=np.random.default_rng(333); N=16; ks=[1,2,3,4,6,8,10,12]
ret={k:[] for k in ks}
for _ in range(300):
    H=q.channel(Nr=128,Nu=8,rng=rng); sI=rng.choice([-1,1],8); sQ=rng.choice([-1,1],8); x=(sI+1j*sQ)/np.sqrt(2); y=q.make_observation(H,x,10,rng); _,h,J,c=q.ml_vector_cost(H,y)
    den=np.sum(np.abs(J[np.triu_indices(N,1)]))
    for k in ks:
        E=q.local_edges(J,k); ret[k].append(sum(abs(J[i,j]) for i,j in E)/den)
means=np.array([np.mean(ret[k]) for k in ks]); stds=np.array([np.std(ret[k],ddof=1) for k in ks])
fig,ax=plt.subplots(figsize=(7.2,4.2)); ax.errorbar(ks,means,yerr=1.96*stds/np.sqrt(300),marker='o',lw=1.8,capsize=3); ax.set_xlabel('Top-K retained neighbors per qubit'); ax.set_ylabel('Fraction of absolute coupling mass retained'); ax.set_ylim(0,1.05); ax.grid(True,ls=':',alpha=.45); ax.set_title('Coupling-mass retention across 300 channel realizations (N=16)'); fig.tight_layout(); fig.savefig(out/'fig3_retention.png',dpi=300,bbox_inches='tight'); plt.close(fig)
# Fig 4 noise sensitivity using saved numbers
NR=json.load(open('/mnt/data/noise_results.json')); noise={'pd':NR['pd'],'ber':NR['ber'],'se':NR['se']}
fig,ax=plt.subplots(figsize=(7.2,4.2)); ax.errorbar(noise['pd'],noise['ber'],yerr=1.96*np.array(noise['se']),marker='o',lw=1.8,capsize=3); ax.set_xlabel('Two-qubit depolarizing probability per entangling interaction'); ax.set_ylabel('BER'); ax.set_title('Stochastic gate-noise sensitivity (8 qubits, SNR = 10 dB)'); ax.grid(True,ls=':',alpha=.45); fig.tight_layout(); fig.savefig(out/'fig4_noise.png',dpi=300,bbox_inches='tight'); plt.close(fig)
# Fig 5 asymptotic envelopes
Nvals=np.arange(4,65,2)
global_v=2.0**(-Nvals); local_v=1.0/(Nvals**2); local2=1.0/(Nvals)
fig,ax=plt.subplots(figsize=(7.2,4.2)); ax.semilogy(Nvals,global_v,lw=2,label=r'Global cost envelope: $O(2^{-N})$'); ax.semilogy(Nvals,local_v,lw=2,label=r'Illustrative local envelope: $O(N^{-2})$'); ax.semilogy(Nvals,local2,lw=2,ls='--',label=r'Illustrative local envelope: $O(N^{-1})$'); ax.set_xlabel('Number of qubits, N'); ax.set_ylabel('Gradient-variance envelope (normalized)'); ax.set_title('Asymptotic trainability regimes from locality-based barren-plateau theory'); ax.grid(True,which='both',ls=':',alpha=.45); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'fig5_trainability.png',dpi=300,bbox_inches='tight'); plt.close(fig)
# Fig 6 resource scaling N=8..32 p=3; dense vs top-K upper bound
Ns=np.array([8,12,16,20,24,28,32]); p=3
fullE=Ns*(Ns-1)//2
K=2; localE=Ns*K # upper bound for union of top-K directed selections
fig,ax=plt.subplots(figsize=(7.2,4.2)); ax.plot(Ns,2*p*fullE,'o-',lw=2,label='Dense QAOA: CNOT-equivalent ZZ interactions'); ax.plot(Ns,2*p*localE,'s-',lw=2,label='Top-K (K=2) upper bound'); ax.set_xlabel('Logical qubits, N = 2N_u'); ax.set_ylabel('CNOT-equivalent two-qubit gates per QAOA layer stack (p = 3)'); ax.set_title('Two-qubit gate scaling before hardware routing'); ax.grid(True,ls=':',alpha=.45); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'fig6_resources.png',dpi=300,bbox_inches='tight'); plt.close(fig)
print('figures created',len(list(out.glob('*.png'))))
