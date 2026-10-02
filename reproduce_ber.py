import numpy as np, json, sys, time
sys.path.insert(0,'/mnt/data')
import qaoa_fast as q

def full_edges(J):
    N=J.shape[0]
    return [(i,j) for i in range(N) for j in range(i+1,N) if abs(J[i,j])>1e-12]

rng=np.random.default_rng(20260923)
snrs=[0,5,10,15,20]
n=50
out={'snrs':snrs,'nchannels':n,'Nr':128,'Nu':4,'Nq':8,'p':2,'K':2,'shots':2000,
     'ML':[],'MMSE_SIC':[],'Full_QAOA':[],'Local_QAOA':[],'full_edges':28}
t0=time.time()
for snr in snrs:
    rows=[]
    for t in range(n):
        H=q.channel(Nr=128,Nu=4,rng=rng)
        sI=rng.choice([-1,1],4); sQ=rng.choice([-1,1],4)
        x=(sI+1j*sQ)/np.sqrt(2); st=np.r_[sI,sQ]
        y=q.make_observation(H,x,snr,rng)
        ml,h,J,c=q.ml_vector_cost(H,y); mm=q.mmse_sic(H,y)
        Ef=full_edges(J); El=q.local_edges(J,2)
        thF,_=q.optimize_qaoa_fast(h,J,2,Ef,seed=1000+t+snr,maxiter=40)
        psi=q.state_from(h,J,2,thF,Ef); prob=np.abs(psi)**2
        samp=rng.choice(len(prob),out['shots'],p=prob)
        bc=samp[np.argmin(c[samp])]; sf=q.basis_states(8)[bc]
        thL,_=q.optimize_qaoa_fast(h,J,2,El,seed=3000+t+snr,maxiter=40)
        psi=q.state_from(h,J,2,thL,El); prob=np.abs(psi)**2
        samp=rng.choice(len(prob),out['shots'],p=prob)
        bc=samp[np.argmin(c[samp])]; sl=q.basis_states(8)[bc]
        rows.append([np.mean(ml!=st),np.mean(mm!=st),np.mean(sf!=st),np.mean(sl!=st)])
    a=np.asarray(rows); means=a.mean(0)
    for k,v in zip(['ML','MMSE_SIC','Full_QAOA','Local_QAOA'],means): out[k].append(float(v))
    print(snr,means,flush=True)
json.dump(out,open('/mnt/data/final_small_results.json','w'),indent=2)
print('runtime',time.time()-t0)
