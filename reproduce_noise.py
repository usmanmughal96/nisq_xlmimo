import numpy as np, json, sys, time
sys.path.insert(0,'/mnt/data')
import qaoa_fast as q
rng=np.random.default_rng(90210)
settings={'nchannels':100,'trajectories':20,'shots':500,'snr_db':10.0,'p':1,'K':2,'Nr':128,'Nu':4,'pd':[0.0,0.001,0.005,0.01,0.02]}
out={**settings,'ber':[],'se':[],'ci95':[]}
t0=time.time()
for pd in settings['pd']:
    per=[]
    for t in range(settings['nchannels']):
        H=q.channel(Nr=128,Nu=4,rng=rng)
        sI=rng.choice([-1,1],4); sQ=rng.choice([-1,1],4)
        x=(sI+1j*sQ)/np.sqrt(2); st=np.r_[sI,sQ]
        y=q.make_observation(H,x,settings['snr_db'],rng)
        ml,h,J,c=q.ml_vector_cost(H,y); E=q.local_edges(J,settings['K'])
        th,_=q.optimize_qaoa_fast(h,J,settings['p'],E,seed=12000+t,maxiter=30)
        vals=[]
        for rr in range(settings['trajectories']):
            psi=q.noisy_local_state(h,J,settings['p'],th,E,pd=pd,rng=rng,N=8)
            prob=np.abs(psi)**2
            samp=rng.choice(len(prob),settings['shots'],p=prob)
            bc=samp[np.argmin(c[samp])]
            vals.append(np.mean(q.basis_states(8)[bc]!=st))
        per.append(np.mean(vals))
    per=np.asarray(per); mean=float(per.mean()); se=float(per.std(ddof=1)/np.sqrt(len(per))); ci=1.96*se
    out['ber'].append(mean); out['se'].append(se); out['ci95'].append([mean-ci,mean+ci])
    print(pd,mean,se,flush=True)
json.dump(out,open('/mnt/data/noise_results.json','w'),indent=2)
print('runtime',time.time()-t0)
