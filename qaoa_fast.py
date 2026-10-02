import numpy as np
from scipy.optimize import minimize
from itertools import product

BASIS_CACHE={}
def basis_states(N):
    if N not in BASIS_CACHE:
        idx=np.arange(1<<N,dtype=np.uint32)
        # bit 0 corresponds first qubit; spin +1 for bit 0
        S=1.0-2.0*((idx[:,None]>>np.arange(N,dtype=np.uint32))&1).astype(np.float64)
        BASIS_CACHE[N]=S.astype(np.float64)
    return BASIS_CACHE[N]

def apply_mixer(psi,beta,N):
    out=psi
    cb=np.cos(beta); sb=-1j*np.sin(beta)
    for q in range(N):
        # pair with stride 2^q using reshape
        out=out.reshape(-1,2,1<<q) if False else out
        # vectorized reshape where q is middle bit: (blocks,2,sub)
        sub=1<<q; block=sub*2
        a=out.reshape(-1,block)
        a=a.reshape(-1,2,sub)
        v0=a[:,0,:].copy(); v1=a[:,1,:].copy()
        a[:,0,:]=cb*v0+sb*v1
        a[:,1,:]=sb*v0+cb*v1
        out=a.reshape(-1)
    return out

def costs_from_hJ(h,J,N):
    S=basis_states(N)
    c=S@h
    c += 0.5*np.sum((S@J)*S,axis=1)
    return c

def local_edges(J,k):
    N=J.shape[0]; E=set()
    for i in range(N):
      inds=np.argsort(np.abs(J[i]))[::-1]
      cnt=0
      for j in inds:
        if j==i or abs(J[i,j])<1e-12: continue
        a,b=sorted((i,j)); E.add((a,b)); cnt+=1
        if cnt>=k: break
    return sorted(E)

def optimize_qaoa(h,J,p,edges,seed=0,maxiter=100):
    N=len(h); S=basis_states(N)
    c=S@h
    for i,j in edges: c += J[i,j]*S[:,i]*S[:,j]
    dim=1<<N
    def obj(th):
        psi=np.ones(dim,dtype=np.complex128)/np.sqrt(dim)
        for l in range(p):
            gamma=th[2*l]; beta=th[2*l+1]
            psi*=np.exp(-1j*gamma*c)
            psi=apply_mixer(psi,beta,N)
        return float(np.dot(np.abs(psi)**2,c))
    rg=np.random.default_rng(seed)
    best=None
    for r in range(2):
      x0=np.zeros(2*p)
      x0[::2]=rg.uniform(-1,1,p); x0[1::2]=rg.uniform(-1,1,p)
      res=minimize(obj,x0,method='COBYLA',options={'maxiter':maxiter,'rhobeg':0.5,'tol':1e-3})
      if best is None or res.fun<best.fun: best=res
    return best.x,c

def state_from(h,J,p,theta,edges):
    N=len(h); S=basis_states(N); c=S@h
    for i,j in edges: c += J[i,j]*S[:,i]*S[:,j]
    psi=np.ones(1<<N,dtype=np.complex128)/np.sqrt(1<<N)
    for l in range(p):
        psi*=np.exp(-1j*theta[2*l]*c)
        psi=apply_mixer(psi,theta[2*l+1],N)
    return psi

def ml_vector_cost(H,y):
    Nu=H.shape[1]; N=2*Nu; S=basis_states(N)
    Ht=(1/np.sqrt(2))*np.block([[H.real,-H.imag],[H.imag,H.real]])
    yt=np.r_[y.real,y.imag]
    R=Ht.T@yt; Q=Ht.T@Ht; h=-2*R; J=2*Q; np.fill_diagonal(J,0)
    c=costs_from_hJ(h,J,N)
    k=np.argmin(c)
    return S[k],h,J,c

def channel(Nr=128,Nu=8,fc=140e9,rng=None):
    if rng is None: rng=np.random.default_rng()
    c0=299792458.; lam=c0/fc; d=lam/2; xr=(np.arange(Nr)-(Nr-1)/2)*d
    H=np.zeros((Nr,Nu),complex)
    baseR=rng.uniform(4,6); baseT=np.deg2rad(rng.uniform(-8,8))
    for u in range(Nu):
        R=baseR+rng.normal(0,.3); th=baseT+rng.normal(0,np.deg2rad(2.5)); xs=R*np.sin(th); zs=R*np.cos(th)
        dist=np.sqrt((xr-xs)**2+zs**2)
        # LOS + two weak geometric reflections from virtual sources
        hu=np.exp(-1j*2*np.pi*dist/lam)/dist
        for gain,dx,dz in [(0.28,0.25,0.8),(0.16,-0.35,1.6)]:
            dist2=np.sqrt((xr-(xs+dx))**2+(zs+dz)**2)
            hu += gain*np.exp(1j*rng.uniform(-np.pi,np.pi))*np.exp(-1j*2*np.pi*dist2/lam)/dist2
        hu*=np.sqrt(Nr)/np.linalg.norm(hu); H[:,u]=hu
    return H

def make_observation(H,x,snr_db,rng):
    Nr=H.shape[0]; sig=np.linalg.norm(H,'fro')**2/Nr; sigma2=sig/10**(snr_db/10)
    n=np.sqrt(sigma2/2)*(rng.standard_normal(Nr)+1j*rng.standard_normal(Nr))
    return H@x+n

def mmse_sic(H,y,reg=0.1):
    Hc=H.copy(); yc=y.copy(); cols=list(range(H.shape[1])); xs=np.zeros(H.shape[1],complex)
    while cols:
      A=Hc.conj().T@Hc+reg*np.eye(Hc.shape[1]); z=np.linalg.solve(A,Hc.conj().T@yc); k=np.argmax(np.abs(z));
      xs[cols[k]]=(np.sign(z[k].real)+1j*np.sign(z[k].imag))/np.sqrt(2)
      yc=yc-Hc[:,k]*xs[cols[k]]; Hc=np.delete(Hc,k,axis=1); cols.pop(k)
    return np.r_[np.sign(xs.real),np.sign(xs.imag)].astype(int)

def run(n=5):
    rng=np.random.default_rng(77)
    for snr in [0,5,10,15,20]:
      rows=[]
      for t in range(n):
        H=channel(rng=rng); Nu=8; sI=rng.choice([-1,1],Nu); sQ=rng.choice([-1,1],Nu); x=(sI+1j*sQ)/np.sqrt(2); st=np.r_[sI,sQ]
        y=make_observation(H,x,snr,rng)
        ml,h,J,c=ml_vector_cost(H,y)
        mm=mmse_sic(H,y)
        E=local_edges(J,2)
        th,cL=optimize_qaoa(h,J,2,E,seed=100+t,maxiter=60)
        psi=state_from(h,J,2,th,E); prob=np.abs(psi)**2
        # best-of-2000 full-cost sampling
        samp=rng.choice(len(prob),2000,p=prob)
        bc=samp[np.argmin(c[samp])]; S=basis_states(16); q=S[bc]
        rows.append([np.mean(ml!=st),np.mean(mm!=st),np.mean(q!=st),np.linalg.cond(H)])
      print(snr,np.mean(rows,axis=0))
if __name__=='__main__': run(5)

def apply_pauli(psi,q,kind,N):
    out=psi.copy(); bit=1<<q
    if kind=='I': return out
    if kind=='Z':
        idx=np.arange(1<<N); out *= np.where((idx&bit)!=0,-1.0,1.0); return out
    # X or Y swap amplitudes
    sub=1<<q; a=out.reshape(-1,2,sub); v0=a[:,0,:].copy(); v1=a[:,1,:].copy()
    if kind=='X':
        a[:,0,:]=v1; a[:,1,:]=v0
    else:
        a[:,0,:]=-1j*v1; a[:,1,:]=1j*v0
    return a.reshape(-1)

def noisy_local_state(h,J,p,theta,edges,pd=0.0,rng=None,N=None):
    if rng is None: rng=np.random.default_rng()
    if N is None: N=len(h)
    S=basis_states(N); dim=1<<N
    psi=np.ones(dim,dtype=np.complex128)/np.sqrt(dim)
    # exact cost decomposition into fields and ZZ interactions, with 2Q Pauli noise after ZZ gates
    for l in range(p):
        gamma=theta[2*l]; beta=theta[2*l+1]
        # fields as 1Q Z rotations
        for i in range(N):
            psi *= np.exp(-1j*gamma*h[i]*S[:,i])
        for i,j in edges:
            psi *= np.exp(-1j*gamma*J[i,j]*S[:,i]*S[:,j])
            if pd>0 and rng.random()<pd:
                z=int(rng.integers(1,16))
                # 2-qubit Pauli encoded bits x_i,z_i,x_j,z_j through 0..15; choose uniformly among 15
                codes=[('I','I'),('X','I'),('Y','I'),('Z','I'),('I','X'),('X','X'),('Y','X'),('Z','X'),('I','Y'),('X','Y'),('Y','Y'),('Z','Y'),('I','Z'),('X','Z'),('Y','Z'),('Z','Z')]
                a,b=codes[z]; psi=apply_pauli(psi,i,a,N); psi=apply_pauli(psi,j,b,N)
        psi=apply_mixer(psi,beta,N)
    return psi

def readout_confusion_inverse(prob,e=0.02,N=None):
    if N is None: N=int(np.log2(len(prob)))
    M=np.array([[1-e,e],[e,1-e]],float)
    T=M
    for _ in range(N-1): T=np.kron(T,M)
    est=np.linalg.solve(T,prob)
    est=np.maximum(est,0); s=est.sum(); return est/s if s>0 else prob


def optimize_qaoa_fast(h,J,p,edges,seed=0,maxiter=30):
    N=len(h); S=basis_states(N); c=S@h
    for i,j in edges: c += J[i,j]*S[:,i]*S[:,j]
    dim=1<<N
    def obj(th):
        psi=np.ones(dim,dtype=np.complex128)/np.sqrt(dim)
        for l in range(p):
            psi*=np.exp(-1j*th[2*l]*c)
            psi=apply_mixer(psi,th[2*l+1],N)
        return float(np.dot(np.abs(psi)**2,c))
    rg=np.random.default_rng(seed); x0=np.empty(2*p); x0[::2]=rg.uniform(-0.6,0.6,p); x0[1::2]=rg.uniform(-0.6,0.6,p)
    res=minimize(obj,x0,method='COBYLA',options={'maxiter':maxiter,'rhobeg':0.4,'tol':2e-3})
    return res.x,c
