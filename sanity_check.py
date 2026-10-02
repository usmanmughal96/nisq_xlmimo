"""Sanity check for the exact QPSK -> real binary -> Ising mapping."""
import numpy as np
import sys
sys.path.insert(0, str(__file__).rsplit('/code/',1)[0] + '/code')
import qaoa_fast as q

rng = np.random.default_rng(1234)
Nr, Nu = 10, 3
H = rng.normal(size=(Nr,Nu)) + 1j*rng.normal(size=(Nr,Nu))
sI = rng.choice([-1,1],Nu); sQ = rng.choice([-1,1],Nu)
x=(sI+1j*sQ)/np.sqrt(2)
y=H@x + 0.1*(rng.normal(size=Nr)+1j*rng.normal(size=Nr))
ml_s,h,J,c=q.ml_vector_cost(H,y)
S=q.basis_states(2*Nu)
idx=np.argmin(c)
s= S[idx]
# Direct complex ML enumeration
best=None; best_s=None
for bits in S:
    si=bits[:Nu]; sq=bits[Nu:]
    xb=(si+1j*sq)/np.sqrt(2)
    cost=np.linalg.norm(y-H@xb)**2
    if best is None or cost<best:
        best=float(cost); best_s=bits.copy()
direct=np.linalg.norm(y-H@((best_s[:Nu]+1j*best_s[Nu:])/np.sqrt(2)))**2
ising= c[idx] + np.vdot(y,y).real + np.trace((H.T.conj()@H).real)
print('QUBO/Ising minimizer matches direct complex ML:', np.array_equal(s,best_s))
print('absolute objective reconstruction error:', abs(direct-ising))
assert np.array_equal(s,best_s)
assert abs(direct-ising) < 1e-8
