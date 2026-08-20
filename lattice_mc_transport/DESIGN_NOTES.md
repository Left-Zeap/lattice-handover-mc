# Design notes

## State
Each Monte Carlo sample is one atom:
`r[N,3], v[N,3], alive[N], unbound_time[N], scatter_count[N]`.

## Stage unification
L1, handover and L2 use the same velocity-Verlet propagator.
Only `RuntimeLattice` changes with time.

## Microscopic forces
- Conservative: `F = -grad(U_AC)`
- Gravity: `m g`
- Dissipative stochastic: off-resonant scattering -> recoil quantum jump

## Handover
During 1 ms:
`P1(t)=P1*(1-t/tau)`, `P2(t)=P2*t/tau`.
No analytic handover-heating term is used.

## Survival
Transport uses local comoving excitation energy vs the acceleration-reduced
escape barrier, with a grace time and 20% energy margin by default.
Handover survival is assessed only after the full time-dependent two-lattice
trajectory has been propagated.

## Temperature
Computed only at the ensemble statistics layer from velocity covariance after
subtracting center-of-mass velocity.
