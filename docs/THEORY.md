# Theory notes

A compact map of the ideas behind each module, what the validation suite checks, and where to read more.

## Models and realisations (`systems`, `polynomials`)

A linear time-invariant system is x′ = Ax + Bu, y = Cx + Du or, for one input and output, the rational
function G(s) = C(sI − A)⁻¹B + D. A transfer function has infinitely many realisations related by similarity
transformations; the controllable and observable canonical forms are built from the coefficients, the modal
form diagonalises A, and a realisation is minimal when it is both controllable and observable - otherwise
modes cancel in the transfer function while remaining in the state (dangerous when unstable). The transfer
function of a realisation is computed by the Leverrier-Faddeev recursion for the characteristic polynomial
and the adjugate. Interconnections (series, parallel, feedback) are polynomial algebra or block matrices.
*Kailath, Linear Systems (1980); Chen, Linear System Theory and Design (1999).*

## Time and frequency response (`response`, `frequency`)

x(t) = e^{At}x₀ + ∫e^{A(t−τ)}Bu(τ)dτ; for a piecewise-constant input the response is exact at the samples with
Φ = e^{Ah} and Γ = ∫₀ʰ e^{At}dt B, both from one exponential of the augmented matrix (Van Loan 1978). The
second-order prototype's overshoot exp(−πζ/√(1−ζ²)) and peak time π/ω_d are the vocabulary of specifications.
G(jω) is the sinusoidal steady state; on Bode's logarithmic axes factors add, each pole or zero contributing
a corner of ±20 dB/decade and 90° of phase; the Nyquist plot draws G(jω) in the complex plane and the margins
are its distances from −1 (gain margin along the real axis, phase margin along the unit circle, and the single
honest number, the closest approach 1/M_s). The H∞ norm is computed by bisection on the Hamiltonian
(Boyd, Balakrishnan & Kabamba 1989). *Franklin, Powell & Emami-Naeini, Feedback Control of Dynamic Systems;
Åström & Murray, Feedback Systems (2008).*

## Stability (`stability`)

A system is asymptotically stable when every eigenvalue of A has negative real part. The Routh array
decides this from the coefficients and counts unstable roots (sign changes in its first column), with the ε
rule for a zero first entry and the auxiliary polynomial for a row of zeros; Hurwitz's determinants are the
same test; Jury's table is the discrete-time counterpart; Lyapunov's equation AᵀP + PA = −Q has a positive
definite solution iff A is stable. Evans' root locus plots the roots of 1 + KG(s) = 0 as K varies, with rules
for asymptotes (centroid, angles), real-axis segments, breakaway points (N′D − ND′ = 0) and departure angles.
The Nyquist criterion rests on the argument principle: the clockwise encirclements of −1 by L along the
Nyquist contour (indented around poles on the axis) plus the open-loop unstable poles give the closed-loop
unstable poles, Z = N + P. *Routh 1877; Evans 1948; Nyquist 1932; Ogata, Modern Control Engineering.*

## Classical design (`design`)

PID control: proportional for speed, integral for zero offset, derivative for damping, with a filter on the
derivative and the derivative on the measurement to avoid the setpoint kick; anti-windup by clamping or
back-calculation when the actuator saturates. Tuning rules for first-order-plus-dead-time models
(Ziegler-Nichols 1942, Cohen-Coon 1953, SIMC - Skogestad 2003) and the relay autotuner (Åström & Hägglund
1984), whose describing-function estimate of the ultimate gain is within a few percent. Lead compensation
adds phase at the crossover ((1 − sin φ)/(1 + sin φ) for the pole-zero ratio), lag raises low-frequency gain
with its zero a decade below the crossover; loop shaping trades bandwidth against control effort and noise
amplification. *Åström & Hägglund, Advanced PID Control (2006).*

## State-space design (`statespace`)

Controllability (rank of [B, AB, …], the PBH test) is the condition for placing every closed-loop pole with
u = −Kx; observability (its dual) for reconstructing the state. Ackermann's formula gives the single-input
gain; for several inputs the Sylvester-equation method (Bhattacharyya & de Souza 1982) chooses one of the
many. The Luenberger observer x̂′ = Ax̂ + Bu + L(y − Cx̂) has error dynamics A − LC; with state feedback on x̂
the closed loop has the eigenvalues of A − BK and A − LC together (the separation principle). The
linear-quadratic regulator minimises ∫(xᵀQx + uᵀRu)dt with K = R⁻¹BᵀP from the algebraic Riccati equation and
guarantees a gain margin of [½, ∞) and 60° of phase margin at the plant input (Kalman's return-difference
inequality); the Kalman filter is its dual with the noise covariances as weights; LQG combines them and has
no guaranteed margins (Doyle 1978), partially recoverable by loop transfer recovery (Doyle & Stein 1979).
*Anderson & Moore, Optimal Control (1990); Kwakernaak & Sivan (1972).*

## Discrete-time control (`discrete`)

A zero-order hold makes the plant exactly x_{k+1} = Φx_k + Γu_k with poles at z = e^{sΔt} (and sampling zeros
that can lie outside the unit circle, Åström, Hagander & Sternby 1984); Tustin's transform s = (2/Δt)(z−1)/(z+1)
preserves stability and warps frequency by (2/Δt)tan(ωΔt/2), corrected at one frequency by prewarping; the
hold itself is a half-sample delay, which with ω_cΔt ≈ 0.2-0.6 sets the sampling rate; aliasing folds
frequencies above π/Δt. Dead-beat control places all poles at z = 0. *Åström & Wittenmark, Computer-Controlled
Systems (1997); Franklin, Powell & Workman, Digital Control of Dynamic Systems.*

## Robustness and limitations (`robust`)

The gang of four S, T, PS, CS describe a loop; S + T = 1; M_s = max|S| bounds both classical margins and is
the inverse of the Nyquist curve's closest approach to −1. Bode's integral ∫ln|S|dω = π Σ Re p_k for unstable
poles p_k (Bode 1945, Freudenberg & Looze 1985) is the waterbed effect. A RHP zero z limits the crossover to
about z/2, an unstable pole p demands about 2p, a delay L limits it to 1/L, and p near z forces
M_s ≥ |p + z|/|p − z|. The small-gain theorem gives robust stability against multiplicative uncertainty when
|WT| < 1. *Skogestad & Postlethwaite, Multivariable Feedback Control (2005); Doyle, Francis & Tannenbaum (1992).*

## Identification and realisation (`identify`, `mimo`)

ARX least squares is exact for equation-error noise and biased for output noise (the regressor contains the
noisy outputs), the more so near z = 1; instrumental variables are consistent; output-error minimisation of
the simulation error is unbiased for output noise. Levy's linearised frequency-domain fit is reweighted by
Sanathanan-Koerner. The Hankel matrix of Markov parameters has rank equal to the order and yields a
realisation by its SVD (Ho & Kalman 1966; Juang & Pappa's ERA 1985); the Hankel singular values order the
balanced realisation (Moore 1981) and bound the truncation error by twice the discarded sum (Glover 1984,
Enns 1984). The relative gain array (Bristol 1966) measures interaction and recommends a pairing; singular
values give direction-dependent gain. *Ljung, System Identification (1999); Skogestad & Postlethwaite.*

## Numerics (`linalg`)

The matrix exponential by scaling and squaring with a [13/13] Padé approximant (Higham 2005); the real Schur
form by the shifted QR algorithm and Bartels-Stewart (1972) for Lyapunov equations; the Riccati equation from
the Hamiltonian's stable invariant subspace (Laub 1979) refined by Newton-Kleinman (1968); the discrete
equation by the structure-preserving doubling algorithm (Chu, Fan & Lin 2004). Polynomial coefficients and
companion forms are ill-conditioned at high order (Wilkinson 1963), and a balanced realisation computed
through ill-conditioned Gramians can be worse still: representations are only as good as the arithmetic that
produced them. *Golub & Van Loan, Matrix Computations (2013); Laub, Matrix Analysis for Scientists and
Engineers (2005).*
