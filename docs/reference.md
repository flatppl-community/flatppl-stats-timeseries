# Time-series reference — 0.1

Load `main.flatppl` to access the following functions. Scales are standard deviations. Time units, initial conditions, priors, and observation processes are explicit choices of the caller.

| Function | Meaning |
|---|---|
| [`random_walk_step(previous, drift, sigma)`](#random-walk-step) | Gaussian increment law |
| [`random_walk(initial, n, drift, sigma)`](#random-walk) | Homogeneous Gaussian walk |
| [`ar1_step(previous, mean, rho, sigma)`](#ar1-step) | Mean-centered AR(1) transition |
| [`ar1(initial, n, mean, rho, sigma)`](#ar1) | Homogeneous AR(1) path |
| [`ou_step(previous, dt, mean, rate, diffusion)`](#ou-step) | Exact Ornstein–Uhlenbeck transition |
| [`ou(initial, dts, mean, rate, diffusion)`](#ou) | Irregular-time OU path |
| [`gaussian_observations(states, offset, sigma)`](#gaussian-observations) | Independent Gaussian errors |
| [`poisson_observations(log_rates, exposure)`](#poisson-observations) | Independent exposed Poisson counts |
| [`hmm_loglikelihood(initial, transition, log_emissions)`](#hmm-loglikelihood) | Finite-state marginal likelihood |

## Path and conditioning conventions

A path contains the states after transitions, excluding the supplied initial state. For `random_walk` and `ar1`, positive integer `n` is both the transition count and returned path length. An OU path has `lengthof(dts)` states, with `dts` the positive elapsed intervals before those states. A path law scores each transition once and supplies no density or prior for the initial value. All scalar inputs are finite unless a log emission is explicitly allowed to be `-inf`.

Path constructors use `markovchain` for homogeneous chains and `kscan` for varying time intervals. Parameters enter callbacks in explicit records. Observation helpers condition on supplied state values. Combining a path with noisy observations does not analytically integrate the states; a caller must retain the latent states or supply a separate marginalization method. The finite-state HMM helper performs its stated exact discrete marginalization.

All Gaussian laws are normalized with Lebesgue reference measure. Poisson laws use counting measure and retain factorial constants. There is no implicit prior, clipping, variance floor, or missing-observation convention.

<a id="random-walk-step"></a>
## `random_walk_step(previous, drift, sigma)`

Returns a scalar transition law for finite real `previous`, `drift`, and `sigma>0`:

```math
X'\mid X=x\sim N(x+\delta,\sigma^2).
```

Drift and variance are per step. A varying physical time interval requires an explicit rescaling of both the drift and noise amplitude.

<a id="random-walk"></a>
## `random_walk(initial, n, drift, sigma)`

Returns a normalized law on real vectors `[n]`, with positive integer `n` and the domains of [`random_walk_step`](#random-walk-step). For fixed `initial=x_0`,

```math
E[X_i]=x_0+i\delta,\qquad
\operatorname{Cov}(X_i,X_j)=\sigma^2\min(i,j),\qquad 1\le i,j\le n.
```

The path density is the product of increment densities; observations of every state can therefore be scored through the independent increments.

<a id="ar1-step"></a>
## `ar1_step(previous, mean, rho, sigma)`

Returns a scalar transition law for finite real `previous`, `mean`, `rho`, and `sigma>0`:

```math
X'\mid X=x\sim N\bigl(m+\rho(x-m),\sigma^2\bigr).
```

`mean` is the equilibrium mean when `abs(rho)<1`; it is not the intercept. A model written `intercept + rho*x` has `intercept=(1-rho)*mean`. Any finite `rho` defines the transition. Stationarity requires `abs(rho)<1`, with stationary variance `sigma^2/(1-rho^2)`. Negative correlation is allowed.

<a id="ar1"></a>
## `ar1(initial, n, mean, rho, sigma)`

Returns a normalized conditional path law on real vectors `[n]`, with positive integer `n` and the domains of [`ar1_step`](#ar1-step).

```math
X_i=m+\rho^i(x_0-m)+\sum_{k=1}^i\rho^{i-k}\epsilon_k,
\qquad \epsilon_k\overset{\mathrm{iid}}\sim N(0,\sigma^2).
```

The initial state is fixed, so this is not a stationary initial-state model. To model a stationary series, explicitly draw the initial state from the stationary law and include that initial density once in the joint model.

<a id="ou-step"></a>
## `ou_step(previous, dt, mean, rate, diffusion)`

Returns the exact scalar transition of the SDE

```math
dX_t=\lambda(m-X_t)\,dt+q\,dW_t,
```

with finite real `previous`, `mean`, and strictly positive `dt`, `rate=lambda`, `diffusion=q`:

```math
X_{t+h}\mid X_t=x\sim N\left(
m+(x-m)e^{-\lambda h},\quad
\frac{q^2}{2\lambda}(1-e^{-2\lambda h})
\right).
```

`diffusion` is the SDE noise amplitude, not a transition standard deviation or stationary standard deviation. The stationary variance is `diffusion^2/(2*rate)`. The implementation uses `expm1` for the mean increment and transition variance to retain short-interval precision. Arguments must keep the resulting positive variance representable.

`dt=0` gives a point mass and `rate=0` gives a Brownian limit; both lie outside this helper's domain. For a Brownian interval use `Normal(previous, diffusion*sqrt(dt))` explicitly. No branch silently substitutes either limiting law.

<a id="ou"></a>
## `ou(initial, dts, mean, rate, diffusion)`

Returns a normalized conditional path law on real vectors of length `lengthof(dts)`. The interval vector is nonempty with finite strictly positive entries. Each transition uses [`ou_step`](#ou-step), conditional on the preceding sampled state and the corresponding elapsed interval.

For absolute times `t_i=sum(dts[1:i])` after a fixed `x_0`,

```math
E[X_{t_i}]=m+(x_0-m)e^{-\lambda t_i},\qquad
\operatorname{Cov}(X_{t_i},X_{t_j})=
\frac{q^2}{2\lambda}\left(e^{-\lambda|t_i-t_j|}-e^{-\lambda(t_i+t_j)}\right).
```

The intervals are exogenous inputs, not additional random variables. An irregular calendar is represented by elapsed intervals rather than absolute timestamps.

<a id="gaussian-observations"></a>
## `gaussian_observations(states, offset, sigma)`

Returns a normalized independent array law with the broadcast shape of `states`, `offset`, and `sigma`:

```math
Y_i\mid x_i\sim N(x_i+b_i,s_i^2).
```

Inputs are finite and aligned; every `sigma` entry is strictly positive. A scalar offset or scale is shared across observations. Aligned vectors permit known observation-specific offsets or scales. This law does not add temporal covariance to the errors or integrate unknown states.

<a id="poisson-observations"></a>
## `poisson_observations(log_rates, exposure)`

Returns a normalized independent count-array law with aligned finite real log rates and finite nonnegative exposures:

```math
Y_i\mid\eta_i\sim\operatorname{Poisson}(E_i e^{\eta_i}).
```

The caller supplies log rates per unit exposure. A scalar exposure broadcasts across observations. Exponentiation and the resulting means must be representable; zero exposure is a point mass at zero. A zero observed count remains in the likelihood. Exposures enter once and are not also included in `log_rates`.

<a id="hmm-loglikelihood"></a>
## `hmm_loglikelihood(initial, transition, log_emissions)`

Returns a scalar log marginal likelihood. `initial=pi[K]` is the probability distribution of the hidden state at the **first observation**, not one step before it. `transition=A[K,K]` is row-stochastic, with `A[i,j]=P(Z_t=j|Z_{t-1}=i)`. `log_emissions=L[T,K]` contains the log emission probabilities or log densities, with time first and state second. There is no transition before the first emission.

```math
\log p(y_{1:T})=\log\sum_{z_1,\ldots,z_T}
\pi_{z_1}e^{L_{1,z_1}}
\prod_{t=2}^T A_{z_{t-1},z_t}e^{L_{t,z_t}}.
```

Domain: positive integer `K`, nonnegative finite `pi` summing to one, nonnegative finite rows of `A` each summing to one, and finite or `-inf` log emissions. All states' emission densities at one time must use the same reference measure. Normalization of `pi`, transition rows, and emissions is the caller's responsibility; the helper does not renormalize supplied values.

The implementation performs a deterministic log-domain forward `scan` with explicit shared inputs. At each transition it uses a separate maximum shift per destination. Exact zeros and unreachable states are preserved. Impossible sequences return `-inf`; derivatives of an impossible sequence's log likelihood are undefined. The cost is `O(T*K^2)` and forward states are retained by `scan`.

One state reduces to the sum of its log emissions. One observation reduces to `logsumexp(log.(initial) .+ log_emissions[1,:])`. An empty emission matrix `zeros([0,K])` has log likelihood zero. To omit an observation under an ignorable missingness model, supply a row of zero log emissions; any transition before that time is still applied. This helper returns a score, not a sampling law or posterior state probabilities.

## Validation

The public mathematical tests compare path densities with dense multivariate Gaussian laws built from independent innovation maps or the OU covariance integral. The HMM oracle enumerates every hidden sequence on short examples, including structural zeros, impossible data, empty data, one state, and three states. `tests/runtime.flatppl` exposes values for every public helper; `tests/oracles.py` supplies independent dense-law values and complex-step Jacobians.
