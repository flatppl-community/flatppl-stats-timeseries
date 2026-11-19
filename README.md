# flatppl-stats-timeseries

[Function reference](docs/reference.md) · [Live example](https://live.flatppl.org/#model=https%3A%2F%2Fraw.githubusercontent.com%2Fflatppl-community%2Fflatppl-stats-timeseries%2Fmain%2Fexamples%2Fplayground.flatppl&target=Pi_post)

Gaussian random walks, AR, MA, ARMA, ARIMA, irregular-time OU laws and finite-state HMM scores for FlatPPL. Library version **0.1**.

Conditional ARMA paths satisfy

$$Y_t=\mu+\sum_{j=1}^p\phi_j(Y_{t-j}-\mu)+\sum_{j=1}^q\theta_j\epsilon_{t-j}+\epsilon_t,
\qquad \epsilon_t\sim\mathcal N(0,\sigma^2).$$

```flatppl
flatppl_compat = "0.1"
ts = load_module("main.flatppl")
path = ts.arma([1.2, -0.4], [0.7], 3, 0.4, [0.3, -0.2], [0.2], 0.7)
score = logdensityof(path, [0.2, -0.4, 0.7])
```

Path laws have support $\mathbb R^n$ and exclude supplied history, ordered newest first. Gaussian scales are positive standard deviations. Initial-history laws and priors are caller-supplied. HMM inputs have shapes `initial[K]`, `transition[K,K]`, `log_emissions[T,K]`; the result is a scalar log marginal likelihood.

See the [drift](examples/playground.flatppl) and [ARMA](examples/arma.flatppl) examples, and the reference for domains, OU diffusion, observation laws and validation.
