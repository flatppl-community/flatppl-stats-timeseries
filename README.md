# flatppl-stats-timeseries

[Function reference](docs/reference.md) · [Live example](https://live.flatppl.org/#model=https%3A%2F%2Fraw.githubusercontent.com%2Fflatppl-community%2Fflatppl-stats-timeseries%2Fmain%2Fexamples%2Fplayground.flatppl&target=Pi_post)

Gaussian random walks, AR, MA, ARMA, ARIMA, and irregular-time Ornstein–Uhlenbeck laws for FlatPPL. The library also provides observation laws and finite-state HMM likelihoods. Library version **0.1**.

```flatppl
flatppl_compat = "0.1"
ts = load_module("main.flatppl")
path = ts.arma([1.2, -0.4], [0.7], 3, 0.4, [0.3, -0.2], [0.2], 0.7)
score = logdensityof(path, [0.2, -0.4, 0.7])
```

Paths exclude the supplied history, which runs from most recent to oldest. Gaussian scales are standard deviations. Parameters, priors, and initial-history laws remain explicit.

The [drift example](examples/playground.flatppl) infers two parameters from eight observed states. The [ARMA example](examples/arma.flatppl) exposes a conditional path likelihood.

See the reference for definitions, domains, numerical limits, and validation commands.
