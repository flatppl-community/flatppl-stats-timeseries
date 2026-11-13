# flatppl-stats-timeseries

Gaussian random-walk, AR(1), and irregular-time Ornstein–Uhlenbeck laws for FlatPPL 0.1, independent observation laws, and exact finite-state hidden Markov log likelihoods.

```flatppl
flatppl_compat = "0.1"
ts = load_module("main.flatppl")
path = ts.random_walk(0, 3, 0.2, 0.7)
score = logdensityof(path, [0.1, 0.5, 0.2])
```

Paths exclude the fixed initial state. Gaussian scales are standard deviations; OU diffusion is the SDE noise amplitude. Parameters and priors are explicit. [Reference](docs/reference.md) lists signatures, domains, and formulas.

The [posterior example](examples/playground.flatppl) estimates drift and step noise from eight observed states. Its posterior has two unknowns and no unobserved trajectory coordinates.

Run the independent mathematical checks with existing NumPy/SciPy and a JavaScript engine checkout:

```sh
FLATPPL_JS_DIR=/path/to/flatppl-js python tests/check.py
```

Compile the public vector fixture with the Rust CLI:

```sh
flatppl stablehlo tests/runtime.flatppl --dtype f64 -o /tmp/timeseries-runtime.mlir
```

`tests/oracles.py` provides independent values and Jacobians for that fixture, plus one-dimensional quadrature moments for the example posterior. These commands install no dependencies.
