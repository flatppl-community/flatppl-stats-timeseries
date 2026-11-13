"""Check public laws against dense Gaussian and enumerated-state oracles."""

import json
import subprocess
from pathlib import Path

import numpy as np
from oracles import (POINTS, ar1_moments, hmm_enumeration, ou_moments,
                     posterior_moments, runtime_values, walk_moments)
from scipy.stats import norm, poisson

ROOT = Path(__file__).resolve().parents[1]


def evaluate(request):
    result = subprocess.run(["node", str(ROOT / "tests/evaluate.cjs")],
                            input=json.dumps(request), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def literal(x):
    if np.ndim(x) == 2:
        if len(x) == 0:
            return f"zeros([0,{np.shape(x)[1]}])"
        return "rowstack([" + ",".join(literal(row) for row in x) + "])"
    if np.ndim(x) == 1:
        return "[" + ",".join(literal(v) for v in x) + "]"
    return "-inf" if np.isneginf(x) else str(float(x))


source = (ROOT / "tests/runtime.flatppl").read_text()
errors = []
for point in POINTS:
    actual = np.array(evaluate({"source": source.replace("x = elementof(cartpow(reals, 6))", "x = " + literal(point)),
                               "bindings": ["walk_step_score", "walk_score", "ar1_step_score", "ar1_score",
                                            "ou_step_score", "ou_score", "gaussian_score", "poisson_score", "hmm_score"]}))
    expected = runtime_values(point)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
    errors.extend(abs(actual - expected))

# The AR(1) stationary law is invariant under one transition, including rho<0.
for rho in [-0.6, 0.0, 0.8]:
    mean, covariance = ar1_moments(1.2, 3, 1.2, rho, 0.7)
    stationary_variance = 0.7**2 / (1 - rho**2)
    inherited = stationary_variance * np.outer(rho**np.arange(1, 4), rho**np.arange(1, 4))
    np.testing.assert_allclose(np.diag(covariance + inherited), stationary_variance)
    expression = f"logdensityof(a.ar1_step(0.3,1.2,{rho},0.7),0.5)"
    np.testing.assert_allclose(evaluate([expression]), norm.logpdf(0.5, 1.2 + rho * (0.3 - 1.2), 0.7))

# The short-time OU density retains variance diffusion^2*dt and mean x+rate*(m-x)*dt.
dt = 1e-12
actual = evaluate([f"logdensityof(a.ou_step(0.3,{dt},1.2,0.8,0.4),0.3)"])[0]
expected = norm.logpdf(0.3, 0.3 + 0.8 * (1.2 - 0.3) * dt, 0.4 * np.sqrt(dt))
np.testing.assert_allclose(actual, expected, atol=2e-11)

# Exact OU semigroup: a two-interval path has the same terminal law as one combined interval.
mean, covariance = ou_moments(0.3, [0.2, 0.7], 1.2, 0.8, 0.4)
combined_mean, combined_covariance = ou_moments(0.3, [0.9], 1.2, 0.8, 0.4)
np.testing.assert_allclose(mean[-1], combined_mean[0])
np.testing.assert_allclose(covariance[-1, -1], combined_covariance[0, 0])

# Observation laws retain normalizing constants, independent errors, and zero exposure.
observation_actual = evaluate([
    "logdensityof(a.gaussian_observations([0.1,0.5],[0.2,-0.3],[0.7,0.4]),[0.2,0.7])",
    "logdensityof(a.poisson_observations([0.1,0.5],[0,2]),[0,3])",
    "logdensityof(a.poisson_observations([0.1,0.5],[0,2]),[1,3])",
])
np.testing.assert_allclose(np.asarray(observation_actual, dtype=float), [
    norm.logpdf([0.2, 0.7], [0.3, 0.2], [0.7, 0.4]).sum(),
    poisson.logpmf([0, 3], [0, 2 * np.exp(0.5)]).sum(), -np.inf])

# Joint draws lock path ordering and dependent state propagation, beyond density scoring.
sample_count = 8192
sampled = evaluate({
    "source": 'flatppl_compat="0.1"\na=load_module("../main.flatppl")\n'
              'x ~ a.random_walk(0,3,0.2,0.7)\n'
              'y ~ a.ar1(0,3,1,-0.6,0.7)\n'
              'z ~ a.ou(0,[0.1,0.5,0.2],1,0.8,0.4)',
    "bindings": ["x", "y", "z"], "sampleCount": sample_count,
})
sampling_errors = []
for measure, (mean, covariance) in zip(sampled, [
        walk_moments(0, 3, 0.2, 0.7), ar1_moments(0, 3, 1, -0.6, 0.7),
        ou_moments(0, [0.1, 0.5, 0.2], 1, 0.8, 0.4)]):
    draws = np.array(list(measure["samples"].values())).reshape(sample_count, 3)
    mean_se = np.sqrt(np.diag(covariance) / sample_count)
    covariance_se = np.sqrt((covariance**2 + np.outer(np.diag(covariance), np.diag(covariance)))
                            / (sample_count - 1))
    mean_z = (draws.mean(axis=0) - mean) / mean_se
    covariance_z = (np.cov(draws, rowvar=False) - covariance) / covariance_se
    assert np.max(abs(mean_z)) < 6
    assert np.max(abs(covariance_z)) < 6
    sampling_errors.append({"mean_z": mean_z.tolist(), "covariance_z": covariance_z.tolist()})

hmm_cases = [
    ([0.6, 0.4], [[0.8, 0.2], [0.3, 0.7]], [[-0.2, -1.5]]),
    ([0.6, 0.4], [[0.8, 0.2], [0.3, 0.7]], [[-1000, -1001], [-1002, -1003]]),
    ([1, 0], [[1, 0], [0, 1]], [[-0.2, -np.inf], [-0.4, -np.inf]]),
    ([1, 0], [[1, 0], [0, 1]], [[-np.inf, 0], [0, 0]]),
    ([1], [[1]], [[-1000], [-1001], [-1002]]),
    ([0.6, 0.4], [[0.8, 0.2], [0.3, 0.7]], np.empty((0, 2))),
    ([0.2, 0.3, 0.5], [[0.6, 0.3, 0.1], [0.1, 0.7, 0.2], [0.3, 0.2, 0.5]],
     [[-1.0, -0.2, -1.8], [-0.3, -1.4, -0.7], [-0.8, -0.5, -1.1]]),
]
expressions, expected = [], []
for initial, transition, emissions in hmm_cases:
    initial, transition, emissions = map(np.asarray, (initial, transition, emissions))
    expressions.append("a.hmm_loglikelihood(" + ",".join(literal(v) for v in [initial, transition, emissions]) + ")")
    expected.append(hmm_enumeration(initial, transition, emissions))
actual = np.asarray(evaluate(expressions), dtype=float)
np.testing.assert_allclose(actual, expected, atol=2e-12)

print(json.dumps({"runtime_values": len(errors), "max_value_error": float(max(errors)),
                  "hmm_enumeration_cases": len(hmm_cases), "sampling_seed": 47710,
                  "sample_count": sample_count, "sampling_errors": sampling_errors,
                  "posterior": posterior_moments()}))
