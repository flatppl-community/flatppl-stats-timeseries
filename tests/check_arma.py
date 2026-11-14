"""Check conditional ARMA/ARIMA densities and draws against dense Gaussian laws."""

import json
import subprocess
from pathlib import Path

import numpy as np
from oracles import arma_moments, gaussian_logpdf

ROOT = Path(__file__).resolve().parents[1]
EMPTY = "filter(value -> false, [0.0])"


def literal(values):
    return "[" + ",".join(map(str, values)) + "]" if len(values) else EMPTY


def evaluate(request):
    run = subprocess.run(
        ["node", str(ROOT / "tests/evaluate.cjs")],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(run.stdout)


cases = [
    ([], [], [], [], None),
    ([1.2, -0.4], [], [0.3, -0.2], [], None),
    ([], [0.7, -0.5], [], [0.2, -0.4], None),
    ([1.2, -0.4], [0.7], [0.3, -0.2], [0.2], None),
    (
        [1.2, -0.4, 2, -0.3],
        [0.7, -0.5, 0.4, 0.8],
        [0.3, -0.2, 0.1, 0.05],
        [0.2, -0.1, 0.3, 0.2],
        None,
    ),
    ([1.2, -0.4], [0.7], [0.3, -0.2], [0.2], []),
    ([1.2, -0.4], [0.7], [0.3, -0.2], [0.2], [1.1]),
    ([1.2, -0.4], [0.7], [0.3, -0.2], [0.2], [1.1, -0.3]),
]
expressions, expected = [], []
for history, errors, ar, ma, levels in cases:
    n = 2 if len(ar) == 4 else 3
    args = f"{literal(history)},{literal(errors)},"
    if levels is not None:
        args += literal(levels) + ","
    args += f"{n},0.4,{literal(ar)},{literal(ma)},0.7"
    law = f"a.{'arma' if levels is None else 'arima'}({args})"
    data = [0.2, -0.4, 0.7][:n]
    expressions.append(f"logdensityof({law},{literal(data)})")
    expected.append(
        gaussian_logpdf(
            data, *arma_moments(history, errors, n, 0.4, ar, ma, 0.7, levels or [])
        )
    )
expressions += [
    "logdensityof(a.ar([1.2,-0.4],3,0.4,[0.3,-0.2],0.7),[0.2,-0.4,0.7])",
    "logdensityof(a.ma([0.7,-0.5],3,0.4,[0.2,-0.4],0.7),[0.2,-0.4,0.7])",
    "logdensityof(a.ar([1.2],3,0.4,[-0.6],0.7),[0.2,-0.4,0.7])",
]
expected += [
    expected[1],
    expected[2],
    gaussian_logpdf(
        [0.2, -0.4, 0.7], *arma_moments([1.2], [], 3, 0.4, [-0.6], [], 0.7)
    ),
]
actual = evaluate(expressions)
np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)

sample_count = 4096
sampled = evaluate(
    {
        "source": 'a=load_module("../main.flatppl")\n'
        "path ~ a.arima([1.2,-0.4],[0.7],[1.1,-0.3],3,0.4,[0.3,-0.2],[0.2],0.7)",
        "bindings": ["path"],
        "sampleCount": sample_count,
    }
)[0]
draws = np.asarray(list(sampled["samples"].values())).reshape(sample_count, 3)
mean, covariance = arma_moments(
    [1.2, -0.4], [0.7], 3, 0.4, [0.3, -0.2], [0.2], 0.7, [1.1, -0.3]
)
mean_se = np.sqrt(np.diag(covariance) / sample_count)
covariance_se = np.sqrt(
    (covariance**2 + np.outer(np.diag(covariance), np.diag(covariance)))
    / (sample_count - 1)
)
mean_z = (draws.mean(axis=0) - mean) / mean_se
covariance_z = (np.cov(draws, rowvar=False) - covariance) / covariance_se
assert np.max(abs(mean_z)) < 6
assert np.max(abs(covariance_z)) < 6
print(
    json.dumps(
        {
            "density_cases": len(expected),
            "max_density_error": float(np.max(abs(np.asarray(actual) - expected))),
            "seed": 47710,
            "samples": sample_count,
            "mean_z": mean_z.tolist(),
            "covariance_z": covariance_z.tolist(),
        }
    )
)
