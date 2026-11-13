"""Independent dense-Gaussian path and exhaustive finite-state oracles."""

from itertools import product

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import gamma, norm

POINTS = np.array([[0.2, 0.7, 0.4, 0.8, 0.4, 1.0], [-0.1, 1.2, -0.6, 0.05, 0.9, -0.3]])
DATA = np.array([0.4, 0.3, 1.0, 1.2, 1.1, 1.7, 1.9, 2.4])
INCREMENTS = np.diff(np.r_[0.0, DATA])


def gaussian_logpdf(y, mean, covariance):
    """Dense normalized Gaussian density; permits complex-step derivatives."""
    residual = np.atleast_1d(y) - np.atleast_1d(mean)
    covariance = np.atleast_2d(covariance)
    return -(len(residual) * np.log(2 * np.pi) + np.log(np.linalg.det(covariance))
             + residual @ np.linalg.solve(covariance, residual)) / 2


def walk_moments(initial, n, drift, sigma):
    times = np.arange(1, n + 1)
    return initial + drift * times, sigma**2 * np.minimum.outer(times, times)


def ar1_moments(initial, n, mean, rho, sigma):
    times = np.arange(1, n + 1)
    # Propagate independent innovations into the whole vector in one linear map.
    powers = np.arange(n)[:, None] - np.arange(n)[None, :]
    lower = np.tril(rho ** np.maximum(powers, 0))
    return mean + (initial - mean) * rho**times, sigma**2 * (lower @ lower.T)


def ou_moments(initial, dts, mean, rate, diffusion):
    times = np.cumsum(dts)
    # Solve the SDE covariance integral directly from absolute observation times.
    covariance = diffusion**2 / (2 * rate) * (
        np.exp(-rate * np.abs(times[:, None] - times[None, :]))
        - np.exp(-rate * (times[:, None] + times[None, :])))
    return mean + (initial - mean) * np.exp(-rate * times), covariance


def hmm_enumeration(initial, transition, log_emissions):
    if len(log_emissions) == 0:
        return 0.0
    terms = []
    with np.errstate(divide="ignore"):
        for states in product(range(len(initial)), repeat=len(log_emissions)):
            score = np.log(initial[states[0]]) + log_emissions[0, states[0]]
            for i in range(1, len(states)):
                score += np.log(transition[states[i - 1], states[i]]) + log_emissions[i, states[i]]
            terms.append(score)
    return logsumexp(terms)


def runtime_values(x):
    drift, sigma, rho, rate, diffusion, offset = x
    data = np.array([0.1, 0.5, 0.2])
    means, covariances = zip(
        walk_moments(0.3, 1, drift, sigma),
        walk_moments(0, 3, drift, sigma),
        ar1_moments(0.3, 1, offset, rho, sigma),
        ar1_moments(0, 3, offset, rho, sigma),
        ou_moments(0.3, [0.2], offset, rate, diffusion),
        ou_moments(0, [0.1, 0.5, 0.2], offset, rate, diffusion),
        (data + offset, sigma**2 * np.eye(3)),
    )
    values = [gaussian_logpdf(y, m, c) for y, m, c in zip(
        [[0.5], data, [0.5], data, [0.5], data, [0.2, 0.3, 0.7]], means, covariances)]
    rates = np.array([np.exp(drift), 2 * np.exp(offset)])
    values.append(-rates.sum() + 3 * np.log(rates[1]) - np.log(6))
    # Exactly four paths, evaluated directly rather than a forward recursion.
    emissions = np.array([[-sigma, -rate], [-diffusion, -sigma]])
    weights = np.array([[0.6 * 0.8, 0.6 * 0.2], [0.4 * 0.3, 0.4 * 0.7]])
    values.append(np.log(np.sum(weights * np.exp(emissions[0, :, None] + emissions[1, None, :]))))
    return np.asarray(values)


def runtime_oracle(x):
    x = np.asarray(x)
    jacobian = np.column_stack([
        np.imag(runtime_values(x.astype(complex) + 1e-25j * np.eye(len(x))[i])) / 1e-25
        for i in range(len(x))])
    return runtime_values(x), jacobian


def posterior_logdensity(p):
    drift, sigma = p
    return float(norm.logpdf(drift) + gamma.logpdf(sigma, 3, scale=0.25)
                 + np.sum(norm.logpdf(INCREMENTS, drift, sigma)))


def posterior_moments():
    """Integrate sigma after analytically integrating the Gaussian drift."""
    n, total = len(INCREMENTS), INCREMENTS.sum()
    centered = np.sum((INCREMENTS - INCREMENTS.mean())**2)

    def integrand(sigma, moment):
        variance = sigma**2 / (n + sigma**2)
        mean = total / (n + sigma**2)
        # Orthogonal residuals and the sample mean have independent Gaussian laws.
        lp = (gamma.logpdf(sigma, 3, scale=0.25) - (n - 1) * np.log(sigma)
              - centered / (2 * sigma**2)
              - 0.5 * np.log(1 + sigma**2 / n)
              - INCREMENTS.mean()**2 / (2 * (1 + sigma**2 / n)))
        return np.exp(lp) * [1, mean, mean**2 + variance, sigma, sigma**2, sigma * mean][moment]

    moments = np.array([quad(integrand, 0, np.inf, args=(i,), epsabs=1e-10, epsrel=1e-10)[0]
                        for i in range(6)])
    moments /= moments[0]
    mean = moments[[1, 3]]
    variance = moments[[2, 4]] - mean**2
    return {"mean": mean.tolist(), "variance": variance.tolist(),
            "covariance": float(moments[5] - mean.prod())}
