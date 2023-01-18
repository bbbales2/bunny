import jax
import numpy
import pytest
from bunny.sampler import sample, warmup_and_sample

def log_density(mu: float, sd: float, x: jax.numpy.ndarray):
    return jax.numpy.sum(jax.scipy.stats.norm.logpdf(x, mu, sd))

def test_warmup_and_sampler():
    mu = 1.7
    sd = 2.5
    negative_log_density_with_parameters = lambda x : -log_density(mu, sd, x)
    value_and_grad = jax.jit(jax.value_and_grad(negative_log_density_with_parameters))
    size = 10
    rng = numpy.random.default_rng()
    initial_draw = 4 * rng.uniform(size=(size)) - 2

    draws_with_chain = warmup_and_sample(value_and_grad=value_and_grad, rng=rng, initial_draw=initial_draw, num_draws=1000)
    draws = draws_with_chain.reshape(-1, draws_with_chain.shape[-1])

    mu_hat = numpy.mean(draws, axis = 0)
    sd_hat = numpy.std(draws, axis = 0)

    assert pytest.approx(mu_hat, rel = 0.1) == mu * numpy.ones(size)
    assert pytest.approx(sd_hat, rel = 0.1) == sd * numpy.ones(size)

def test_sample():
    mu = 0.0
    sd = 1.0
    negative_log_density_with_parameters = lambda x : -log_density(mu, sd, x)
    value_and_grad = jax.jit(jax.value_and_grad(negative_log_density_with_parameters))
    size = 10
    initial_draw = numpy.zeros(size)
    diagonal_inverse_metric = numpy.ones(size)

    rng = numpy.random.default_rng()

    draws= sample(
        value_and_grad=value_and_grad,
        rng=rng,
        initial_draw=initial_draw,
        stepsize=0.5,
        diagonal_inverse_metric=diagonal_inverse_metric,
        num_draws=1000
    )

    mu_hat = numpy.mean(draws, axis = 0)
    sd_hat = numpy.std(draws, axis = 0)

    assert pytest.approx(mu_hat, rel = 0.1, abs = 0.1) == mu * numpy.ones(size)
    assert pytest.approx(sd_hat, rel = 0.1) == sd * numpy.ones(size)