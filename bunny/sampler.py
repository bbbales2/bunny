from dataclasses import dataclass
import math
import numpy
import scipy
from typing import Callable, Tuple


from concurrent.futures import ThreadPoolExecutor
import numpy
from tqdm import tqdm

from bunny import nuts

@dataclass
class StepsizeAdapter:
    target_accept_stat: float
    initial_stepsize: float
    gamma: float = 0.05
    kappa: float = 0.75
    t0: float = 10.0
    _counter: int = 0
    _s_bar: int = 0
    _x_bar: int = 0

    def adapt(self, accept_stat: float) -> float:
        self._counter += 1

        accept_stat = min(1.0, accept_stat)

        # Nesterov Dual-Averaging of log(epsilon)
        eta = 1.0 / (self._counter + self.t0)

        self._s_bar = (1.0 - eta) * self._s_bar + eta * (self.target_accept_stat - accept_stat)

        x = math.log(self.initial_stepsize) - self._s_bar * math.sqrt(self._counter) / self.gamma
        x_eta = self._counter ** (-self.kappa)

        self._x_bar = (1.0 - x_eta) * self._x_bar + x_eta * x

        return math.exp(x)

    def adapted_stepsize(self):
        return math.exp(self._x_bar)


def warmup(
    value_and_grad: Callable[[numpy.ndarray], Tuple[float, numpy.ndarray]],
    rng: numpy.random.Generator,
    initial_draw: numpy.ndarray,
    initial_stepsize: float = 1.0,
    initial_diagonal_inverse_metric: numpy.ndarray = None,
    max_treedepth: int = 10,
    target_accept_stat: float = 0.8,
    stage_1_size: int = 100,
    stage_2_size: int = 850,
    stage_2_window_count: int = 4,
    stage_3_size: int = 50,
    debug: bool = False,
):
    size = initial_draw.shape[0]
    stepsize = initial_stepsize

    if initial_diagonal_inverse_metric is None:
        diagonal_inverse_metric = numpy.ones(size)
    else:
        diagonal_inverse_metric = initial_diagonal_inverse_metric
        assert diagonal_inverse_metric.shape[0] == size and diagonal_inverse_metric.shape[1] == size

    with tqdm(total=stage_1_size + stage_2_size + stage_3_size, desc="Moving from initial condition") as progress_bar:
        # Find an initial stepsize that is large enough we are under our target accept rate
        while True:
            next_draw, accept_stat, steps = nuts.one_draw(value_and_grad, rng, initial_draw, stepsize, diagonal_inverse_metric)
            if accept_stat < target_accept_stat:
                break
            stepsize = stepsize * 2.0

        # Back off until we hit our target accept rate
        while True:
            stepsize = stepsize / 2.0
            next_draw, accept_stat, steps = nuts.one_draw(value_and_grad, rng, initial_draw, stepsize, diagonal_inverse_metric)
            if accept_stat > target_accept_stat:
                break

        # Stage 1, leave initial conditions in the dust, adapting stepsize
        current_draw = initial_draw
        stepsize_adapter = StepsizeAdapter(target_accept_stat, stepsize)
        for i in range(stage_1_size):
            current_draw, accept_stat, steps = nuts.one_draw(value_and_grad, rng, current_draw, stepsize, diagonal_inverse_metric)
            stepsize = stepsize_adapter.adapt(accept_stat)
            progress_bar.update()
            progress_bar.set_description(f"Moving from initial condition [leapfrogs {steps}]")

        # Stage 2, estimate diagonal of covariance in windows of increasing size
        stage_2_windows = []

        # Compute window size such that we have a sequence of stage_2_window_count
        # windows that sequentially double in size and in total take less than
        # or equal to stage_2_size draws. In math terms that is something like:
        # window_size + 2 * window_size + 4 * window_size + ... <= stage_2_size
        window_size = math.floor(stage_2_size / (2 ** stage_2_window_count - 1))

        for i in range(stage_2_window_count - 1):
            stage_2_windows.append(window_size * 2 ** i)

        # The last window is whatever is left
        stage_2_windows.append(stage_2_size - sum(stage_2_windows))

        for current_window_size in stage_2_windows:
            stepsize_adapter = StepsizeAdapter(target_accept_stat, stepsize)
            window_draws = numpy.zeros((current_window_size, size))
            for i in range(current_window_size):
                current_draw, accept_stat, steps = nuts.one_draw(value_and_grad, rng, current_draw, stepsize, diagonal_inverse_metric)
                window_draws[i] = current_draw
                stepsize = stepsize_adapter.adapt(accept_stat)
                progress_bar.update()
                progress_bar.set_description(f"Building a metric [leapfrogs {steps}]")
            new_diagonal_inverse_metric = numpy.var(window_draws, axis=0)
            max_scale_change = max(diagonal_inverse_metric / new_diagonal_inverse_metric)
            stepsize = stepsize * max_scale_change
            diagonal_inverse_metric = new_diagonal_inverse_metric

        # Stage 3, fine tune timestep
        stepsize_adapter = StepsizeAdapter(target_accept_stat, stepsize)
        for i in range(stage_3_size):
            current_draw, accept_stat, steps = nuts.one_draw(value_and_grad, rng, current_draw, stepsize, diagonal_inverse_metric)
            stepsize = stepsize_adapter.adapt(accept_stat)
            progress_bar.update()
            progress_bar.set_description(f"Finalizing timestep [leapfrogs {steps}]")

    return current_draw, stepsize_adapter.adapted_stepsize(), diagonal_inverse_metric


def sample(
    value_and_grad: Callable[[numpy.ndarray], Tuple[float, numpy.ndarray]],
    rng: numpy.random.Generator,
    initial_draw: numpy.ndarray,
    stepsize: float,
    diagonal_inverse_metric: numpy.ndarray,
    num_draws: int,
    max_treedepth: int = 10,
):
    size = initial_draw.shape[0]
    draws = numpy.zeros((num_draws, size))

    with tqdm(total=num_draws, desc="Sampling") as progress_bar:
        current_draw = initial_draw
        for i in range(num_draws):
            current_draw, accept_stat, steps = nuts.one_draw(value_and_grad, rng, current_draw, stepsize, diagonal_inverse_metric)
            draws[i, :] = current_draw
            progress_bar.update()
            progress_bar.set_description(f"Sampling [leapfrogs {steps}]")

    return draws


def warmup_and_sample(
    value_and_grad: Callable[[numpy.ndarray], Tuple[float, numpy.ndarray]],
    size: int,
    num_draws: int = 200,
    num_warmup: int = 1000,
    chains: int = 4,
    init: int = 2,
    thin: int = 1,
    target_acceptance_rate: float = 0.85
):
    """
    Sample the target log density using NUTS given the value and gradient of the negative
    log density.
    Sample using `chains` different chains with parameters initialized in unconstrained
    space [-2, 2]. Use `num_warmup` draws to warmup and collect `num_draws` draws in each
    chain after warmup.
    If `thin` is greater than 1, then compute internally `num_draws * thin` draws and
    output only every `thin` draws (so the output is size `num_draws`).
    `target_acceptance_rate` is the target acceptance rate for adaptation. Should be less
    than one and greater than zero.
    """
    # Currently only doing warmup on one chain
    rng = numpy.random.default_rng()

    initial_position = 2 * init * rng.uniform(size=(size)) - init

    assert target_acceptance_rate < 1.0 and target_acceptance_rate > 0.0
    assert num_warmup > 200

    # Ordered as (draws, chains, param)
    unconstrained_draws = numpy.zeros((num_draws, chains, size))
    leapfrog_steps = numpy.zeros((num_draws, chains), dtype=int)
    divergences = numpy.zeros((num_draws, chains), dtype=bool)

    def generate_draws():
        stage_1_size = 100
        stage_3_size = 50
        stage_2_size = num_warmup - stage_1_size - stage_3_size

        initial_draw, stepsize, diagonal_inverse_metric = warmup(
            value_and_grad,
            rng,
            initial_position,
            target_accept_stat=target_acceptance_rate,
            stage_1_size=stage_1_size,
            stage_2_size=stage_2_size,
            stage_3_size=stage_3_size,
        )

        return sample(value_and_grad, rng, initial_draw, stepsize, diagonal_inverse_metric, num_draws, thin)

    with ThreadPoolExecutor(max_workers=chains) as e:
        results = []
        for chain in range(chains):
            results.append(e.submit(generate_draws))

        for chain, result in enumerate(results):
            unconstrained_draws[:, chain, :] = result.result()

    return unconstrained_draws