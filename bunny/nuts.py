import math
import numpy
import scipy
from typing import Callable, Tuple

def uturn(q_plus: numpy.ndarray, q_minus: numpy.ndarray, p_plus: numpy.ndarray, p_minus: numpy.ndarray) -> bool:
    # no_uturn_forward = numpy.dot(p_plus, q_plus - q_minus) > 0
    # no_uturn_backward = numpy.dot(-p_minus, q_minus - q_plus) > 0

    # return not(no_uturn_forward or no_uturn_backward)
    uturn_forward = numpy.dot(p_plus, q_plus - q_minus) <= 0
    uturn_backward = numpy.dot(-p_minus, q_minus - q_plus) <= 0

    return uturn_forward and uturn_backward


def one_draw(
    value_and_grad: Callable[[numpy.ndarray], Tuple[float, numpy.ndarray]],
    rng: numpy.random.Generator,
    current_draw: numpy.ndarray,
    stepsize: float,
    diagonal_inverse_metric: numpy.ndarray,
    max_treedepth: int = 10,
    debug: bool = False,
):
    """
    Generate a draw using Multinomial NUTS (https://arxiv.org/abs/1701.02434 with the original
    Uturn criteria https://arxiv.org/abs/1111.4246)
    * potential - Potential object representing the unnormalized negative log density of distribution to sample
    * rng - Instance of numpy.random.Generator for generating rngs
    * current_draw - current draw
    * stepsize - leapfrog stepsize
    * diagonal_inverse_metric - Diagonal of the inverse metric (usually diagonal of covariance of draws)
    * max_treedepth - max treedepth
    * debug - if true, then return a bunch of debugging info, otherwise return only the next draw
    If debug is false, return dictionary containing containing next draw and acceptance probability statistic.
    If debug is true, a lot of stuff
    """
    q0 = current_draw

    # TODO: Clean up code with changed variable names
    h = stepsize
    diag_M_inv = diagonal_inverse_metric

    # L_inv = numpy.linalg.cholesky(M_inv)
    diag_L_inv = numpy.sqrt(diag_M_inv)
    size = diag_L_inv.shape[0]

    z = rng.normal(0.0, 1.0, size)

    # p0 = numpy.linalg.solve(L_inv.transpose(), z)
    p0 = z / diag_L_inv

    def kinetic_energy(p):
        # return 0.5 * numpy.dot(p, numpy.dot(M_inv, p))
        return 0.5 * numpy.dot(p, p * diag_M_inv)

    U0, grad0 = value_and_grad(q0)
    H0 = kinetic_energy(p0) + U0

    # directions is a length max_treedepth vector that maps each treedepth
    #  to an integration direction (left or right)
    directions = rng.integers(low=0, high=2, size=max_treedepth + 1) * 2 - 1

    choice_draws = rng.random(size=max_treedepth)
    pick_draws = rng.random(size=max_treedepth)

    # depth_map will be a vector of length 2^max_treedepth that maps each of
    #  the possibly 2^max_treedepth points to a treedepth
    depth_map = numpy.zeros(1)
    for depth in range(1, max_treedepth + 1):
        direction = directions[depth]

        new_section = numpy.repeat(depth, 2 ** max(0, depth - 1))
        if direction < 0:
            # depth_map = ([depth] * 2 ** max(0, depth - 1)) + depth_map
            depth_map = numpy.hstack((new_section, depth_map))
        else:
            # depth_map = depth_map + ([depth] * 2 ** max(0, depth - 1))
            depth_map = numpy.hstack((depth_map, new_section))

    depth_map = depth_map.astype(int)

    # Steps is a dict that maps treedepth to which leapfrog steps were
    #  computed in that treedepth (kinda the opposite of depth_map)
    steps = {}
    # qs stores our positions
    qs = numpy.zeros((2 ** max_treedepth, size))
    # ps stores our momentums
    ps = numpy.zeros((2 ** max_treedepth, size))
    # log_pi defined in section A.2.3 of https://arxiv.org/abs/1701.02434
    log_pi = numpy.zeros(2 ** max_treedepth)
    # index of initial state
    i_first = numpy.nonzero(depth_map == 0)[0][0]

    qs[i_first, :] = q0
    ps[i_first, :] = p0
    log_pi[i_first] = -H0
    accept_stat = None
    # i_left and i_right are indices that track the leftmost and rightmost
    #  states of the integrated trajectory
    i_left = i_first
    i_right = i_first
    # log_sum_pi_old is the log of the sum of the pis (of log_pi) for the
    #  tree processed so far
    log_sum_pi_old = log_pi[i_first]
    # i_old will be the sample chosen (the sample from T(z' | told) in section
    #  A.3.1 of https://arxiv.org/abs/1701.02434)
    i_old = i_first
    # We need to know whether we terminated the trajectory cause of a uturn or we
    #  hit the max trajectory length
    uturn_detected = False
    # For trees of increasing treedepth
    for depth in range(1, max_treedepth + 1):
        # Figure out what leapfrog steps we need to compute. If integrating in the
        #  positive direction update the index that points at the right side of the
        #  trajectory. If integrating in the negative direction update the index pointing
        #  to the left side of the trajectory.
        if directions[depth] < 0:
            depth_steps = numpy.flip(numpy.nonzero(depth_map == depth)[0])
            i_left = depth_steps[-1]
        else:
            depth_steps = numpy.nonzero(depth_map == depth)[0]
            i_right = depth_steps[-1]

        steps[depth] = depth_steps

        checks = []
        # What we're doing here is generating a trajectory composed of a number of leapfrog states.
        # We apply a set of comparisons on this trajectory that can be organized to look like a binary tree.
        # Sometimes I say trajectory instead of tree. Each piece of the tree in the comparison corresponds
        #  to a subset of the trajectory. When I say trajectory I'm referring to a piece of the trajectory that
        #  also corresponds to some sorta subtree in the comparisons.
        #
        # This is probably confusing but what I want to communicate is trajectory and tree are very related
        #  but maybe technically not the same thing.
        #
        # Detect U-turns in newly integrated subtree
        uturn_detected_new_tree = False
        tree_depth = round(numpy.log2(len(depth_steps)))

        # Starts and ends are relative because they point to the ith leapfrog step in a sub-trajectory
        #  of size 2^tree_depth which needs to be mapped to the global index of qs
        #  (which is defined in the range 1:2^max_treedepth)
        #
        # The sort is necessary because depth_steps is sorted in order of leapfrog steps taken
        #  which might be backwards in time (so decreasing instead of increasing)
        #
        # This sort is important because we need to keep track of what is left and what is right
        #  in the trajectory so that we do the right comparisons
        sorted_depth_steps = sorted(depth_steps)

        if tree_depth > 0:
            # Start at root of new subtree and work down to leaves
            for uturn_depth in range(tree_depth, 0, -1):
                # The root of the comparison tree compares the leftmost to the rightmost states of the new
                #  part of the trajectory.
                #  The next level down in the tree of comparisons cuts that trajectory in two and compares
                #  the leftmost and rightmost elements of those smaller trajectories.
                #  Etc. Etc.
                div_length = 2 ** (uturn_depth)

                # Starts are relative indices pointing to the leftmost state of each comparison to be done
                starts = numpy.arange(0, len(depth_steps), div_length)  # seq(1, length(depth_steps), div_length)
                # Ends are relative indices pointing to the rightmost state for each comparison to be done
                ends = starts + div_length - 1

                for start, end in zip(starts, ends):
                    checks.append((start, end))

        # Sort into order that checks happen
        checks.sort(key=lambda x: x[1])

        # Initialize u-turn check variable
        uturn_detected_new_tree = False

        # Actually do the integrationg
        if True:
            # with potential.activate_thread() as value_and_grad:
            dt = h * directions[depth]

            i_prev = depth_steps[0] - directions[depth]

            q = qs[i_prev, :].copy()
            p = ps[i_prev, :].copy()

            U, grad = value_and_grad(q)

            # Initialize pointer into checks list
            check_i = 0

            # These are a bunch of temporaries to minimize numpy
            # allocations during integration
            p_half = numpy.zeros(size)
            half_dt = dt / 2
            half_dt_grad = half_dt * grad
            dt_diag_M_inv = dt * diag_M_inv
            for leapfrogs_taken, i in enumerate(depth_steps, start=1):
                # leapfrog step
                # p_half = p - (dt / 2) * grad
                p -= half_dt_grad  # p here is actually p_half
                # q = q + dt * numpy.dot(M_inv, p_half)
                # q = q + dt * diag_M_inv * p_half
                q += dt_diag_M_inv * p
                U, grad = value_and_grad(q)
                half_dt_grad = half_dt * grad
                # p = p_half - (dt / 2) * grad
                p -= half_dt_grad  # p here is indeed p

                K = kinetic_energy(p)
                H = K + U

                qs[i] = q
                ps[i] = p
                log_pi[i] = -H

                while check_i < len(checks) and checks[check_i][1] < leapfrogs_taken:
                    start, end = checks[check_i]

                    start_i = sorted_depth_steps[start]
                    end_i = sorted_depth_steps[end]

                    is_uturn = uturn(
                        qs[end_i, :],
                        qs[start_i, :],
                        ps[end_i, :],
                        ps[start_i, :],
                    )

                    if is_uturn:
                        uturn_detected_new_tree = True
                        break

                    check_i += 1

                if uturn_detected_new_tree:
                    break

        # Merging the two trees requires one more uturn check from the overall left to right states
        uturn_detected = uturn_detected_new_tree | uturn(qs[i_right, :], qs[i_left, :], ps[i_right, :], ps[i_left, :])

        # Accept statistic from ordinary HMC
        # Only compute the accept probability for the steps done
        log_pi_steps = log_pi[depth_steps[0:leapfrogs_taken]]
        energy_loss = H0 + log_pi_steps
        p_step_accept = numpy.minimum(1.0, numpy.exp(energy_loss))

        # Average the acceptance statistic for each step in this branch of the tree
        p_tree_accept = numpy.mean(p_step_accept)

        # Divergence
        if max(numpy.abs(energy_loss)) > 1000 or numpy.isnan(energy_loss).any():
            if accept_stat is None:
                accept_stat = 0.0

            break

        if uturn_detected:
            # If we u-turn on the first step, grab something for the accept_stat
            if accept_stat is None:
                accept_stat = p_tree_accept

            break

        old_accept_stat = accept_stat

        if accept_stat is None:
            accept_stat = p_tree_accept
        else:
            accept_stat = (accept_stat * (len(depth_steps) - 1) + p_tree_accept * leapfrogs_taken) / (
                leapfrogs_taken + len(depth_steps) - 1
            )

        # log of the sum of pi (A.3.1 of https://arxiv.org/abs/1701.02434) of the new subtree
        log_sum_pi_new = scipy.special.logsumexp(log_pi_steps)

        # sample from the new subtree according to the equation in A.2.1 in https://arxiv.org/abs/1701.02434
        #  (near the end of that section)
        if depth > 1:
            i_new = depth_steps[numpy.where(choice_draws[depth - 1] < numpy.cumsum(scipy.special.softmax(log_pi_steps)))[0][0]]
        else:
            i_new = depth_steps[0]

        # Pick between the samples generated from the new and old subtrees using the biased progressive sampling in
        #  A.3.2 of https://arxiv.org/abs/1701.02434
        p_new = min(1, numpy.exp(log_sum_pi_new - log_sum_pi_old))
        if pick_draws[depth - 1] < p_new:
            i_old = i_new

        # Update log of sum of pi of overall tree
        log_sum_pi_old = numpy.logaddexp(log_sum_pi_old, log_sum_pi_new)

    # Get the final sample
    q = qs[i_old, :]

    steps_taken = numpy.nonzero(depth_map <= depth)[0]

    # if debug:
    #     q_columns = [f"q{i}" for i in range(len(q0))]
    #     p_columns = [f"p{i}" for i in range(len(p0))]

    #     qs_df = pandas.DataFrame(qs, columns=q_columns)
    #     ps_df = pandas.DataFrame(ps, columns=p_columns)

    #     # For a system with N parameters, this tibble will have
    #     #  2N + 2 columns. The first column (log_pi) stores log of pi (A.2.3 of https://arxiv.org/abs/1701.02434)
    #     #  The second column (depth_map) stores at what treedepth that state was added to the trajectory
    #     #  The next N columns are the positions (all starting with q)
    #     #  The next N columns are the momentums (all starting with p)
    #     trajectory_df = (
    #         pandas.concat([pandas.DataFrame({"log_pi": log_pi, "depth_map": depth_map}), qs_df, ps_df], axis=1).assign(
    #             valid=lambda df: True if not uturn_detected else df["depth_map"] < depth
    #         )
    #     ).iloc[steps_taken]

    #     return (
    #         q,
    #         accept_stat,
    #         2 ** depth,
    #         {
    #             "i": i_old - min(numpy.nonzero(depth_map <= depth)[0]) + 1,  # q is the ith row of trajectory
    #             "q0": q0,
    #             "h": h,
    #             "max_treedepth": max_treedepth,
    #             "trajectory": trajectory_df,  # tibble containing details of trajectory
    #             "directions": directions,  # direction integrated in each subtree
    #         },
    #     )
        
    return q, accept_stat, 2 ** depth
