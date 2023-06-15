import numpy
from typing import List, Tuple
from numpy.typing import ArrayLike

def generate_beta_and_gamma(S: List[ArrayLike], Z: List[ArrayLike]) -> Tuple[ArrayLike, ArrayLike]:
    pass

def lbfgs(
    target_value_and_grad
    , theta_init : ArrayLike
    , J
    , tau_rel
    , L
    , c1
    , c2
    , epsilon
) -> Tuple[List[ArrayLike], List[ArrayLike]]:
    theta_current = theta_init
    S = []
    Z = []
    alpha = numpy.ones(theta_init.shape)
    value_current, grad_current = target_value_and_grad(theta_current)
    thetas = [theta_current]
    grads = [grad_current]
    for l in range(L):
        beta, gamma = generate_beta_and_gamma(S, Z)
        delta = alpha * grad_current + (beta @ (gamma @ (beta.T @ grad_current)))
        for i in range(20):
            lambda_ = 0.5 ** i
            theta_next = theta_current + lambda_ * delta

            value_next, grad_next = target_value_and_grad(theta_next)

            grad_current_dot_delta = grad_current.T @ delta
            wolf_value_check = value_next >= value_current + c1 * l * grad_current_dot_delta
            wolf_grad_check = (grad_next.T @ delta) <= c2 * grad_current_dot_delta

            if wolf_value_check and wolf_grad_check:
                break
        else:
            raise Exception("Wolf conditions never met")

        thetas.append(theta_next)
        grads.append(grad_next)

        if abs((value_next - value_current) / value_current) < tau_rel:
            return thetas, grads
        
        S_next = theta_next - theta_current
        Z_next = grad_next - grad_current

        Z_next_squared = Z_next.T @ Z_next
        S_next_T_Z_next = S_next.T @ Z_next

        if S_next_T_Z_next > epsilon * Z_next_squared:
            S = S[-J + 1:]
            Z = Z[-J + 1:]

            S.append(S_next)
            Z.append(Z_next)

            alpha = numpy.ones(alpha.shape) * Z_next_squared / S_next_T_Z_next
    else:
        raise Exception("Reached maximum number of iterations without converging")
