import numpy as np
import scipy.signal


def discount(vector, gamma):
    """
    computes discounted sums along 0th dimension of vector x.
        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1

    :param vector: (np.ndarray) the input vector
    :param gamma: (float) the discount value
    :return: (np.ndarray) the output vector
    """
    assert vector.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], vector[::-1], axis=0)[::-1]


def explained_variance(y_pred, y_true):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: (np.ndarray) the prediction
    :param y_true: (np.ndarray) the expected value
    :return: (float) explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


def explained_variance_2d(y_pred, y_true):
    """
    Computes fraction of variance that ypred explains about y, for 2D arrays.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: (np.ndarray) the prediction
    :param y_true: (np.ndarray) the expected value
    :return: (float) explained variance of ypred and y
    """
    assert y_true.ndim == 2 and y_pred.ndim == 2
    var_y = np.var(y_true, axis=0)
    explained_var = 1 - np.var(y_true - y_pred) / var_y
    explained_var[var_y < 1e-10] = 0
    return explained_var


def flatten_arrays(arrs):
    """
    flattens a list of arrays down to 1D

    :param arrs: ([np.ndarray]) arrays
    :return: (np.ndarray) 1D flattend array
    """
    return np.concatenate([arr.flat for arr in arrs])


def unflatten_vector(vec, shapes):
    """
    reshape a flattened array

    :param vec: (np.ndarray) 1D arrays
    :param shapes: (tuple)
    :return: ([np.ndarray]) reshaped array
    """
    i = 0
    arrs = []
    for shape in shapes:
        size = np.prod(shape)
        arr = vec[i:i + size].reshape(shape)
        arrs.append(arr)
        i += size
    return arrs


def discount_with_boundaries(rewards, episode_starts, gamma):
    """
    computes discounted sums along 0th dimension of x (reward), while taking into account the start of each episode.
        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1

    :param rewards: (np.ndarray) the input vector (rewards)
    :param episode_starts: (np.ndarray) 2d array of bools, indicating when a new episode has started
    :param gamma: (float) the discount factor
    :return: (np.ndarray) the output vector (discounted rewards)
    """
    discounted_rewards = np.zeros_like(rewards)
    n_samples = rewards.shape[0]
    discounted_rewards[n_samples - 1] = rewards[n_samples - 1]
    for step in range(n_samples - 2, -1, -1):
        discounted_rewards[step] = rewards[step] + gamma * discounted_rewards[step + 1] * (1 - episode_starts[step + 1])
    return discounted_rewards

def conjugate_gradient(f_ax, b_vec, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    conjugate gradient calculation (Ax = b), bases on
    https://epubs.siam.org/doi/book/10.1137/1.9781611971446 Demmel p 312

    :param f_ax: (function) The function describing the Matrix A dot the vector x
                 (x being the input parameter of the function)
    :param b_vec: (numpy float) vector b, where Ax = b
    :param cg_iters: (int) the maximum number of iterations for converging
    :param callback: (function) callback the values of x while converging
    :param verbose: (bool) print extra information
    :param residual_tol: (float) the break point if the residual is below this value
    :return: (numpy float) vector x, where Ax = b
    """
    first_basis_vect = b_vec.copy()  # the first basis vector
    residual = b_vec.copy()  # the residual
    x_var = np.zeros_like(b_vec)  # vector x, where Ax = b
    residual_dot_residual = residual.dot(residual)  # L2 norm of the residual

    fmt_str = "%10i %10.3g %10.3g"
    title_str = "%10s %10s %10s"
    if verbose:
        print(title_str % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x_var)
        if verbose:
            print(fmt_str % (i, residual_dot_residual, np.linalg.norm(x_var)))
        z_var = f_ax(first_basis_vect)
        v_var = residual_dot_residual / first_basis_vect.dot(z_var)
        x_var += v_var * first_basis_vect
        residual -= v_var * z_var
        new_residual_dot_residual = residual.dot(residual)
        mu_val = new_residual_dot_residual / residual_dot_residual
        first_basis_vect = residual + mu_val * first_basis_vect

        residual_dot_residual = new_residual_dot_residual
        if residual_dot_residual < residual_tol:
            break

    if callback is not None:
        callback(x_var)
    if verbose:
        print(fmt_str % (i + 1, residual_dot_residual, np.linalg.norm(x_var)))
    return x_var
