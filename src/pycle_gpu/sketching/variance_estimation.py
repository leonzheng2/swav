import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.pycle_gpu.sketching.feature_map import ComplexExponentialFeatureMap, compute_sketch_rff
from src.pycle_gpu.sketching.frequency_matrix import DenseFrequencyMatrix


def estimate_log_sigma_from_sketch(sketch, phi, log_sigma2_bar, c=20, tol=1e-4, max_iter=1000, should_plot=False):
    """
    Estimation of mean variance for frequency sampling, see "Sketching for Large-Scale Learning of Mixture Models",
    Keriven et al. (see section 3.3)
    Always max for the mode. log_sigma2_bar is the parameter to optimize. Require grad is true for this tensor.
    tol for torch optimization. Includes tensorboard visualization of loss during optimization.
    :param sketch: torch tensor
    :param phi: ComplexExponentialFeatureMap
    :param log_sigma2_bar: tensor which requires grad
    :param c: number of blocks
    :param tol: tolerance for optimization. Stopping criteria based on relative difference.
    :param max_iter: Max iteration for optimization.
    :param should_plot: set True to plot graphs
    :return: log_sigma2_bar optimized
    """
    assert isinstance(phi, ComplexExponentialFeatureMap)

    # Sort the frequencies by norm
    omega = phi.freq_matrix.get_dense_frequency_matrix()
    radii = torch.linalg.norm(omega, axis=0)
    i_sort = torch.argsort(radii)
    radii = radii[i_sort]
    sketch_sorted = sketch[i_sort] / phi.c_norm  # sort and normalize individual entries to 1

    # Finding indices of maximum absolute values
    s = phi.m // c     # number of frequencies per box
    indices_of_max = torch.empty(c).to(phi.device)     # find the indices of the max of each block
    for ic in range(c):
        start_of_bloc = ic * s
        j_max = torch.argmax(torch.abs(sketch_sorted)[start_of_bloc: start_of_bloc + s]) + start_of_bloc
        indices_of_max[ic] = j_max
    indices_of_max = indices_of_max.to(torch.long)

    # Torch optimization
    radii_to_fit = radii[indices_of_max]
    sketch_to_fit = torch.abs(sketch_sorted)[indices_of_max]
    optimizer = torch.optim.Adam([log_sigma2_bar])
    if should_plot:
        writer = SummaryWriter()
    for iteration in range(max_iter):
        optimizer.zero_grad()
        tensor = sketch_to_fit - torch.exp(torch.mul(torch.exp(log_sigma2_bar), -0.5) * torch.square(radii_to_fit))
        loss = torch.norm(tensor)
        loss.backward()
        optimizer.step()

        # Stopping criteria
        if iteration == 0:
            previous_loss = loss
        else:
            relative_diff = torch.abs(previous_loss - loss) / previous_loss
            if should_plot:
                writer.add_scalar('EstimateSigma/loss', loss, iteration)
            if relative_diff.item() < tol:
                break
            previous_loss = loss

    # Plot if required
    if should_plot:
        plt.figure(figsize=(10, 5))
        rfit = np.linspace(0, torch.max(radii).item(), 100)
        zfit = np.exp(- 0.5 * torch.exp(log_sigma2_bar).item() * rfit ** 2)
        plt.plot(radii, np.abs(sketch_sorted), '.')
        plt.plot(radii_to_fit, sketch_to_fit, '.')
        plt.plot(rfit, zfit)
        plt.xlabel('R')
        plt.ylabel('|z|')
        plt.show()

    return log_sigma2_bar


def estimate_sigma_adapted_radius(dataset, m0, n0, c=20, n_iter=5, k_means=False, should_plot=False):
    """Automatically estimates the "Sigma" parameter(s) (the scale of data clusters) for generating the sketch operator.
    See "Sketching for Large-Scale Learning of Mixture Models", Keriven et al. (see section 3.3)

    We assume here that Sigma = sigma2_bar * identity matrix.
    To estimate sigma2_bar, lightweight sketches of size m0 are generated from (a small subset of) the dataset
    with candidate values for sigma2_bar. Then, sigma2_bar is updated by fitting a Gaussian
    to the absolute values of the obtained sketch.

    Arguments:
        - dataset: (n,d) tensor, the dataset X: n examples in dimension d
        - m0: int, number of candidate 'frequencies' to draw (can be typically smaller than m).
        - c:  int (default 20), number of 'boxes' (i.e. number of maxima of sketch absolute values to fit)
        - n0: int or None, if given, n0 samples from the dataset are subsampled to be used for Sigma estimation
        - n_iter: int (default 5), the maximum number of iteration (typically stable after 2 iterations)

    Returns:
        - sigma2_bar: float
    """

    (n, d) = dataset.shape
    sub_dataset = dataset[np.random.choice(n, n0, replace=False)]
    dataloader = DataLoader(sub_dataset, batch_size=200)

    # Check if we dont overfit the empirical Fourier measurements
    if m0 < 2 * c:
        print("WARNING: overfitting regime detected for frequency sampling fitting")

    # Parameter to optimize. Log of sigma**2.
    log_sigma2_bar = nn.Parameter(torch.zeros(()), requires_grad=True)

    # Actual algorithm
    for i in range(n_iter):
        # Dense frequency matrix
        dense_omega_0 = DenseFrequencyMatrix(d, m0, torch.device('cpu'), dataset.dtype)
        # Draw frequencies according to current estimate
        sigma2_bar_matrix = torch.exp(log_sigma2_bar).item() * np.eye(d)
        dense_omega_0.sampling(sigma2_bar_matrix, k_means)
        phi_0 = ComplexExponentialFeatureMap(dense_omega_0)
        # Compute unnormalized complex exponential sketch
        sketch_0_np = compute_sketch_rff(dataloader, phi_0)
        # Estimation step
        log_sigma2_bar = estimate_log_sigma_from_sketch(sketch_0_np, phi_0, log_sigma2_bar, should_plot=should_plot)

    sigma2_bar = torch.exp(log_sigma2_bar).item()
    return sigma2_bar
