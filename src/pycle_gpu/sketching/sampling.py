import numpy as np


def sample_from_pdf(pdf, x, n_samples=1):
    """
    x is a vector (the support of the pdf), pdf is the values of pdf eval at x
    :param pdf: array of size n
    :param x: array of size n
    :param n_samples: int. Number of sampling.
    :return: array of size n
    """
    pdf = pdf / np.sum(pdf)  # ensure pdf is normalized
    cdf = np.cumsum(pdf)
    sample_cdf = np.random.uniform(0, 1, n_samples)
    sample_x = np.interp(sample_cdf, cdf, x)
    return sample_x


def pdf_adapted_radius(r, k_means=False):
    """up to a constant"""
    if k_means:
        return r*np.exp(-(r**2)/2)  # Dont take the gradient according to sigma into account
    else:
        return np.sqrt(r**2 + (r**4)/4)*np.exp(-(r**2)/2)


def draw_frequencies_adapted_radius(d, m, sigma, k_means=False):
    """
    Draw the 'frequencies' or projection matrix Omega for sketching. Adapted Radius heuristics.
    The covariance matrix sigma is diagonal.
    Draws frequencies according to some sampling pattern
    omega = R*Sigma^{-1/2}*phi, for R from adapted with variance 1, phi uniform on the unit ball
    Set k_means to True if considering adapted_radius pdf for K_means.

    Arguments:
        - d: int, dimension of the data to sketch
        - m: int, number of 'frequencies' to draw (the target sketch dimension)
        - sigma: (d,d)-numpy array, the covariance of the data (note that we typically use Sigma^{-1} in the frequency domain).
        - k_means: bool, for 'pdf_adapted_radius' method

    Returns:
        - omega: (d,m)-numpy array containing the 'frequency' projection matrix
    """
    # Sample the radii
    r = np.linspace(0, 5, 2001)
    sampled_radii = sample_from_pdf(pdf_adapted_radius(r, k_means), r, n_samples=m)

    phi = np.random.randn(d, m)
    phi = phi / np.linalg.norm(phi, axis=0)  # normalize -> randomly sampled from unit sphere
    sig_fact = np.linalg.inv(np.linalg.cholesky(sigma))
    return sig_fact @ phi * sampled_radii   # @ is computed before *
