import torch
import numpy as np
from src.pycle_gpu.cl_algo.solver import CompMixtureLearning
from src.pycle_gpu.cl_algo.clomp import TorchClomp
from src.pycle_gpu.cl_algo.hierarchical import HierarchicalCompLearning
from src.pycle_gpu.cl_algo.optimization import Projector, ProjectorClip
from src.pycle_gpu.sketching.feature_map import ComplexExponentialFeatureMap, gaussian_characteristic_function_diag_covariance

class CompressiveDiagGmm(CompMixtureLearning):
    """ Compressive Gaussian Mixture Learning, where the covariance of each Gaussian component is diagonal """
    def __init__(self, phi, nb_mixtures, data_lower_bounds, data_upper_bounds, mean_projector, sketch, sigma2_bar,
                 freq_blocks_batch=None, atoms_batch=None, **kwargs):
        assert isinstance(data_lower_bounds, torch.Tensor)
        assert isinstance(data_upper_bounds, torch.Tensor)

        # Initialization from parent class
        CompMixtureLearning.__init__(self, phi, nb_mixtures, 2 * phi.d, sketch, atoms_batch=atoms_batch,
                                                                  **kwargs)

        # Manage bounds
        variance_relative_lower_bound = 1e-4 ** 2
        variance_relative_upper_bound = 0.5 ** 2
        self.max_variance = torch.square(data_upper_bounds - data_lower_bounds).to(self.real_dtype).to(self.device)
        self.lower_data = data_lower_bounds.to(self.real_dtype).to(self.device)
        self.upper_data = data_upper_bounds.to(self.real_dtype).to(self.device)
        self.lower_var = variance_relative_lower_bound * self.max_variance
        self.upper_var = variance_relative_upper_bound * self.max_variance

        # For initialization of Gaussian atoms
        self.sigma2_bar = torch.tensor(sigma2_bar, device=self.device)

        # Compute by block the matrix product with frequency matrix
        self.freq_blocks_batch = freq_blocks_batch

        # Projection step
        assert isinstance(mean_projector, Projector)
        self.mean_projector = mean_projector
        if isinstance(self.mean_projector, ProjectorClip):
            self.mean_projector.lower_bound = self.mean_projector.lower_bound.to(self.real_dtype).to(self.device)
            self.mean_projector.upper_bound = self.mean_projector.upper_bound.to(self.real_dtype).to(self.device)

    def sketch_of_atoms(self, theta, phi):
        """
        Computes and returns A_Phi(P_theta_k) for 1 or K atoms P_theta_k.
        :param theta: tensor of size (2d) or (K, 2d). The first d dimension of atoms is mu, the last d dimensions is
        the diagonal variances sigma.
        :param phi: sk.ComplexExponentialFeatureMap
        :return: tensor of size (K, m)
        """
        assert isinstance(phi, ComplexExponentialFeatureMap)
        mu = theta[..., :self.phi.d]
        sigma = theta[..., -self.phi.d:]
        atoms_sketch = phi.c_norm * gaussian_characteristic_function_diag_covariance(mu, sigma, phi.freq_matrix,
                                                                                     freq_blocks_batch=self.freq_blocks_batch)
        return atoms_sketch

    def randomly_initialize_several_atoms(self, nb_atoms):
        """
        Randomly initialize several Gaussian parameters.
        :param nb_atoms: int
        :return: tensor
        """
        all_new_mu = (self.upper_data - self.lower_data) * torch.rand(nb_atoms, self.phi.d).to(self.device) + self.lower_data
        all_new_sigma = (1.5 - 0.5) * torch.rand(nb_atoms, self.phi.d, device=self.device) + 0.5
        all_new_sigma *= self.sigma2_bar
        new_theta = torch.cat((all_new_mu, all_new_sigma), dim=1)
        return new_theta

    def projection_step(self, theta):
        """
        In GMM, theta[..., :d] is the mean of the Gaussians, while theta[..., -d:]
        :param theta: tensor of size (n_atoms, d) or (d)
        :return:
        """
        # Uniform normalization of the variances
        sigma = theta[..., -self.phi.d:]
        variance_projector = ProjectorClip(self.lower_var * torch.ones_like(sigma), self.upper_var * torch.ones_like(sigma))
        variance_projector.project(sigma)

        # Normalization of the means
        mu = theta[..., :self.phi.d]
        self.mean_projector.project(mu)

    def get_gmm(self, return_numpy=True):
        """
        Return weights, mus and sigmas as diagonal matrices.
        :param return_numpy: bool
        :return:
        """
        weights = self.alphas
        mus = self.all_thetas[:, :self.phi.d]
        sigmas = self.all_thetas[:, -self.phi.d:]
        sigmas_mat = torch.diag_embed(sigmas)
        if return_numpy:
            return weights.cpu().detach().numpy(), mus.cpu().detach().numpy(), sigmas_mat.cpu().detach().numpy()
        return weights, mus, sigmas_mat


class ClompDiagGmm(TorchClomp, CompressiveDiagGmm):
    """
    CLOMP solver for diagonal Gaussian Mixture Modeling (dGMM), where we fit a mixture of K Gaussians
    with diagonal covariances to the sketch.
    The main algorithm is handled by the parent class.
    Torch implementation.
    Init_variance_mode is either "bounds" or "sketch" (default).

    theta: tensor of size (2d), where the first d dimensions represent the mean of the Gaussian, and the last d
    dimensions are the diagonal entries of the covariance matrix.
    """
    pass


class HierarchicalDiagGmm(HierarchicalCompLearning, CompressiveDiagGmm):
    """
    CL Hierarchical Splitting solver for diagonal Gaussian Mixture Modeling (dGMM), where we fit a mixture of K
    Gaussians
    with diagonal covariances to the sketch.
    Due to strong overlap, this algorithm is strongly based on CLOMP for GMM algorithm (its the parent class),
    but the core fitting method is overridden.
    Requires the feature map to be Fourier features.
    """

    # New split methods
    def split_one_atom(self, k):
        """Splits the atom at index k in two.
        The first result of the split is replaced at the k-th index,
        the second result is added at the end of the atom list."""

        # Pick the dimension with most variance
        theta_k = self.all_thetas[k]
        (mu, sig) = (theta_k[:self.phi.d], theta_k[-self.phi.d:])
        i_max_var = torch.argmax(sig)

        # Direction and stepsize
        direction_max_var = torch.zeros(self.phi.d).to(self.device)
        direction_max_var[i_max_var] = 1.  # i_max_var-th canonical basis vector in R^d
        max_deviation = torch.sqrt(sig[i_max_var])  # max standard deviation
        if self.verbose:
            print(f'Step size for splitting: {max_deviation}. Direction: {i_max_var}')

        # Split!
        right_split = torch.cat((mu + max_deviation * direction_max_var, sig), dim=0)
        self.add_atom(right_split)  # "Right" split
        left_split = torch.cat((mu - max_deviation * direction_max_var, sig), dim=0)
        self.replace_atom(k, left_split)  # "Left" split


class RandomSplitHierarchicalDiagGmm(HierarchicalCompLearning, CompressiveDiagGmm):
    """
    Random splitting of the Gaussian atoms. Choose random direction.
    """

    def split_one_atom(self, k):
        # Pick the dimension with most variance
        theta_k = self.all_thetas[k]
        (mu, sig) = (theta_k[:self.phi.d], theta_k[-self.phi.d:])

        # Pick a random dimension
        d = self.phi.d
        gaussian = np.random.multivariate_normal(np.zeros(d), np.eye(d))
        split_direction = gaussian / np.linalg.norm(gaussian)
        split_direction = torch.from_numpy(split_direction).to(self.real_dtype).to(self.device)

        # Stepsize
        deviation = torch.sqrt(self.sigma2_bar)  # max standard deviation
        if self.verbose:
            print(f'Step size for splitting: {deviation}.')

        # Split!
        right_split = torch.cat((mu + deviation * split_direction, sig), dim=0)
        self.add_atom(right_split)  # "Right" split
        left_split = torch.cat((mu - deviation * split_direction, sig), dim=0)
        self.replace_atom(k, left_split)  # "Left" split
