import numpy as np
import torch
from src.pycle_gpu.cl_algo.solver import CompMixtureLearning
from src.pycle_gpu.cl_algo.clomp import TorchClomp
from src.pycle_gpu.cl_algo.hierarchical import HierarchicalCompLearning
from src.pycle_gpu.cl_algo.optimization import Projector, ProjectorClip


class CompressiveKMeans(CompMixtureLearning):

    def __init__(self, phi, nb_clusters, lower_bounds, upper_bounds, centroid_projector,
                 sketch, **kwargs):
        """ Lower and upper bounds are for random initialization, not for projection step ! """
        CompMixtureLearning.__init__(self, phi, nb_clusters, phi.d, sketch, **kwargs)
        # Bound for random initialization
        self.lower_bounds = lower_bounds.to(self.real_dtype).to(self.device)
        self.upper_bounds = upper_bounds.to(self.real_dtype).to(self.device)
        assert isinstance(centroid_projector, Projector)
        self.centroid_projector = centroid_projector
        if isinstance(self.centroid_projector, ProjectorClip):
            self.centroid_projector.lower_bound = self.centroid_projector.lower_bound.to(self.real_dtype).to(self.device)
            self.centroid_projector.upper_bound = self.centroid_projector.upper_bound.to(self.real_dtype).to(self.device)

    def sketch_of_atoms(self, theta, phi):
        """
        Computes and returns A_Phi(P_theta_k) for one or several atoms P_theta_k.
        d is the dimension of atom, m is the dimension of sketch
        :param theta: tensor of size (d) or (K, d)
        :param phi: sk.ComplexExponentialFeatureMap object
        :return: tensor of size (m) or (K, m)
        """
        assert theta.size()[-1] == self.d_atom
        return phi(theta)

    def randomly_initialize_several_atoms(self, nb_atoms):
        """
        Uniform initialization of several centroids between the lower and upper bounds.
        :param nb_atoms: int. Number of atoms to initialize randomly.
        :return: tensor
        """
        all_new_theta = (self.upper_bounds -
                         self.lower_bounds) * torch.rand(nb_atoms, self.d_atom).to(self.device) + self.lower_bounds
        return all_new_theta

    def projection_step(self, theta):
        self.centroid_projector.project(theta)

    def get_centroids(self, return_numpy=True):
        """
        Get centroids, in numpy array if required.
        :param return_numpy: bool
        :return: tensor or numpy array
        """
        if return_numpy:
            return self.all_thetas.cpu().numpy()
        return self.all_thetas


class ClompCkm(TorchClomp, CompressiveKMeans):
    """
    CLOMP solver for Compressive K-Means (CKM), where we fit a mixture of K Diracs to the sketch.
    The main algorithm is handled by the parent class.

    theta: tensor of size (d), representing the centroid of dimension d.
    """
    pass


class HierarchicalCkm(HierarchicalCompLearning, CompressiveKMeans):
    """
    CL Hierarchical Splitting solver for Compressive K-means
    Due to strong overlap, this algorithm is strongly based on CLOMP for GMM algorithm (its the parent class),
    but the core fitting method is overridden.
    Requires the feature map to be Fourier features.
    """
    def __init__(self, sigma2_bar, *args, **kwargs):
        """ sigma2_bar is torch """
        CompressiveKMeans.__init__(self, *args, **kwargs)
        self.sigma2_bar = sigma2_bar

    # New split methods
    def split_one_atom(self, k):
        """Splits the atom at index k in two.
        The first result of the split is replaced at the k-th index,
        the second result is added at the end of the atom list."""
        theta_k = self.all_thetas[k]

        # Pick a random dimension
        d = self.phi.d
        gaussian = np.random.multivariate_normal(np.zeros(d), np.eye(d))
        split_direction = gaussian / np.linalg.norm(gaussian)
        split_direction = torch.from_numpy(split_direction).to(self.real_dtype).to(self.device)

        # Stepsize
        deviation = np.sqrt(self.sigma2_bar)  # max standard deviation
        if self.verbose:
            print(f'Step size for splitting: {deviation}.')

        # Split
        right_split = theta_k + deviation * split_direction
        self.add_atom(right_split)  # "Right" split
        left_split = theta_k - deviation * split_direction
        self.replace_atom(k, left_split)  # "Left" split
