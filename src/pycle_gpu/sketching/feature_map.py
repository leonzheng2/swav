"""
Sketching methods. Torch implementation, inspired from Pycle toolbox.
Leon Zheng
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.pycle_gpu.sketching.frequency_matrix import FrequencyMatrix, DenseFrequencyMatrix, TripleRademacherFrequencyMatrix


class ComplexExponentialFeatureMap:
    """
    Complex exponential feature map, with implementation in torch.
    Omega is of size d \times m. Tensor stored in the required device.
    """
    def __init__(self, freq_matrix):
        """
        :param omega: (d, m) tensor
        :param device: torch device
        """
        assert isinstance(freq_matrix, FrequencyMatrix)
        self.freq_matrix = freq_matrix
        self.device = freq_matrix.device
        self.d = self.freq_matrix.d
        self.m = self.freq_matrix.m
        self.dtype = self.freq_matrix.dtype
        self.c_norm = 1. / np.sqrt(self.m)

    def move_to(self, device):
        """
        Move frequencies to device.
        :param device: torch.device
        :return: new phi with moved device
        """
        self.freq_matrix.move_to(device)

    def __call__(self, x):
        """
        Compute exp(-i \Omega_j^T x) on each component j.
        :param x: batch of vectors of dimension d. Should be in GPU if cuda == True
        :return: tensor
        """
        x = -1j * self.freq_matrix.transpose_apply(x)
        return self.c_norm * torch.exp(x)


# Compute sketch
def compute_sketch_rff(dataloader, phi):
    """
    Efficient implementation of sketching computation, with Phi = random fourier features.
    Equirepartition of the weights over the dataset, i.e. 1/n.

    :param dataloader: dataloader of the dataset which is a (n, d)-tensor array, not in cuda.
    :param phi: ComplexExponentialFeatureMap object, containing frequencies as (d,m) tensor in cuda.
    :return: return sketch of size (m)
    """
    assert isinstance(dataloader, DataLoader)
    dataset = dataloader.dataset
    assert isinstance(dataset, torch.Tensor)
    (n, d) = dataset.size()

    # Determine the sketch dimension and sanity check: the dataset is nonempty and the map works
    if isinstance(phi, ComplexExponentialFeatureMap):  # featureMap is the argument, FeatureMap is the class
        m = phi.m
    else:
        raise ValueError("featureMap is not instance of complexExpFeatureMap")

    # Using GPU with torch
    sketch = torch.zeros(m, dtype=torch.complex64).to(phi.device)
    for batch in dataloader:
        batch = batch.to(phi.device)
        complex_exp = phi(batch)
        sketch += torch.sum(complex_exp, dim=0)
    return 1./n * sketch


# Characteristic function of a Gaussian parametrised by (mu, sigma), evaluated at frequencies omega.
def gaussian_characteristic_function(mu, sigma_mat, omega):
    """
    Characteristic function of a Gaussian parametrised by (mu, sigma), evaluated at frequencies omega.
    Computation can be for 1 parameter (mu, sigma) or for K parameters (mu, sigma).
    Evaluation at m frequencies.
    Dimension is d.
    :param mu: size (d) or (K, d)
    :param sigma_mat: size (d, d) or (K, d, d)
    :param omega: size (d, m)
    :return: tensor of size (K, m)
    """
    part1 = -1j*(torch.matmul(mu, omega))
    part2 = torch.matmul(torch.transpose(omega, 0, 1), sigma_mat)
    if len(sigma_mat.size()) == 3:
        part3 = torch.einsum('kij,ji->ki', part2, omega) / 2.
    else:
        part3 = torch.einsum('ij,ji->i', part2, omega) / 2.
    res = torch.exp(part1 - part3)
    return res


def gaussian_characteristic_function_diag_covariance(mu, sigma, freq_mat, freq_blocks_batch=None):
    """
    Characteristic function of a Gaussian parametrised by (mu, sigma), evaluated at frequencies omega.
    Computation can be for 1 parameter (mu, sigma) or for K parameters (mu, sigma).
    Evaluation at m frequencies.
    Dimension is d.
    :param mu: size (d) or (K, d)
    :param sigma: size (d) or (K, d)
    :param freq_mat: FrequencyMatrix object
    :param freq_blocks_batch: int
    :return: tensor of size (m) or (K, m)
    """
    assert isinstance(freq_mat, FrequencyMatrix)

    # All the frequencies at the same time
    if freq_blocks_batch is None:
        if isinstance(freq_mat, DenseFrequencyMatrix):
            tensor = sigma
            tensor = tensor.unsqueeze(-1) * freq_mat.omega
            tensor = freq_mat.omega * tensor
            tensor = - 0.5 * torch.sum(tensor, dim=-2)
        else:
            assert isinstance(freq_mat, TripleRademacherFrequencyMatrix)
            tensor = torch.diag_embed(sigma)
            tensor = freq_mat.transpose_apply(tensor)
            tensor = torch.transpose(tensor, -2, -1)
            tensor = freq_mat.transpose_apply(tensor)
            tensor = - 0.5 * torch.diagonal(tensor, dim1=-2, dim2=-1)
        tensor = -1j * freq_mat.transpose_apply(mu) + tensor
        return torch.exp(tensor)

    # Batch of frequencies block
    if isinstance(freq_mat, TripleRademacherFrequencyMatrix):
        diag_and_radii_dataset = TensorDataset(freq_mat.diag[0], freq_mat.diag[1], freq_mat.diag[2], freq_mat.radii)
        diag_and_radii_loader = DataLoader(diag_and_radii_dataset, batch_size=freq_blocks_batch)
        result = []
        for diag_1, diag_2, diag_3, radii in diag_and_radii_loader:
            tensor = torch.diag_embed(sigma)
            tensor = freq_mat.transpose_apply(tensor, diag_1, diag_2, diag_3, radii)
            tensor = torch.transpose(tensor, -2, -1)
            tensor = freq_mat.transpose_apply(tensor, diag_1, diag_2, diag_3, radii)
            tensor = - 0.5 * torch.diagonal(tensor, dim1=-2, dim2=-1)
            tensor = -1j * freq_mat.transpose_apply(mu, diag_1, diag_2, diag_3, radii) + tensor
            tensor = torch.exp(tensor)
            result.append(tensor)
        return torch.cat(result, dim=-1)
    else:
        assert isinstance(freq_mat, DenseFrequencyMatrix)
        all_omega_transpose = torch.transpose(freq_mat.omega, 0, 1)
        frequency_loader = DataLoader(all_omega_transpose, batch_size=freq_blocks_batch * freq_mat.d)
        result = []
        for omega_transpose in frequency_loader:
            omega = torch.transpose(omega_transpose, 0, 1)
            tensor = sigma
            tensor = tensor.unsqueeze(-1) * omega
            tensor = omega * tensor
            tensor = - 0.5 * torch.sum(tensor, dim=-2)
            tensor = -1j * freq_mat.transpose_apply(mu, omega) + tensor
            tensor = torch.exp(tensor)
            result.append(tensor)
        return torch.cat(result, dim=-1)


# Deprecated
# class SketchFrequenciesDataset(Dataset):
#
#     def __init__(self, sketch, omega):
#         assert isinstance(sketch, torch.Tensor)
#         assert isinstance(sketch, torch.Tensor)
#         assert sketch.size()[-1] == omega.size()[-1]
#         self.sketch = sketch
#         self.omega = omega
#
#     def __len__(self):
#         return len(self.sketch)
#
#     def __getitem__(self, item):
#         return self.sketch[item], self.omega[:, item]
