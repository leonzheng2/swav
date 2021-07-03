"""
Methods for fast transforms in PyTorch.
Leon Zheng
"""

import math
import torch
import numpy as np
from src.pycle_gpu.sketching.sampling import draw_frequencies_adapted_radius, sample_from_pdf
from src.pycle_gpu.structure.hadamard import hadamard_transform


class FrequencyMatrix:
    def __init__(self, dim_freq, nb_freq, device, dtype):
        """ Frequency matrix is of size (d, m) """
        self.d = dim_freq
        self.m = nb_freq
        self.device = device
        self.dtype = dtype

    def sampling(self, **kwargs):
        """ Define sampling strategy, and instantiate to object containing result of sampling """
        raise NotImplementedError

    # def apply(self, tensor):
    #     """ Compute Omega @ x"""
    #     raise NotImplementedError

    def transpose_apply(self, tensor):
        """ Compute Omega^T @ x"""
        raise NotImplementedError

    def move_to(self, device):
        """ Move the object to given device """
        raise NotImplementedError

    def get_dense_frequency_matrix(self):
        """ Move the object to given device """
        raise NotImplementedError


class DenseFrequencyMatrix(FrequencyMatrix):
    def __init__(self, dim_freq, nb_freq, device, dtype):
        super(DenseFrequencyMatrix, self).__init__(dim_freq, nb_freq, device, dtype)
        self.omega = torch.empty(dim_freq, nb_freq, device=device)
        self.d, self.m = self.omega.shape

    def sampling(self, sigma, k_means):
        omega = draw_frequencies_adapted_radius(self.d, self.m, sigma, k_means=k_means)
        omega = torch.from_numpy(omega).to(self.dtype)
        self.omega = omega.to(self.device)

    # def apply(self, tensor):
    #     return torch.matmul(self.omega, tensor)

    def transpose_apply(self, tensor, omega=None):
        if omega is None:
            omega = self.omega
        assert tensor.shape[-1] == self.d
        return torch.matmul(tensor, omega)

    def move_to(self, device):
        self.device = device
        self.omega = self.omega.to(device)

    def get_dense_frequency_matrix(self, return_numpy=False):
        """ Return the dense frequency matrix """
        return self.omega


class TripleRademacherFrequencyMatrix(FrequencyMatrix):
    """
    Implementation of Omega = [B_1, ..., B_b],
    where B_i = M_i R_i = 1/d^{3/2} D^i_1 H D^i_2 H D^i_3 H R_i
    and B_i^T = R_i M_i^T = 1/d^{3/2} R_i H D^i_3 H D^i_2 H D^i_1
    """
    def __init__(self, dim_freq, nb_freq, device, dtype):
        super(TripleRademacherFrequencyMatrix, self).__init__(dim_freq, nb_freq, device, dtype)
        self.d_pad = 2**(math.ceil(np.log2(dim_freq)))
        self.nb_block = math.ceil(nb_freq / self.d_pad)
        self.m_pad = self.nb_block * self.d_pad
        self.radii = torch.empty(self.nb_block, self.d_pad, device=self.device, dtype=self.dtype)
        self.diag = torch.empty(3, self.nb_block, self.d_pad, device=self.device, dtype=self.dtype)

    def move_to(self, device):
        self.device = device
        self.radii = self.radii.to(self.device)
        self.diag = self.diag.to(self.device)

    def sampling(self, pdf_radius=None):
        if pdf_radius is not None:
            r = np.linspace(0, 5, 2001)
            sampled_radii = sample_from_pdf(pdf_radius(r), r, n_samples=self.nb_block * self.d_pad)
        else:
            sampled_radii = np.ones(self.nb_block * self.d_pad)
        self.radii = torch.from_numpy(sampled_radii).view(self.nb_block, self.d_pad).to(self.dtype).to(self.device)
        self.diag = 2 * torch.randint_like(self.diag, 2, device=self.device, dtype=self.dtype) - 1

    def scale_radii(self, sigma_bar):
        self.radii = self.radii / sigma_bar

    def pad_tensor(self, tensor):
        assert tensor.shape[-1] <= self.d_pad
        size = list(tensor.size())
        size[-1] = self.d_pad - tensor.shape[-1]
        size = tuple(size)
        return torch.cat((tensor, torch.zeros(size, device=self.device)), -1)

    def transpose_apply(self, tensor, diag_1=None, diag_2=None, diag_3=None, radii=None):
        """
        Compute fast transform Omega^T @ X, where X is represented by tensor
        :param tensor: size (..., d)
        :param diag_1: useful for block computing
        :param diag_2: useful for block computing
        :param diag_3: useful for block computing
        :param radii: useful for block computing
        :return: tensor of size (..., m)
        """
        assert tensor.size()[-1] == self.d
        if diag_1 is None:
            diag_1 = self.diag[0]
        if diag_2 is None:
            diag_2 = self.diag[1]
        if diag_3 is None:
            diag_3 = self.diag[2]
        if radii is None:
            radii = self.radii
        tensor = self.pad_tensor(tensor)
        tensor = torch.unsqueeze(tensor, dim=-2)
        tensor = diag_1 * tensor
        tensor = hadamard_transform(tensor)
        tensor = diag_2 * tensor
        tensor = hadamard_transform(tensor)
        tensor = diag_3 * tensor
        tensor = hadamard_transform(tensor)
        tensor = radii * tensor
        tensor = tensor / np.sqrt(self.d_pad ** 3)
        size = list(tensor.shape)
        size[-2] = size[-2] * size[-1]
        size.pop(-1)
        return tensor.view(size)[..., :self.m]

    def get_dense_frequency_matrix(self):
        tensor = self.transpose_apply(torch.eye(self.d, device=self.device, dtype=self.dtype))
        return torch.transpose(tensor, 0, 1)
