import os

import numpy as np
import torch
from torch.nn import functional as f
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.pycle_gpu.sketching.frequency_matrix import DenseFrequencyMatrix


class HierarchicalCompressiveGMM:

    def __init__(self, freq_matrix, nb_mixtures, sketch, sigma2_bar, freq_epochs, freq_batch_size, lr, beta_1, beta_2,
                 gamma, step_size, initial_atom_mean, project, verbose):
        assert isinstance(freq_matrix, DenseFrequencyMatrix)
        self.freq_matrix = freq_matrix
        self.nb_mixtures = nb_mixtures
        self.sketch = sketch
        self.verbose = verbose
        self.project = project
        self.sigma2_bar = sigma2_bar
        self.initial_atom_mean = initial_atom_mean

        # Optimization parameters
        self.freq_epochs = freq_epochs
        self.freq_batch_size = freq_batch_size
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.gamma = gamma
        self.step_size = step_size

        # Misc
        self.device = freq_matrix.device
        self.real_dtype = freq_matrix.dtype
        if self.real_dtype == torch.float32:
            self.comp_dtype = torch.complex64
        elif self.real_dtype == torch.float64:
            self.comp_dtype = torch.complex128

        # Initialization
        if self.verbose:
            print('Initialize empty solution...')
        self.n_atoms = 0
        self.all_mus = torch.empty(0, self.freq_matrix.d, dtype=self.real_dtype).to(self.device)
        self.all_sigmas = torch.empty(0, self.freq_matrix.d, dtype=self.real_dtype).to(self.device)
        self.alphas = torch.empty(0, dtype=self.real_dtype).to(self.device)
        if self.verbose:
            print('End of empty solution initialization!')

    def sketch_of_atoms(self, mus, sigmas, freq_matrix):
        assert isinstance(freq_matrix, DenseFrequencyMatrix)
        assert len(mus) == len(sigmas)
        tensor = sigmas
        tensor = tensor.unsqueeze(-1) * freq_matrix.omega
        tensor = freq_matrix.omega * tensor
        tensor = - 0.5 * torch.sum(tensor, dim=-2)
        tensor = -1j * freq_matrix.transpose_apply(mus) + tensor
        return torch.exp(tensor)

    def add_several_atoms(self, mus, sigmas):
        """
        Adding a new atom.
        :param mus: tensor size (n_atoms_to_add, d_atom)
        :param sigmas: tensor size (n_atoms_to_add, d_atom)
        :return:
        """
        assert len(mus) == len(sigmas)
        self.n_atoms += len(mus)
        self.all_mus = torch.cat((self.all_mus, mus), dim=0)
        self.all_sigmas = torch.cat((self.all_sigmas, sigmas), dim=0)

    def remove_all_atoms(self):
        self.n_atoms = 0
        self.all_mus = torch.empty(0, self.freq_matrix.d, dtype=self.real_dtype).to(self.device)
        self.all_sigmas = torch.empty(0, self.freq_matrix.d, dtype=self.real_dtype).to(self.device)

    def split_all_current_thetas_alphas(self):
        print(torch.max(self.all_sigmas, dim=1)[0])
        all_i_max_var = torch.argmax(self.all_sigmas, dim=1).to(torch.long)
        print(f"Splitting directions: {all_i_max_var}")
        all_direction_max_var = f.one_hot(all_i_max_var, num_classes=self.freq_matrix.d)
        all_max_var = self.all_sigmas.gather(1, all_i_max_var.view(-1, 1)).squeeze()
        all_max_deviation = torch.sqrt(all_max_var)
        all_sigma_step = all_max_deviation.unsqueeze(-1) * all_direction_max_var

        # Split !
        right_splitted_mus = self.all_mus + all_sigma_step
        left_splitted_mus = self.all_mus - all_sigma_step
        left_and_right_splitted_sigmas = self.all_sigmas.repeat(2, 1)
        self.remove_all_atoms()
        self.add_several_atoms(torch.cat((left_splitted_mus, right_splitted_mus), dim=0), left_and_right_splitted_sigmas)

        # Split alphas
        self.alphas = self.alphas.repeat(2) / 2.

    def sketch_of_solution(self, alphas, all_mus, all_sigmas, freq_matrix):
        return torch.matmul(alphas.to(self.comp_dtype), self.sketch_of_atoms(all_mus, all_sigmas, freq_matrix))

    def projection_step(self, param):
        norms = torch.norm(param, dim=-1)
        if len(param.shape) == 2:
            indices = norms > 1
            if torch.any(indices):
                param[indices] = param[indices] / norms[indices].unsqueeze(-1)
        else:
            if norms.item() > 1:
                torch.div(param, norms, out=param)

    def minimize_cost_from_current_sol(self, log_dir=None, sub_log_dir=None):
        # Preparing frequencies dataloader
        dataset = TensorDataset(torch.transpose(self.freq_matrix.omega, 0, 1), self.sketch)
        dataloader = DataLoader(dataset, batch_size=self.freq_batch_size)

        # Parameters, optimizer
        log_alphas = torch.log(self.alphas).requires_grad_()
        all_mus = self.all_mus.requires_grad_()
        all_log_sigmas = torch.log(self.all_sigmas).requires_grad_()
        params = [log_alphas, all_mus, all_log_sigmas]
        # Adam optimizer
        optimizer = torch.optim.Adam(params, lr=self.lr, betas=(self.beta_1, self.beta_2))

        if self.step_size > 0:
            scheduler = StepLR(optimizer, self.step_size, gamma=self.gamma)

        # Tensorboard
        if log_dir:
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            event_name = current_time + '_' + socket.gethostname()
            if sub_log_dir:
                event_name = event_name + "_" + sub_log_dir
            comment = f'BATCH_{self.freq_batch_size}_#EPOCHS_{self.freq_epochs}_LR_{self.lr}_BETA1_{self.beta_1}_BETA2_{self.beta_2}_GAMMA_{self.gamma}_STEP_SIZE_{self.step_size}'
            event_name = event_name + "_" + comment
            writer = SummaryWriter(os.path.join(log_dir, event_name))

        for i in range(self.freq_epochs):
            for iter_batch, (freq_transpose_batch, sketch_batch) in enumerate(dataloader):
                # freq_batch is of size (freq_batch_size, nb_freq)
                reduced_freq_mat = DenseFrequencyMatrix(self.freq_matrix.d, self.freq_matrix.m,
                                                        self.freq_matrix.device, self.freq_matrix.dtype)
                reduced_freq_mat.omega = torch.transpose(freq_transpose_batch, 0, 1)
                reduced_sketch_of_sol = self.sketch_of_solution(torch.exp(log_alphas), all_mus,
                                                                torch.exp(all_log_sigmas), reduced_freq_mat)
                loss = torch.square(torch.linalg.norm(sketch_batch - reduced_sketch_of_sol))
                loss.backward()
                optimizer.step()
                # Projection step
                if self.project:
                    with torch.no_grad():
                        self.projection_step(all_mus)
                # Tracking loss
                if log_dir:
                    writer.add_scalar(f"minimize_cost_from_current_sol/epoch-{i}", loss.item(), iter_batch)
                # Scheduler step
                if self.step_size > 0:
                    scheduler.step()
        if log_dir:
            writer.flush()
            writer.close()
        self.alphas = torch.exp(log_alphas).detach()
        self.all_mus = all_mus.detach()
        self.all_sigmas = torch.exp(all_log_sigmas).detach()

    def fit_once(self, runs_dir=None):
        n_iterations = int(np.ceil(np.log2(self.nb_mixtures)))  # log_2(K) iterations
        new_sigma = (1.5 - 0.5) * torch.rand(1, self.freq_matrix.d, device=self.device) + 0.5
        new_sigma = torch.mul(new_sigma, self.sigma2_bar)
        self.add_several_atoms(self.initial_atom_mean, new_sigma)
        self.alphas = torch.ones(1, dtype=self.real_dtype).to(self.device)

        for i_iter in range(n_iterations):
            if self.verbose:
                print(f'Iteration {i_iter + 1} / {n_iterations}')
            print("Splitting all atoms...")
            self.split_all_current_thetas_alphas()
            print("Fine-tuning...")
            self.minimize_cost_from_current_sol(log_dir=runs_dir, sub_log_dir=f'ITER_{i_iter + 1}')

        print("Final fine-tuning...")
        self.minimize_cost_from_current_sol(log_dir=runs_dir, sub_log_dir='FINAL_FINE_TUNING')
        if self.project:
            self.projection_step(self.all_mus)
        self.alphas /= torch.sum(self.alphas)

    def get_gmm(self, return_numpy=True):
        """
        Return weights, mus and sigmas as diagonal matrices.
        :param return_numpy: bool
        :return:
        """
        weights = self.alphas
        sigmas_mat = torch.diag_embed(self.all_sigmas)
        if return_numpy:
            return weights.cpu().detach().numpy(), self.all_mus.cpu().detach().numpy(), sigmas_mat.cpu().detach().numpy()
        return weights, self.all_mus, sigmas_mat
