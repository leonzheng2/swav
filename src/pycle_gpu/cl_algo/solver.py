"""
Torch implementation of compressive learning methods, inspired from the toolbox Pycle.
Leon Zheng
"""

import src.pycle_gpu.sketching.feature_map as feature_map
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import torch.nn.functional as f
from torch.utils.data import DataLoader, TensorDataset


class CompMixtureLearning:
    """
    Template for a compressive learning solver, using torch implementation, to solve the problem
        min_(theta) || sketch_weight * z - A_Phi(P_theta) ||_2.

    Some size of tensors to keep in mind:
        - alphas: (n_atoms,)-tensor, weigths of the mixture elements
        - all_thetas:  (n_atoms,d_atom)-tensor, all the found parameters in matrix form
        - all_atoms: (m,n_atoms)-tensor, the sketch of the found parameters (m is sketch size)
    """

    def __init__(self, phi, nb_mixtures, d_atom, sketch, sketch_weight=1., atoms_batch=None, verbose=False):
        """
        - phi: a ComplexExponentialFeatureMap object
        - sketch: tensor
        - sketch_weight: float, a re-scaling factor for the data sketch
        - verbose: bool
        """
        # Encode feature map
        assert isinstance(phi, feature_map.ComplexExponentialFeatureMap)
        self.phi = phi

        # Encode sketch and sketch weight
        self.sketch = sketch
        assert isinstance(sketch_weight, float) or isinstance(sketch_weight, int)
        self.sketch_weight = sketch_weight
        self.sketch_reweighted = self.sketch_weight * self.sketch

        # Assert sketch and phi are on the same device
        assert phi.device.type == sketch.device.type
        self.device = phi.device

        # Variable type
        self.real_dtype = self.phi.dtype
        if self.real_dtype == torch.float32:
            self.comp_dtype = torch.complex64
        elif self.real_dtype == torch.float64:
            self.comp_dtype = torch.complex128

        # Set other values
        self.nb_mixtures = nb_mixtures
        self.d_atom = d_atom

        # Other minor params
        self.minimum_atom_norm = 1e-15 * np.sqrt(self.d_atom)

        # Atoms batch
        self.atoms_batch = atoms_batch

        # Verbose
        self.verbose = verbose

        # Initialization
        if self.verbose:
            print('Initialize empty solution...')
        self.n_atoms = 0
        self.alphas = torch.empty(0, dtype=self.real_dtype).to(self.device)
        self.all_thetas = torch.empty(0, self.d_atom, dtype=self.real_dtype).to(self.device)
        self.all_atoms = torch.empty(self.phi.m, 0, dtype=self.comp_dtype).to(self.device)
        self.residual = torch.clone(self.sketch_reweighted).to(self.device)
        if self.verbose:
            print('End of empty solution initialization!')

    # Abstract methods
    # ===============
    # Methods that have to be instantiated by child classes
    def sketch_of_atoms(self, all_theta, phi):
        """
        Computes and returns A_Phi(P_theta_k) for all theta_k. Do the computation with torch.
        Return size (m) or (K, m)
        """
        raise NotImplementedError

    def randomly_initialize_several_atoms(self, nb_atoms):
        """
        Define how to initialize a number nb_atoms of new atoms.
        :return: torch tensor for new atoms
        """
        raise NotImplementedError

    def projection_step(self, theta):
        raise NotImplementedError

    def fit_once(self, **kwargs):
        """Optimizes the cost to the given sketch, by starting at the current solution"""
        raise NotImplementedError

    # Generic methods
    # ===============
    # Methods that are general for all instances of this class
    # Instantiation of methods of parent class
    def sketch_of_solution(self, solution):
        """
        Returns the sketch of the solution, A_Phi(P_theta) = sum_k alpha_k A_Phi(P_theta_k).
        In: solution = (all_thetas, alphas)
            phi = sk.ComplexExpFeatureMap
            one_by_one = compute one atom by one atom in case atom computation does not fit in GPU
        Out: sketch_of_solution: (m,)-tensor containing the sketch
        """
        all_thetas, alphas = solution
        if self.atoms_batch:
            thetas_dataset = TensorDataset(all_thetas)
            dataloader = DataLoader(thetas_dataset, batch_size=self.atoms_batch)
            result = []
            for (thetas, ) in dataloader:
                result.append(self.sketch_of_atoms(thetas, self.phi))
            all_atoms = torch.transpose(torch.cat(result, dim=0), 0, 1)
            # all_atoms = torch.empty(self.phi.m, all_thetas.shape[0], dtype=self.comp_dtype, device=self.device)
            # for k, theta in enumerate(all_thetas):
            #     all_atoms[:, k] = self.sketch_of_atoms(theta, self.phi)
        else:
            all_atoms = torch.transpose(self.sketch_of_atoms(all_thetas, self.phi), 0, 1)
        return torch.matmul(all_atoms, alphas.to(self.comp_dtype))

    def add_atom(self, new_theta):
        """
        Adding a new atom.
        :param new_theta: tensor
        :return:
        """
        self.n_atoms += 1
        self.all_thetas = torch.cat((self.all_thetas, torch.unsqueeze(new_theta, 0)), dim=0)
        sketch_atom = self.sketch_of_atoms(new_theta, self.phi)
        self.all_atoms = torch.cat((self.all_atoms, torch.unsqueeze(sketch_atom, 1)), dim=1)

    def remove_atom(self, index_to_remove):
        """
        Remove an atom.
        :param index_to_remove: int
        :return:
        """
        self.n_atoms -= 1
        self.all_thetas = torch.cat((self.all_thetas[:index_to_remove], self.all_thetas[index_to_remove+1:]), dim=0)
        self.all_atoms = torch.cat((self.all_atoms[:, :index_to_remove], self.all_atoms[:, index_to_remove + 1:]),
                                   dim=1)

    def replace_atom(self, index_to_replace, new_theta):
        """
        Replace an atom
        :param index_to_replace: int
        :param new_theta: tensor
        :return:
        """
        self.all_thetas[index_to_replace] = new_theta
        self.all_atoms[:, index_to_replace] = self.sketch_of_atoms(new_theta, self.phi)

    def loss_atom_correlation(self, theta):
        sketch_of_atom = self.sketch_of_atoms(theta, self.phi)
        norm_atom = torch.norm(sketch_of_atom)
        # Trick to avoid division by zero (doesn't change anything because everything will be zero)
        if norm_atom.item() < self.minimum_atom_norm:
            norm_atom = torch.tensor(self.minimum_atom_norm)
        return -1. / norm_atom * torch.real(torch.vdot(sketch_of_atom, self.residual))

    def maximize_atom_correlation(self, new_theta, tol=1e-2, max_iter=1000, tensorboard=False):
        """
        Step 1 in CLOMP-R algorithm. Find most correlated atom. Torch optimization, using Adam.
        :param new_theta: torch tensor for atom
        :param tol: stopping criteria is to stop when the relative difference of loss between
        two consecutive iterations is less than tol.
        :param max_iter: max iterations number for optimization.
        :param tensorboard: set True to plot loss in Tensorboard.
        :return: updated new_theta
        """
        params = [torch.nn.Parameter(new_theta, requires_grad=True)]
        optimizer = torch.optim.Adam(params, lr=0.01)

        if tensorboard:
            writer = SummaryWriter()
        for i in range(max_iter):
            optimizer.zero_grad()
            loss = self.loss_atom_correlation(params[0])
            loss.backward()
            optimizer.step()
            # Projection step
            with torch.no_grad():
                self.projection_step(new_theta)
            if i == 0:
                previous_loss = torch.clone(loss)
            else:
                relative_loss_diff = torch.abs(previous_loss - loss) / torch.abs(previous_loss)
                if tensorboard:
                    writer.add_scalar(f'CLOMP/step1/loss/', loss.item(), i)
                if relative_loss_diff.item() <= tol:
                    break
                previous_loss = torch.clone(loss)
        if tensorboard:
            writer.flush()
            writer.close()
        return new_theta.data.detach()

    def find_optimal_weights(self, normalize_atoms=False, tol=1e-2, max_iter=1000, tensorboard=False):
        """Using the current atoms matrix, find the optimal weights"""
        log_alphas = torch.nn.Parameter(torch.zeros(self.n_atoms, device=self.device), requires_grad=True)
        optimizer = torch.optim.Adam([log_alphas], lr=0.01)
        if tensorboard:
            writer = SummaryWriter()
        for i in range(max_iter):
            optimizer.zero_grad()
            if normalize_atoms:
                all_atoms = f.normalize(self.all_atoms, dim=1, eps=self.minimum_atom_norm)
            else:
                all_atoms = self.all_atoms
            sketch_solution = torch.matmul(all_atoms, torch.exp(log_alphas).to(self.comp_dtype))
            loss = torch.linalg.norm(self.sketch_reweighted - sketch_solution)
            loss.backward()
            optimizer.step()

            if i == 0:
                previous_loss = torch.clone(loss)
            else:
                relative_loss_diff = torch.abs(previous_loss - loss) / torch.abs(previous_loss)
                if tensorboard:
                    writer.add_scalar(f'CLOMP/step3-4/loss/', loss.item(), i)
                if relative_loss_diff.item() <= tol:
                    break
                previous_loss = torch.clone(loss)
        if tensorboard:
            writer.flush()
            writer.close()
        alphas = torch.exp(log_alphas)
        normalized_alphas = alphas / torch.sum(alphas)
        # return torch.exp(log_alphas).detach()
        return normalized_alphas.detach()

    def minimize_cost_from_current_sol(self, tol=5e-3, max_iter=1000, tensorboard=False, stochastic=False):
        """
        Step 5 in CLOMP-R algorithm. At the end of the method, update the parameters self.alphas and self.all_thetas.
        :param tol: float. Stopping criteria for optimization: when the relative difference in loss is less than tol.
        :param max_iter: int. Maximum number of iterations.
        :param tensorboard: set True to plot loss in Tensorboard.
        :param stochastic: stochastic version of cost minimization
        :return:
        """
        # Parameters, optimizer
        log_alphas = torch.log(self.alphas).requires_grad_()
        all_thetas = self.all_thetas.requires_grad_()
        params = [log_alphas, all_thetas]
        optimizer = torch.optim.Adam(params, lr=0.01)

        if tensorboard:
            writer = SummaryWriter()

        for iteration in range(max_iter):
            optimizer.zero_grad()
            # Designing loss
            alphas = torch.exp(log_alphas)
            sketch_solution = self.sketch_of_solution((all_thetas, alphas))
            loss = torch.linalg.norm(self.sketch_reweighted - sketch_solution)
            loss.backward()
            optimizer.step()
            # Projection step
            with torch.no_grad():
                self.projection_step(all_thetas)

            # Tracking loss
            if iteration == 0:
                previous_loss = loss
            else:
                relative_loss_diff = torch.abs(previous_loss - loss) / previous_loss
                if tensorboard:
                    writer.add_scalar('CLOMP/step5/loss', loss.item(), iteration)
                if relative_loss_diff.item() < tol:
                    break
                previous_loss = loss

        if tensorboard:
            writer.flush()
            writer.close()
        self.all_thetas = all_thetas.detach()
        self.alphas = torch.exp(log_alphas).detach()

    def do_step_4_5(self, tensorboard=False):
        """
        Do step 4 and 5 of CLOMP-R.
        :param tensorboard:
        :return:
        """
        # Step 4: project to find weights
        since = time.time()
        self.alphas = self.find_optimal_weights(tensorboard=tensorboard)
        if self.verbose:
            print(f'Time for step 4: {time.time() - since}')
        # Step 5: fine-tune
        since = time.time()
        self.minimize_cost_from_current_sol(tensorboard=tensorboard)
        if self.verbose:
            print(f'Time for step 5: {time.time() - since}')
        # The atoms have changed: we must re-compute their sketches matrix
        self.residual = self.sketch_reweighted - self.sketch_of_solution((self.all_thetas, self.alphas))

    def final_fine_tuning(self):
        print(f'Final fine-tuning...')
        self.minimize_cost_from_current_sol()
        self.projection_step(self.all_thetas)
        if self.verbose:
            print(torch.norm(self.all_thetas[:, :self.phi.d], dim=1))
        self.alphas /= torch.sum(self.alphas)


