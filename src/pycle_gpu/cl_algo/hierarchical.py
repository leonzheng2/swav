import time
import numpy as np
import torch
from src.pycle_gpu.cl_algo.solver import CompMixtureLearning

class HierarchicalCompLearning(CompMixtureLearning):
    """
    Hierarchical splitting.
    """

    def split_one_atom(self, k):
        raise NotImplementedError

    # Implementation of abstract methods
    def fit_once(self, tensorboard=False):
        """
        Overwrite the method in TorchClomp.
        :param random_restart: bool. Set True to reinitialize.
        :param tensorboard: bool. Set True to plot loss of objective functions during optimization steps in Tensorboard.
        :return:
        """
        # Initialization
        n_iterations = int(np.ceil(np.log2(self.nb_mixtures))) # log_2(K) iterations

        # Add the starting atom
        new_theta = self.randomly_initialize_several_atoms(1)
        new_theta = self.maximize_atom_correlation(new_theta.squeeze())
        self.add_atom(new_theta)

        # Main loop
        for i_iter in range(n_iterations):
            if self.verbose:
                print(f'Iteration {i_iter + 1} / {n_iterations}')
            # Step 1-2: split the currently selected atoms
            since = time.time()
            for k in range(self.n_atoms):
                self.split_one_atom(k)
            if self.verbose:
                print(f'Time for step 1-2: {time.time() - since}')
            # Step 3: if necessary, hard-threshold to enforce sparsity
            since = time.time()
            while self.n_atoms > self.nb_mixtures:
                if self.verbose:
                    print(f'Remaining atoms to remove: {self.n_atoms - self.nb_mixtures}')
                beta = self.find_optimal_weights(normalize_atoms=True, tensorboard=True)
                index_to_remove = torch.argmin(beta).to(torch.long)
                self.remove_atom(index_to_remove)
                if index_to_remove == self.nb_mixtures:
                    continue
            if self.verbose:
                print(f'Time for step 3: {time.time() - since}')
            # Step 4 and 5
            self.do_step_4_5(tensorboard=tensorboard)

        # Final fine-tuning with increased optimization accuracy
        self.final_fine_tuning()
