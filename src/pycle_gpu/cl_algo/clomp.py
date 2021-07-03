import time
import torch
from src.pycle_gpu.cl_algo.solver import CompMixtureLearning


class TorchClomp(CompMixtureLearning):
    """
    Template for Compressive Learning with Orthogonal Matching Pursuit (CL-OMP) solver,
    used to the CL problem
        min_(theta) || sketch_weight * z - A_Phi(P_theta) ||_2,
    where $P_theta = \sum_{k=1}^K \alpha_k A_Phi(P_theta_k)$ is a weighted mixture composed of K components P_theta_k,
    hence the problem to solve becomes
        min_(alpha,theta_k) || sketch_weight * z - sum_k alpha_k*A_Phi(P_theta_k) ||_2.
    The CLOMP algorithm works by adding new elements to the mixture one by one.
    Torch implementation.
    """

    def fit_once(self, tensorboard=False):
        """
        CLOMP-R algorithm implementation.
        If random_restart is True, constructs a new solution from scratch with CLOMP-R, else fine-tune.
        """
        n_iterations = 2 * self.nb_mixtures
        all_new_theta = self.randomly_initialize_several_atoms(n_iterations)
        for i_iter in range(n_iterations):
            if self.verbose:
                print(f'Iteration {i_iter + 1} / {n_iterations}')
            # Step 1: find new atom theta most correlated with residual
            new_theta = all_new_theta[i_iter]
            since = time.time()
            new_theta = self.maximize_atom_correlation(new_theta, tensorboard=tensorboard)
            if self.verbose:
                print(f'Time for step 1: {time.time() - since}')
            # Step 2: add it to the support
            self.add_atom(new_theta)
            # Step 3: if necessary, hard-threshold to enforce sparsity
            if self.n_atoms > self.nb_mixtures:
                since = time.time()
                beta = self.find_optimal_weights(normalize_atoms=True, tensorboard=tensorboard)
                index_to_remove = torch.argmin(beta).to(torch.long)
                self.remove_atom(index_to_remove)
                if self.verbose:
                    print(f'Time for step 3: {time.time() - since}')
                if index_to_remove == self.nb_mixtures:
                    continue
            # Step 4 and 5
            self.do_step_4_5(tensorboard=tensorboard)

        # Final fine-tuning with increased optimization accuracy
        self.final_fine_tuning()
