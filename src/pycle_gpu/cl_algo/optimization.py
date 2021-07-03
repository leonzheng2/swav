import torch
from torch.utils.tensorboard import SummaryWriter

#
# def l2_projection_step(tensor):
#     """
#     If the l2-norm is greater than 1, normalize the tensor.
#     :param tensor: tensor of size (K, d) or (d)
#     :return:
#     """


class Projector:
    def project(self, param):
        raise NotImplementedError


class ProjectorNoProjection(Projector):
    def project(self, param):
        pass


class ProjectorClip(Projector):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def project(self, param):
        assert isinstance(param, torch.Tensor)
        self.upper_bound = self.upper_bound.to(param)

        torch.minimum(param, self.upper_bound, out=param)
        torch.maximum(param, self.lower_bound, out=param)


class ProjectorLessUnit2Norm(Projector):
    def project(self, param):
        norm = torch.norm(param, dim=-1)
        if len(param.shape) == 2:
            indices = norm > 1
            if torch.any(indices):
                param[indices] = param[indices] / norm[indices].unsqueeze(-1)
        else:
            if norm.item() > 1:
                torch.div(param, norm, out=param)


class ProjectorExactUnit2Norm(Projector):
    def project(self, param):
        norm = torch.norm(param, dim=-1)
        torch.div(param, norm.unsqueeze(-1), out=param)


class TorchOptimizer:
    # TODO Write a class for torch optimization, in order to factorize code in CLOMP algo implementation.
    def __init__(self, compute_loss, params, tol, max_iter, projection_step=None,
                 tensorboard=False, name=None):
        """
        Constructor.
        :param compute_loss: function which takes params as input
        :param params: parameters to optimize
        :param tol:
        :param max_iter:
        :param projection_step: function which takes params as input
        :param tensorboard:
        :param name:
        """
        if tensorboard:
            assert name is not None
        self.compute_loss = compute_loss
        self.projection_step = projection_step
        self.params = params
        self.tol = tol
        self.max_iter = max_iter
        self.tensorboard = tensorboard
        self.name = name

    def optimize(self):
        optimizer = torch.optim.Adam(self.params, lr=0.01)
        if self.tensorboard:
            writer = SummaryWriter()
        for i in range(self.max_iter):
            optimizer.zero_grad()
            loss = self.compute_loss(self.params)
            loss.backward()
            optimizer.step()
            if self.projection_step is not None:
                self.projection_step(self.params)
            if i == 0:
                previous_loss = torch.clone(loss)
            else:
                relative_loss_diff = torch.abs(previous_loss - loss) / torch.abs(previous_loss)
                if self.tensorboard:
                    writer.add_scalar(self.name, loss.item(), i)
                if relative_loss_diff.item() <= self.tol:
                    break
                previous_loss = torch.clone(loss)
        if self.tensorboard:
            writer.flush()
            writer.close()


