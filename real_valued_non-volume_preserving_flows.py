import torch
from torch import nn
# Density estimation using Real NVP
# https://arxiv.org/abs/1605.08803

"""
Coupling layers:
    Divides an input into two parts: x = [x_a, x_b]. The division into two parts
    by dividing the vector x into x_{1:d} and x_{d+1:D}.

        y_a = x_a
        y_b = exp(s(x_a)) \odot x_b + t(x_a)

    where s, t are scaling and transition networks
    This transformation invertible:

        x_b = (y_b - t(y_a)) \cdot exp(-s(y_a))
        x_a = y_a

    The logarithm of the Jacobian-determinant:

        det(J) = \prod^{D-d}_{j=1}exp(s(x_a))_j = exp(\sum^{D-d}_{j=1}s(x_a)_j)
"""
class RealNVP(nn.Module):
    def __init__(self, scaling_network, transition_network, num_flows, prior,
            dimnension, is_dequantization = True):
        self.scaling_network = scaling_network
        self.transition_network = transition_network
        self.num_flows = num_flows
        self.prior = prior
        self.dimension = dimension
        self.is_dequantization = is_dequantization

    def coupling_layer(self, x, index, forward = True):
        """
        x: input or outputs from the previous transformation
        index: determines the index of the transformation
        forward: if True: from x to y (True), from y to x otherwise
        """
        # divides the input into x_a and x_b
        (x_a, x_b) = torch.chunk(x, 2, 1)
        # calculates scaling_network(x_a)
        scaling = self.scaling_network[index](x_a)
        # calculates transition_network(x_a)
        transition = self.transition_network[index](x_a)
        # calculates forward pass (x -> z) or inverse pass (z -> x)
        if forward:
            y_b = (x_b - transition) * torch.exp(-scaling)
        else:
            y_b = torch.exp(scaling) * x_b + transition
        # returns output  y = [y_a, y_b],and scaling for calculating the log-J-
        # acobian-determinant
        return torch.cat((x_a, y_b), 1), scaling

    def permutation_layer(self, x):
        return x.flip(1)

    def forward_layer(self, x):
        # performs full forward pass through coupling and permutation layers
        # transforms [x_a, x_b] to [z_a, z_b]
        # initializes the log-Jacobian-determinant
        # det(J) = \prod^{D-d}_{j=1}exp(s(x_a))_j =
        #   exp(\sum^{D-d}_{j=1}s(x_a)_j)
        log_jacobian_det = x.new_zeros(x.shape[0])
        z = x
        # iterates through all layers
        for i in range(self.num_flows):
            # performs coupling layer
            z, scaling = self.coupling_layer(z, i, forward = True)
            # performs permutation layer
            z = self.permutation_layer(z)
            # calculates log-Jacobian-determinant of the sequence of transforma-
            # tions, summing over all of them
            log_jacobian_det = log_jacobian_det - scaling.sum(dim = 1)
        # returns z and log-Jacobian-determinant
        return z, log_jacobian_det

    def inverse_layer(self, z):
        # performs inverse path
        # transforms [z_a, z_b] to [x_a, x_b]
        # applies all transformations in reverse order
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permutation_layer(x)
            x, _ = self.coupling_layer(x, i, forward = False)
        return x

    def forward(self, x, reduction = 'avg'):
        # calculates the forward pass from x to z, and log-Jacobian-determinant
        z, log_jacobian_det = self.forward_layer(x)
        # calculates objective
        # ln p(x) = ln N(z_0 =
        #   f^{-1}(x)|0, I) - \sum^K_{i=1}(\sum^{D-d}_{j=1}s_k(x^k_a)_j)
        # considers the minimization problem, while looking for the maximum li-
        # kelihood estimate
        #   max_x F(x) <=> min_x -F(x)
        result = -(self.prior.log_prob(z) + log_jacobian_det)
        if reduction == 'sum':
            return result.sum()
        else:
            return result.mean()

    def sample(self, batch_size):
        # samples from prior
        # z ~ N(z|0, 1)
        z = z[:, 0, :]
        # performs inverse path
        x = self.inverse_layer(z)
        return x.view(-1, self.dimension)

