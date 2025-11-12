from torch import ones, multinomial
from torch.distributions import MultivariateNormal
from overcomplete.sae.losses import top_k_auxiliary_loss
import torch


def mse(x, x_hat):
    mse = (x - x_hat).square().mean()
    return mse


class ZeroLogProbException(Exception):
    def __init__(self, means, samples):
        super().__init__("Log prob became zero")
        self.samples = samples
        self.means = means


class WeightedICALossFunction:
    """
    The weighted correlation loss (the independence loss).
    """

    def __init__(self, power, number_of_gausses, cuda, z_dim=None):
        super(WeightedICALossFunction, self).__init__()
        self.power = power
        self.number_of_gausses = number_of_gausses
        self.z_dim = z_dim
        self.cuda = cuda
        self.reduction_type = "mean"

    def random_choice_full(self, input, n_samples):
        if n_samples * self.number_of_gausses < input.shape[0]:
            replacement = False
        else:
            replacement = True
        idx = multinomial(
            ones(input.shape[0]),
            n_samples * self.number_of_gausses,
            replacement=replacement,
        )
        sampled = input[idx].reshape(self.number_of_gausses, n_samples, -1)
        return torch.mean(sampled, axis=1)

    def loss(self, z, latent_normalization=True):
        if latent_normalization:
            x = (z - z.mean(axis=0)) / z.std(axis=0)
        else:
            x = z

        if (x == 0).all(dim=0).any(dim=0):
            print((x == 0).all(dim=0).nonzero())

        if x.isnan().any():
            print(x.isnan().any(dim=0).nonzero())
        dim = self.z_dim if self.z_dim is not None else x.shape[1]
        scale = (1 / dim) ** self.power
        sampled_points = self.random_choice_full(x, dim)

        cov_mat = (scale * torch.eye(dim)).repeat(self.number_of_gausses, 1, 1)
        if self.cuda:
            cov_mat = cov_mat.cuda()

        mvn = MultivariateNormal(loc=sampled_points, covariance_matrix=cov_mat)

        weight_vector = torch.exp(mvn.log_prob(x.reshape(-1, 1, dim)))
        if (weight_vector.sum(dim=0) == 0).any():
            raise ZeroLogProbException(sampled_points, x)
        sum_of_weights = torch.sum(weight_vector, axis=0)

        weight_sum = torch.sum(
            x * weight_vector.T.reshape(self.number_of_gausses, -1, 1), axis=1
        )
        weight_mean = weight_sum / sum_of_weights.reshape(-1, 1)

        xm = x - weight_mean.reshape(self.number_of_gausses, 1, -1)
        wxm = xm * weight_vector.T.reshape(self.number_of_gausses, -1, 1)

        wcov = (wxm.permute(0, 2, 1).matmul(xm)) / sum_of_weights.reshape(-1, 1, 1)

        diag = torch.diagonal(wcov**2, dim1=1, dim2=2)
        diag_pow_plus = diag.reshape(diag.shape[0], diag.shape[1], -1) + diag.reshape(
            diag.shape[0], -1, diag.shape[1]
        )

        tmp = 2 * wcov**2 / diag_pow_plus
        triu = torch.triu(tmp, diagonal=1)
        normalize = 2.0 / (dim * (dim - 1))
        cost = torch.sum(normalize * triu) / self.number_of_gausses
        return cost

    def mse_with_wica_report(self, x, x_hat, pre_codes, codes, dictionary):
        with torch.no_grad():
            wica = self.loss(pre_codes)
        loss = mse(x, x_hat)

        return loss, wica.item()

    def topk_auxillary_loss_with_wica_report(
        self, x, x_hat, pre_codes, codes, dictionary
    ):
        with torch.no_grad():
            wica = self.loss(pre_codes)
        loss = top_k_auxiliary_loss(x, x_hat, pre_codes, codes, dictionary)

        return loss, wica.item()

    def topk_wica(self, x, x_hat, pre_codes, codes, dictionary):
        wica = self.loss(pre_codes)
        loss = top_k_auxiliary_loss(x, x_hat, pre_codes, codes, dictionary)
        combined_loss = loss + wica

        return combined_loss, wica.item()
