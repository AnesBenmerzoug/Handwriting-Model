import torch
from torch.nn.modules import Linear, Module


class GaussianWindow(Module):
    r"""Gaussian Window model

    .. math::

        \begin{array}{lll}
        (\hat{\alpha}_t, \hat{\beta}_t, \hat{\kappa}_t) &=& W_{h^1p}h^{1}_{t} + b_p \\
        \alpha_t &=& \exp(\hat{\alpha}_t) \\
        \beta_t &=& \exp(\hat{\beta}_t) \\
        \kappa_t &=& \kappa_{t-1} + \exp(\hat{\kappa}_t)) \\
        \end{array}

    .. math::

        \begin{array}{lll}
        \phi(t, u) &=& \sum_{k=1}^{K} \alpha^k_t \exp\left( \beta^k_t (\kappa^k_t - u)^2 \right) \\
        w_t &=& \sum_{u=1}^{U} \phi(t, u)c_u \\
        \end{array}

    """

    def __init__(self, input_size: int, n_components: int):
        super(GaussianWindow, self).__init__()
        self.input_size = input_size
        self.n_components = n_components
        self.parameter_layer = Linear(
            in_features=input_size, out_features=3 * n_components
        )

    def forward(
        self,
        input_: torch.Tensor,
        onehot: torch.Tensor,
        prev_kappa: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        abk_hats = self.parameter_layer(input_)
        abk = torch.exp(abk_hats).unsqueeze(3)
        alpha, beta, kappa = abk.chunk(3, dim=2)
        if prev_kappa is not None:
            kappa = kappa + prev_kappa
        u = torch.arange(0, onehot.size(1)).to(input_.device)
        phi = torch.sum(alpha * torch.exp(-beta * ((kappa - u) ** 2)), dim=2)
        window = torch.matmul(phi, onehot)
        return window, kappa, phi

    def __repr__(self):
        s = "{name}(input_size={input_size}, n_components={n_components})"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class MixtureDensityNetwork(Module):
    r"""Mixture Density Network model.

    .. math::

        \begin{array}{lll}
        e_t &=& \frac{1}{1 + \exp(\hat{e}_t)} \\
        \pi^i_t &=& \frac{\exp(\hat{\pi}^i_t)}{\sum_{j^{\prime}=1}^{M} \exp(\hat{\pi}^{j^{\prime}}_t)} \\
        \mu^{j}_t &=& \hat{\mu}^{j}_t \\
        \sigma^{j}_t &=& \exp(\hat{\sigma}^{j}_t) \\
        \rho^{j}_t &=& \tanh(\hat{\rho}^{j}_t) \\
        \end{array}

    .. math::

        \begin{array}{lll}
        \hat{y}_t &=&
        \left( \hat{e}_t, \left\{ \hat{w}^j_t, \hat{\mu}^j_t, \hat{\sigma}^j_t, \hat{\rho}^j_t \right\}^{M}_{j=1} \right) \\
        &=& b_y + \sum_{n=1}^{N} W_{h^ny}h^n_t
        \end{array}

    .. math::

        P\left[ x_{t+1} | y_t \right] = \sum_{j=1}^{M} \pi^j_t \mathcal{N}\left( x_{t+1}|\mu^{j}_t, \sigma^{j}_t, \rho^{j}_t \right)
        \begin{cases}
        e_t, \text{ if } (x_{t+1})_3 = 1\\
        1 - e_t, \text{ otherwise}
        \end{cases}

    """

    def __init__(self, input_size: int, n_mixtures: int):
        super(MixtureDensityNetwork, self).__init__()
        self.input_size = input_size
        self.n_mixtures = n_mixtures
        self.parameter_layer = Linear(
            in_features=input_size, out_features=1 + 6 * n_mixtures
        )

    def forward(
        self, input_: torch.Tensor, bias: torch.Tensor | None = None
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        mixture_parameters = self.parameter_layer(input_)
        eos_hat = mixture_parameters[:, :, 0:1]
        pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = torch.chunk(
            mixture_parameters[:, :, 1:], 6, dim=2
        )
        eos = torch.sigmoid(-eos_hat)
        mu1 = mu1_hat
        mu2 = mu2_hat
        rho = torch.tanh(rho_hat)
        if bias is None:
            bias = torch.zeros_like(rho)
        pi = torch.softmax(pi_hat * (1 + bias), dim=2)
        sigma1 = torch.exp(sigma1_hat - bias)
        sigma2 = torch.exp(sigma2_hat - bias)
        return eos, pi, mu1, mu2, sigma1, sigma2, rho

    def __repr__(self):
        s = "{name}(input_size={input_size}, n_mixtures={n_mixtures})"
        return s.format(name=self.__class__.__name__, **self.__dict__)
