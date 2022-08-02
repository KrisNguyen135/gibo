from typing import Tuple

import torch
import gpytorch
import botorch

from src.cholesky import one_step_cholesky
from src.utils import plot_along_direction

import matplotlib.pyplot as plt

from gpytorch.utils.cholesky import psd_safe_cholesky


class GradientInformation(botorch.acquisition.AnalyticAcquisitionFunction):
    """Acquisition function to sample points for gradient information.

    Attributes:
        model: Gaussian process model that supplies the Jacobian (e.g. DerivativeExactGPSEModel).
    """

    def __init__(self, model):
        """Inits acquisition function with model."""
        super().__init__(model)

    def update_theta_i(self, theta_i: torch.Tensor):
        """Updates the current parameters.

        This leads to an update of K_xX_dx.

        Args:
            theta_i: New parameters.
        """
        if not torch.is_tensor(theta_i):
            theta_i = torch.tensor(theta_i)
        self.theta_i = theta_i
        self.update_K_xX_dx()

    def update_K_xX_dx(self):
        """When new x is given update K_xX_dx."""
        # Pre-compute large part of K_xX_dx.
        X = self.model.train_inputs[0]
        x = self.theta_i.view(-1, self.model.D)
        self.K_xX_dx_part = self._get_KxX_dx(x, X)

    def _get_KxX_dx(self, x, X) -> torch.Tensor:
        """Computes the analytic derivative of the kernel K(x,X) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D) The derivative of K(x,X) w.r.t. x.
        """
        N = X.shape[0]
        n = x.shape[0]
        K_xX = self.model.covar_module(x, X).evaluate()
        lengthscale = self.model.covar_module.base_kernel.lengthscale.detach()
        return (
            -torch.eye(self.model.D, device=X.device)
            / lengthscale ** 2
            @ (
                (x.view(n, 1, self.model.D) - X.view(1, N, self.model.D))
                * K_xX.view(n, N, 1)
            ).transpose(1, 2)
        )

    # TODO: nicer batch-update for batch of thetas.
    @botorch.utils.transforms.t_batch_mode_transform(expected_q=1)
    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """Evaluate the acquisition function on the candidate set thetas.

        Args:
            thetas: A (b) x D-dim Tensor of (b) batches with a d-dim theta points each.

        Returns:
            A (b)-dim Tensor of acquisition function values at the given theta points.
        """
        sigma_n = self.model.likelihood.noise_covar.noise
        D = self.model.D
        X = self.model.train_inputs[0]
        x = self.theta_i.view(-1, D)

        variances = []
        for theta in thetas:
            theta = theta.view(-1, D)
            # Compute K_Xθ, K_θθ (do not forget to add noise).
            K_Xθ = self.model.covar_module(X, theta).evaluate()
            K_θθ = self.model.covar_module(theta).evaluate() + sigma_n * torch.eye(
                K_Xθ.shape[-1]
            ).to(theta)

            # Get Cholesky factor.
            L = one_step_cholesky(
                top_left=self.model.get_L_lower().transpose(-1, -2),
                K_Xθ=K_Xθ,
                K_θθ=K_θθ,
                A_inv=self.model.get_KXX_inv(),
            )

            # Get K_XX_inv.
            K_XX_inv = torch.cholesky_inverse(L, upper=True)

            # get K_xX_dx
            K_xθ_dx = self._get_KxX_dx(x, theta)
            K_xX_dx = torch.cat([self.K_xX_dx_part, K_xθ_dx], dim=-1)

            # Compute_variance.
            variance_d = -K_xX_dx @ K_XX_inv @ K_xX_dx.transpose(1, 2)
            variances.append(torch.trace(variance_d.view(D, D)).view(1))

            with torch.no_grad():
                distance = torch.norm(theta - x).item()

                with open("distance.txt", "a") as f:
                    f.write(f"{distance}\n")
                with open("variance.txt", "a") as f:
                    f.write(f"{variances[-1].item()}\n")

        return -torch.cat(variances, dim=0)


class DownhillQuadratic(GradientInformation):
    def update_theta_i(self, theta_i: torch.Tensor):
        super().update_theta_i(theta_i)

        self.mean_d, _ = self.model.posterior_derivative(self.theta_i)

    def forward(self, x):
        sigma_f = self.model.covar_module.outputscale.detach()
        sigma_n = self.model.likelihood.noise_covar.noise
        D = self.model.D
        X = self.model.train_inputs[0]

        x = x.view(-1, D)

        # Compute K_Xθ, K_θθ (do not forget to add noise).
        K_Xθ = self.model.covar_module(X, x).evaluate()
        K_θθ = self.model.covar_module(x).evaluate() + sigma_n * torch.eye(
            K_Xθ.shape[-1]
        ).to(x)

        # Get Cholesky factor.
        L = one_step_cholesky(
            top_left=self.model.get_L_lower().mT,
            K_Xθ=K_Xθ,
            K_θθ=K_θθ,
            A_inv=self.model.get_KXX_inv(),
        )

        # Get K_XX_inv.
        K_XX_inv = torch.cholesky_inverse(L, upper=True)

        # get K_xX_dx
        K_xθ_dx = self._get_KxX_dx(self.theta_i.view(-1, D), x)
        K_xX_dx = torch.cat([self.K_xX_dx_part, K_xθ_dx], dim=-1)

        # Compute_variance.
        covar_xstar_xstar_condx = (
            self.model._get_Kxx_dx2() - K_xX_dx @ K_XX_inv @ K_xX_dx.mT
        )

        # if self.model.train_inputs[0].size(0) == 3:
        #     from IPython.core.debugger import set_trace
        #     set_trace()

        L_xstar_xstar_condx = psd_safe_cholesky(covar_xstar_xstar_condx)
        covar_xstar_x = (
            K_xθ_dx - self.K_xX_dx_part @ self.model.get_KXX_inv() @ K_Xθ
        )
        covar_x_x = K_θθ - K_Xθ.mT @ self.model.get_KXX_inv() @ K_Xθ

        # from Kaiwen
        Lxx = psd_safe_cholesky(covar_x_x)
        A = torch.triangular_solve(covar_xstar_x.mT, Lxx, upper=False).solution.mT

        LinvMu = torch.triangular_solve(
            self.mean_d.unsqueeze(-1), L_xstar_xstar_condx, upper=False
        ).solution
        LinvA = torch.triangular_solve(A, L_xstar_xstar_condx, upper=False).solution

        with torch.no_grad():
            distance = torch.norm(self.theta_i.view(-1, D) - x).item()

            with open("distance.txt", "a") as f:
                f.write(f"{distance}\n")
            with open("acq_func_val.txt", "a") as f:
                f.write(f"{(LinvMu.square().sum() + LinvA.square().sum()).item()}\n")

        return torch.atleast_1d(LinvMu.square().sum() + LinvA.square().sum())


def optimize_acqf_custom_bo(
    acq_func: botorch.acquisition.AcquisitionFunction,
    bounds: torch.Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to optimize the GradientInformation acquisition function for custom Bayesian optimization.

    Args:
        acq_function: An AcquisitionFunction.
        bounds: A 2 x D tensor of lower and upper bounds for each column of X.
        q: The number of candidates.
        num_restarts: The number of starting points for multistart acquisition function optimization.
        raw_samples: The number of samples for initialization.

    Returns:
        A two-element tuple containing:
            - a q x D-dim tensor of generated candidates.
            - a tensor of associated acquisition values.
    """
    candidates, acq_value = botorch.optim.optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=q,  # Analytic acquisition function.
        num_restarts=num_restarts,
        raw_samples=raw_samples,  # Used for initialization heuristic.
        options={"nonnegative": True, "batch_limit": 5},
        return_best_only=True,
        sequential=False,
    )
    # Observe new values.
    new_x = candidates.detach()
    return new_x, acq_value


def optimize_acqf_vanilla_bo(
    acq_func: botorch.acquisition.AcquisitionFunction, bounds: torch.Tensor
) -> torch.Tensor:
    """Function to optimize the acquisition function for vanilla Bayesian optimization.

    For instance for expected improvement (botorch.acquisition.analytic.ExpectedImprovement).

    Args:
        acq_function: An AcquisitionFunction.
        bounds: A 2 x D tensor of lower and upper bounds for each column of X.

    Returns:
        A q x D-dim tensor of generated candidates.
    """
    candidates, _ = botorch.optim.optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=1,  # Analytic acquisition function.
        num_restarts=5,
        raw_samples=64,  # Used for initialization heuristic.
        options={},
    )
    # Observe new values.
    new_x = candidates.detach()
    return new_x
