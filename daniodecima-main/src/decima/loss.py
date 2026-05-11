# Modified from the original Genentech/decima source by the DanioDecima authors
# (Chan Zuckerberg Biohub) on 2026-05-10. See daniodecima-main/FORK_NOTES.md for the scope of
# modifications. Original copyright Genentech, Inc., 2024 (Genentech Non-Commercial
# Software License v1.0).

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TaskWisePoissonMultinomialLoss(nn.Module):

    def __init__(
        self,
        total_weight: float = 1,
        eps: float = 1e-7,
        debug=False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.total_weight = total_weight
        self.debug = debug

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        input = torch.exp(input).squeeze(-1)  # B, T
        target = target.squeeze(-1)  # B, T

        total_target = target.sum(axis=-1)  # B,
        total_input = input.sum(axis=-1)  # B,

        # total count poisson loss, mean across targets
        poisson_term_raw = F.poisson_nll_loss(
            total_input, total_target, log_input=False, reduction="mean"
        )  # B
        poisson_term = self.total_weight * poisson_term_raw  # B,

        # Get multinomial probabilities
        p_input = input / total_input.unsqueeze(1)  # B, T
        log_p_input = torch.log(p_input)  # B, T

        # multinomial loss
        multinomial_dot = -torch.multiply(target, log_p_input)  # B x T
        multinomial_term = multinomial_dot.mean()

        #print(f"total_input: {total_input.mean().item()}, total_target: {total_target.mean().item()}, raw loss: {poisson_term_raw.item()}")

        # Combine
        loss = multinomial_term + poisson_term
        if self.debug:
            print(
                f"Multinomial: {multinomial_term}, Poisson: {poisson_term}"
            )
        return loss, poisson_term_raw, multinomial_term
