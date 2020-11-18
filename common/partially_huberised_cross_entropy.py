import torch
from torch.autograd import Variable
from torch.nn.functional import log_softmax, nll_loss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.tensor import Tensor

# See https://openreview.net/pdf?id=rklB76EKPr


def partially_huberised_cross_entropy(
    input,
    target,
    tau,
    weight=None,
    ignore_index=-100,
):
    log_p = log_softmax(input, 1)
    p = torch.exp(log_p)
    return torch.where(
        p <= 1.0 / tau,
        -tau * p + torch.log(tau) + 1,
        nll_loss(log_p, target, weight, None, ignore_index, None, "mean"),
    ).mean()


class PartiallyHuberisedCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, tau: float = 10.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = Tensor([tau]).float()
        self.register_buffer("tau_const", self.tau)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return partially_huberised_cross_entropy(
            input,
            target,
            tau=Variable(self.tau_const),
            weight=self.weight,
            ignore_index=self.ignore_index,
        )
