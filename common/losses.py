import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def noise_robust_dice(
    pr, gt, eps=1e-7, gamma=1.5, threshold=None, ignore_channels=None
):
    """Calculate Noise Robust Dice cofficient between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        gamma: scalar, [1.0, 2.0].
            When γ = 2.0, LNR-Dice equals to the Dice loss LDice.
            When γ = 1.0, LNR-Dice becomes a weighted version of LMAE.
        threshold: threshold for outputs binarization
    Returns:
        float: dice score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(torch.pow(torch.abs(pr - gt), gamma))
    union = torch.sum(torch.square(gt)) + torch.sum(torch.square(pr)) + eps
    return 1 - intersection / union


class NoiseRobustDiceLoss(nn.Module):
    def __init__(
        self,
        eps=1.0,
        gamma=1.5,
        ignore_channels=None,
    ):
        super().__init__()
        self.eps = eps
        self.activation = nn.Sigmoid()
        self.gamma = gamma
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - noise_robust_dice(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=None,
            gamma=self.gamma,
            ignore_channels=self.ignore_channels,
        )


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [
            channel
            for channel in range(xs[0].shape[1])
            if channel not in ignore_channels
        ]
        xs = [
            torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device))
            for x in xs
        ]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


class CrossEntropyLoss2d(nn.Module):
    """Cross-entropy.
    See: http://cs231n.github.io/neural-networks-2/#losses
    """

    def __init__(self, weight=None):
        """Creates an `CrossEntropyLoss2d` instance.
        Args:
          weight: rescaling weight for each class.
        """

        super().__init__()
        self.nll_loss = nn.NLLLoss(weight)

    def forward(self, inputs, targets):
        return self.nll_loss(nn.functional.log_softmax(inputs, dim=1), targets)


class FocalLoss2d(nn.Module):
    """Focal Loss.
    Reduces loss for well-classified samples putting focus on hard mis-classified samples.
    See: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, gamma=2, weight=None):
        """Creates a `FocalLoss2d` instance.
        Args:
          gamma: the focusing parameter; if zero this loss is equivalent with `CrossEntropyLoss2d`.
          weight: rescaling weight for each class.
        """

        super().__init__()
        self.nll_loss = nn.NLLLoss(weight)
        self.gamma = gamma

    def forward(self, inputs, targets):
        penalty = (1 - nn.functional.softmax(inputs, dim=1)) ** self.gamma
        return self.nll_loss(
            penalty * nn.functional.log_softmax(inputs, dim=1), targets
        )


class mIoULoss2d(nn.Module):
    """mIoU Loss.
    See:
      - http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf
      - http://www.cs.toronto.edu/~wenjie/papers/iccv17/mattyus_etal_iccv17.pdf
    """

    def __init__(self, weight=None):
        """Creates a `mIoULoss2d` instance.
        Args:
          weight: rescaling weight for each class.
        """

        super().__init__()
        self.nll_loss = nn.NLLLoss(weight)

    def forward(self, inputs, targets):

        N, C, H, W = inputs.size()

        softs = nn.functional.softmax(inputs, dim=1).permute(1, 0, 2, 3)
        masks = (
            torch.zeros(N, C, H, W)
            .to(targets.device)
            .scatter_(1, targets.view(N, 1, H, W), 1)
            .permute(1, 0, 2, 3)
        )

        inters = softs * masks
        unions = (softs + masks) - (softs * masks)

        miou = (
            1.0 - (inters.view(C, N, -1).sum(2) / unions.view(C, N, -1).sum(2)).mean()
        )

        return max(
            miou, self.nll_loss(nn.functional.log_softmax(inputs, dim=1), targets)
        )


class LovaszLoss2d(nn.Module):
    """Lovasz Loss.
    See: https://arxiv.org/abs/1705.08790
    """

    def __init__(self):
        """Creates a `LovaszLoss2d` instance."""
        super().__init__()

    def forward(self, inputs, targets):

        N, C, H, W = inputs.size()
        masks = (
            torch.zeros(N, C, H, W)
            .to(targets.device)
            .scatter_(1, targets.view(N, 1, H, W), 1)
        )

        loss = 0.0

        for mask, input in zip(masks.view(N, -1), inputs.view(N, -1)):

            max_margin_errors = 1.0 - ((mask * 2 - 1) * input)
            errors_sorted, indices = torch.sort(max_margin_errors, descending=True)
            labels_sorted = mask[indices.data]

            inter = labels_sorted.sum() - labels_sorted.cumsum(0)
            union = labels_sorted.sum() + (1.0 - labels_sorted).cumsum(0)
            iou = 1.0 - inter / union

            p = len(labels_sorted)
            if p > 1:
                iou[1:p] = iou[1:p] - iou[0:-1]

            loss += torch.dot(nn.functional.relu(errors_sorted), iou)

        return loss / N
