import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def noise_robust_dice(pr,
                      gt
                      ,
                      eps=1e-7,
                      gamma=1.5,
                      threshold=None,
                      ignore_channels=None):
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
    def __init__(self,
                 eps=1.,
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
            channel for channel in range(xs[0].shape[1])
            if channel not in ignore_channels
        ]
        xs = [
            torch.index_select(x,
                               dim=1,
                               index=torch.tensor(channels).to(x.device))
            for x in xs
        ]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x
