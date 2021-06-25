import torch
from torch import nn
import re


class BaseObject(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name


class Loss(BaseObject):
    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError('Loss should be inherited from `Loss` class')

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError('Loss should be inherited from `BaseLoss` class')

    def __rmul__(self, other):
        return self.__mul__(other)


class SumOfLosses(Loss):
    def __init__(self, l1, l2):
        name = '{} + {}'.format(l1.__name__, l2.__name__)
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def __call__(self, *inputs):
        return self.l1.forward(*inputs) + self.l2.forward(*inputs)


class MultipliedLoss(Loss):
    def __init__(self, loss, multiplier):

        # resolve name
        if len(loss.__name__.split('+')) > 1:
            name = '{} * ({})'.format(multiplier, loss.__name__)
        else:
            name = '{} * {}'.format(multiplier, loss.__name__)
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def __call__(self, *inputs):
        return self.multiplier * self.loss.forward(*inputs)


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


class NoiseRobustDiceLoss(Loss):
    def __init__(self,
                 eps=1.,
                 gamma=1.5,
                 ignore_channels=None,
                 **kwargs):
        super().__init__(**kwargs)
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
