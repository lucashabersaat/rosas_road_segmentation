import torch


def init_tensor(x):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        return x.cpu()
    else:
        return (
            x.contiguous().pin_memory().to(device=device, non_blocking=True)
        )


def np_to_tensor(x, device):
    # allocates tensors from np.arrays
    if device == "cpu":
        return torch.from_numpy(x).cpu()
    else:
        return (
            torch.from_numpy(x)
                .contiguous()
                .pin_memory()
                .to(device=device, non_blocking=True)
        )


def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()


def print_model_memory(model):
    """Print how much memory the model utilized."""
    mem = 0
    for name, param in model.state_dict().items():
        if len(param.shape) != 0:
            tmp_mem = param.element_size() * param.nelement()
            mem += tmp_mem
    print_memory(mem)


def print_tensor_info(t, name=None):
    """Print shape & num elements of a tensor."""
    if name is not None:
        print(name, end=': ')

    print("Size:", list(t.shape), "(" + str(t.nelement()) + ")", end=' ')

    size = t.element_size() * t.nelement()
    print_memory(size)


def print_memory(size):
    """Given a byte number, print out a suitable conversion to kb, mb or gb."""
    if size > 1024 ** 3:
        print(size / 1024 / 1024 / 1024, "gb")
    elif size > 1024 ** 2:
        print(size / 1024 / 1024, "mb")
    elif size > 1024:
        print(size / 1024 / 1024, "kb")
