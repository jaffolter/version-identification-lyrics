from torch.optim.lr_scheduler import LambdaLR


def get_linear_warmup_scheduler(optimizer, warmup_steps):
    """
    Create a learning rate scheduler with linear warmup.
    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        warmup_steps (int): Number of steps to linearly increase the learning rate.
    Returns:
        LambdaLR: A PyTorch learning rate scheduler.
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0  # constant after warmup

    return LambdaLR(optimizer, lr_lambda)
