import torch.nn as nn


def hard_target_update(src: nn.Module, tgt: nn.Module) -> None:
    """Hard update model parameters."""

    tgt.load_state_dict(src.state_dict())


def soft_target_update(src: nn.Module, tgt: nn.Module, tau=0.005) -> None:
    """Soft update model parameters.

    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        src: PyTorch model (weights will be copied from)
        tgt: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
    for src_param, tgt_param in zip(src.parameters(), tgt.parameters()):
        tgt_param.data.copy_(tau * src_param.data +
                             (1.0 - tau) * tgt_param.data)
