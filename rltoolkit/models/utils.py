import torch.nn as nn


def hard_target_update(src: nn.Module, tgt: nn.Module) -> None:
    tgt.load_state_dict(src.state_dict())


def soft_target_update(src: nn.Module, tgt: nn.Module, tau=0.005) -> None:
    for src_param, tgt_param in zip(src.parameters(), tgt.parameters()):
        tgt_param.data.copy_(tau * src_param.data +
                             (1.0 - tau) * tgt_param.data)
