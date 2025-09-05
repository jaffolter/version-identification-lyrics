from omegaconf import OmegaConf
import torch
from hydra.utils import instantiate


def load_model(cfg: OmegaConf, checkpoint_path: str, device: torch.device):
    model = instantiate(cfg.model)
    state_dict = torch.load(checkpoint_path, map_location=device)
    state_dict.pop("logit_scale", None)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model
