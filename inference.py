"""Shared full-model loading and block prediction for inference and evaluation."""
from pathlib import Path

import torch

from train_detction_um import decode_output, nms_distance_um, set_random_seed


def load_model(model_path):
    """Strictly restore the local model architecture on CPU before device transfer."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f'Model checkpoint not found: {model_path}')
    from feature_extract_net import Model_net
    model = Model_net([3, 3])
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and isinstance(state.get('state_dict'), dict):
        state = state['state_dict']
    if not isinstance(state, dict) or not state:
        raise ValueError('Checkpoint must contain a nonempty model state_dict')
    stripped = {key.removeprefix('module.'): value for key, value in state.items()}
    if len(stripped) != len(state):
        raise ValueError('Duplicate checkpoint keys after removing the module. prefix')
    model.load_state_dict(stripped, strict=True)
    return model.eval()


def load_detector(model_path, seed=2025, device='cuda:0'):
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('PGRMamba inference requires an NVIDIA CUDA GPU and the bundled extensions.')
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f'Unavailable CUDA device: {device}')
    set_random_seed(seed)
    return load_model(model_path).to(device).eval()


@torch.no_grad()
def predict_image(model, image, distance_um=6., spacing_xyz=(1., 1., 2.)):
    device = next(model.parameters()).device
    output = model(torch.as_tensor(image[None], device=device))
    return nms_distance_um(decode_output(output[0], 0.5), distance_um, spacing_xyz)
