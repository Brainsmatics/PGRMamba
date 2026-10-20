"""Check the actual CUDA forward/backward path on a small synthetic volume."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from train_detction_um import set_random_seed
from get_loss import GCA_Loss
from inference import load_detector, predict_image


def main():
    print(f'PyTorch={torch.__version__}, CUDA={torch.version.cuda}')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: run this check on the training GPU host.')
    from feature_extract_net import Model_net
    from mamba_ssm.modules.mamba_simple import Mamba
    import inspect
    print('Mamba source:', inspect.getfile(Mamba))
    set_random_seed(2025)
    model = Model_net([3, 3]).cuda().train()
    assert model.mamba.ehance.fusion_channels == 8
    pge_calls = []
    handle = model.mamba.ehance.register_forward_hook(lambda *_: pge_calls.append(True))
    image = torch.rand(1, 1, 32, 32, 32, device='cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    output = model(image)
    assert output.shape == (1, 4, 8, 8, 8)
    labels = (np.array([[12., 16., 20., 1.]], dtype=np.float32),)
    loss = GCA_Loss(output, labels, torch.device('cuda'), 1.5, 6).get_loss()[0]
    assert torch.isfinite(loss)
    loss.backward()
    assert pge_calls and model.mamba.ehance.av.weight.grad is not None
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    handle.remove()
    optimizer.step()
    with tempfile.TemporaryDirectory() as temp:
        checkpoint = Path(temp) / 'smoke_model.pth'
        torch.save(model.state_dict(), checkpoint)
        restored = load_detector(checkpoint, seed=2025)
        predictions = predict_image(restored, image[0].cpu().numpy())
        assert predictions.ndim == 2 and predictions.shape[1] == 4
        assert np.isfinite(predictions).all()
    print('PASS: full 3D PGE + training RegMamba + GCA backward/optimizer step,')
    print('checkpoint reload and the shared inference path.')


if __name__ == '__main__':
    main()
