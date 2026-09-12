"""保存/恢复后，下一步采样、随机前向和AdamW更新须与连续运行一致。"""
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_blurball_midpoint as training


def check(device):
    torch.manual_seed(17)
    rng = np.random.default_rng(23)

    def build():
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.3),
                                    torch.nn.Linear(4, 1)).to(device)
        return model, torch.optim.AdamW(model.parameters(), lr=.01)

    x = torch.arange(24, dtype=torch.float32, device=device).reshape(8, 3) / 24
    model, optimizer = build()

    def step(model, optimizer):
        order = rng.permutation(8)
        optimizer.zero_grad(set_to_none=True)
        loss = model(x[order]).square().mean()
        loss.backward()
        optimizer.step()
        return order, loss.detach().clone()

    step(model, optimizer)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'last.pt'
        progress = {'epoch': 1, 'history': [{'epoch': 1}], 'best': (.4, .5)}
        training.save_training_state(path, model, optimizer, rng, progress)
        order, loss = step(model, optimizer)
        expected = {k: v.clone() for k, v in model.state_dict().items()}
        restored, restored_optimizer = build()
        saved = training.restore_training_state(path, restored, restored_optimizer, rng)
        assert saved['epoch'] == 1 and saved['history'] == progress['history']
        assert tuple(saved['best']) == progress['best']
        resumed_order, resumed_loss = step(restored, restored_optimizer)
        assert np.array_equal(order, resumed_order), '恢复后的样本顺序发生改变'
        torch.testing.assert_close(loss, resumed_loss, rtol=0, atol=0)
        for name, actual in restored.state_dict().items():
            torch.testing.assert_close(actual, expected[name], rtol=0, atol=0)
        assert not path.with_suffix('.pt.tmp').exists()
    print(f'PASS: {device} next-step sampling, RNG and AdamW match uninterrupted training')


if __name__ == '__main__':
    assert hasattr(training, 'save_training_state'), '训练入口尚不能保存完整续训状态'
    check('cpu')
    if torch.cuda.is_available():
        check('cuda')
