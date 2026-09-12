"""预取仅改变CPU读取时机，保留窗口、顺序、尾batch和参数更新。"""
import copy
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import train_blurball_midpoint as training


def check():
    assert hasattr(training, 'prefetched_batches'), '尚无CPU单批预取'
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'rgb.npy'
        np.save(path, np.arange(8*3*2*2, dtype=np.uint8).reshape(8, 3, 2, 2))
        rgb = np.load(path, mmap_mode='r')
        windows = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6]])
        order = np.array([2, 0, 1])
        torch.manual_seed(4)
        reference = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(36, 2))
        prefetched = copy.deepcopy(reference)
        cpu_rng = torch.get_rng_state().clone()
        actual = list(training.prefetched_batches(rgb, windows, order, 2))
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert np.array_equal(np.concatenate([ids for ids, _ in actual]), order)
        assert [len(ids) for ids, _ in actual] == [2, 1]
        for ids, pixels in actual:
            assert pixels.dtype == torch.uint8 and pixels.device.type == 'cpu'
            expected = training.model_input(rgb, windows, ids, 'cpu', 'dino')
            torch.testing.assert_close(pixels, expected, rtol=0, atol=0)
        for model, batches in ((reference, [(order[i:i+2], training.model_input(
                rgb, windows, order[i:i+2], 'cpu', 'dino')) for i in range(0, len(order), 2)]),
                               (prefetched, actual)):
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            for ids, pixels in batches:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(pixels.float()/255),
                                                          torch.from_numpy(ids % 2))
                loss.backward()
                optimizer.step()
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, prefetched.state_dict()[name], rtol=0, atol=0)
    print('PASS: exact RGB/order/tail batch, unchanged RNG and AdamW updates')


if __name__ == '__main__':
    check()
