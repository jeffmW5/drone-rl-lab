"""GPU smoke training or training from session-split camera data.

Real data layout: data/train/{absent,left,center,right}/*.png and
data/val/{absent,left,center,right}/*.png. Split whole recording sessions,
not neighboring frames. Synthetic validation proves pipeline operation only.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from model import TetherNet
from preprocessing import CameraTensor

CLASSES = ['absent', 'left', 'center', 'right']


class SyntheticLines(Dataset):
    def __init__(self, count, seed):
        self.count, self.seed = count, seed

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i)
        label = int(rng.integers(4))
        background = np.clip(rng.normal(rng.uniform(90, 170), 14, (64, 64)), 0, 255).astype('uint8')
        frame = Image.fromarray(background)
        draw = ImageDraw.Draw(frame)
        for _ in range(3):
            x, y = rng.integers(0, 56, 2).tolist()
            draw.rectangle((x, y, x + 5, y + 5), fill=int(rng.integers(70, 190)))
        if label:
            x = int(rng.integers(*{1: (8, 21), 2: (26, 39), 3: (44, 57)}[label]))
            dx = int(rng.integers(-5, 6))
            draw.line((x - dx, 0, x + dx, 63), fill=int(rng.choice([10, 240])), width=int(rng.integers(1, 4)))
        return torch.from_numpy(np.asarray(frame).copy()).float().unsqueeze(0) / 255, label


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path)
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--output', type=Path, default=Path('runs/smoke'))
    args = p.parse_args()
    if bool(args.data) == args.synthetic:
        p.error('Choose exactly one of --data or --synthetic')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing a silent CPU fallback')
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device('cuda')
    if args.synthetic:
        train, val = SyntheticLines(2048, 1000), SyntheticLines(512, 100000)
    else:
        transform = CameraTensor()
        train = datasets.ImageFolder(args.data / 'train', transform)
        val = datasets.ImageFolder(args.data / 'val', transform)
        if train.class_to_idx != val.class_to_idx or set(train.classes) != set(CLASSES):
            raise ValueError('Both splits must contain absent/left/center/right folders')
    classes = CLASSES if args.synthetic else train.classes
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    net = TetherNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.003)
    history = []
    for epoch in range(args.epochs):
        net.train(); total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = torch.nn.functional.cross_entropy(net(x), y)
            loss.backward(); opt.step(); total_loss += loss.item() * len(y)
        net.eval(); correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                correct += (net(x.to(device)).argmax(1).cpu() == y).sum().item()
        row = {'epoch': epoch + 1, 'loss': total_loss / len(train), 'val_accuracy': correct / len(val)}
        history.append(row); print(json.dumps(row), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': net.cpu().state_dict(), 'classes': classes, 'synthetic': args.synthetic}, args.output / 'model.pt')
    report = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'capability': torch.cuda.get_device_capability(0),
              'architectures': torch.cuda.get_arch_list(), 'synthetic': args.synthetic,
              'parameters': sum(v.numel() for v in net.parameters()), 'history': history,
              'flight_ready': False}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
