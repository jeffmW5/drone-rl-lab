"""Export FLOAT ONNX for inspection; NOT yet DORY quantized input."""
import argparse
from pathlib import Path
import json
import numpy as np
import onnx
import onnxruntime as ort
import torch
from model import TetherNet

p = argparse.ArgumentParser()
p.add_argument('checkpoint', type=Path)
args = p.parse_args()
checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
model = TetherNet().eval()
model.load_state_dict(checkpoint['state_dict'])
x = torch.rand(1, 1, 64, 64)
target = args.checkpoint.with_suffix('.float.onnx')
torch.onnx.export(model, x, target, input_names=['gray'], output_names=['logits'], opset_version=13, dynamo=False)
onnx.checker.check_model(onnx.load(target))
session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
with torch.no_grad():
    ref = model(x).numpy()
actual = session.run(None, {'gray': x.numpy()})[0]
np.testing.assert_allclose(ref, actual, atol=1e-5, rtol=1e-4)
report = {'onnx': str(target), 'max_abs_error': float(np.abs(ref-actual).max()),
          'classes': checkpoint['classes'], 'dory_ready': False,
          'next': 'NEMO/Quantlab integer quantization, DORY graph generation and integer golden-vector comparison'}
target.with_suffix('.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
