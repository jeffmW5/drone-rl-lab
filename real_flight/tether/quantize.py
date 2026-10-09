"""Experimental power-of-two PTQ exporter for this exact TetherNet architecture.

Emits NEMO-pattern ONNX understood by DORY, without claiming to implement NEMO.
All stored graph weights/biases are integer-valued float32; inference arithmetic
is uint8/int8/int32. Shift-only requantization avoids multiplier overflow.
Must pass generated-code golden tests before hardware deployment.
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as H, numpy_helper as NH, TensorProto as T
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import TetherNet
from train import SyntheticLines
from preprocessing import CameraTensor


def pow2_scale(maximum, limit):
    return 2.0 ** math.ceil(math.log2(max(float(maximum), 1e-12) / limit))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--data', type=Path, help='Session-split data root containing train/val folders')
    p.add_argument('--output', type=Path, default=Path('runs/quantized'))
    args = p.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = TetherNet().eval()
    model.load_state_dict(checkpoint['state_dict'])
    if args.data:
        tfm = CameraTensor()
        calibration = datasets.ImageFolder(args.data / 'train', tfm)
        validation = datasets.ImageFolder(args.data / 'val', tfm)
        if calibration.classes != checkpoint['classes'] or validation.classes != checkpoint['classes']:
            raise ValueError('Class ordering differs from checkpoint')
    elif checkpoint['synthetic']:
        calibration = SyntheticLines(512, 200000)
        validation = SyntheticLines(512, 300000)
    else:
        p.error('Real checkpoint requires --data for calibration/validation')
    maxima = {}
    with torch.no_grad():
        for x, _ in DataLoader(calibration, batch_size=64):
            for i, layer in enumerate(model.layers):
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    maxima[i] = max(maxima.get(i, 0), x.max().item())

    nodes, initializers, specifications = [], [], []
    current = '0'; serial = 0; scale = 1 / 256
    def constant(name, values):
        initializers.append(NH.from_array(np.asarray(values, dtype=np.float32), name))
        return name
    def node(op, inputs, **attrs):
        nonlocal current, serial
        serial += 1
        current = str(serial)
        nodes.append(H.make_node(op, inputs, [current], name=f'{op}_{serial}', **attrs))
        return current

    for i, layer in enumerate(model.layers):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            ws = pow2_scale(layer.weight.detach().abs().max(), 127)
            w = torch.round(layer.weight.detach() / ws).clamp(-127, 127).to(torch.int64)
            b = torch.round(layer.bias.detach() / (scale * ws)).to(torch.int64)
            if b.abs().max() >= 2**31:
                raise ValueError('Bias exceeds int32')
            weight = constant(f'weight_{i}', w.numpy())
            bias = constant(f'bias_{i}', b.numpy())
            if isinstance(layer, nn.Conv2d):
                node('Conv', [current, weight, bias], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1], dilations=[1, 1], group=1)
            else:
                node('Gemm', [current, weight, bias], transB=1)
            bound = int(w.abs().reshape(w.shape[0], -1).sum(1).max()) * 255 + int(b.abs().max())
            if bound >= 2**24:
                raise ValueError('Accumulator exceeds exact float32 golden-graph integer range')
            specifications.append({'op': 'conv' if isinstance(layer, nn.Conv2d) else 'linear',
                                   'index': i, 'weight_scale': ws, 'input_scale': scale,
                                   'accumulator_bound': bound})
            scale *= ws
        elif isinstance(layer, nn.ReLU):
            target = pow2_scale(maxima[i], 255)
            shift = round(math.log2(target / scale))
            if not 0 <= shift <= 31:
                raise ValueError('Unsupported requantization shift')
            node('Mul', [current, constant(f'multiplier_{i}', [1])])
            node('Div', [current, constant(f'divisor_{i}', [2**shift])])
            node('Floor', [current])
            node('Clip', [current], min=0.0, max=255.0)
            specifications.append({'op': 'relu', 'index': i, 'shift': shift, 'scale': target})
            scale = target
        elif isinstance(layer, nn.MaxPool2d):
            node('MaxPool', [current], kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0])
            specifications.append({'op': 'maxpool', 'index': i})
        elif isinstance(layer, nn.AvgPool2d):
            node('AveragePool', [current], kernel_shape=[4, 4], strides=[4, 4], pads=[0, 0, 0, 0])
            node('Floor', [current])
            specifications.append({'op': 'avgpool', 'index': i})
        elif isinstance(layer, nn.Flatten):
            node('Flatten', [current], axis=1)
        else:
            raise ValueError(f'Unsupported layer {layer}')

    args.output.mkdir(parents=True, exist_ok=True)
    graph = H.make_graph(nodes, 'tether_integer', [H.make_tensor_value_info('0', T.FLOAT, [1, 1, 64, 64])],
                         [H.make_tensor_value_info(current, T.FLOAT, [1, 4])], initializers)
    exported = H.make_model(graph, opset_imports=[H.make_opsetid('', 10)], ir_version=7)
    exported = onnx.shape_inference.infer_shapes(exported)
    onnx.checker.check_model(exported)
    onnx.save(exported, args.output / 'model_int.onnx')
    session = ort.InferenceSession(str(args.output / 'model_int.onnx'), providers=['CPUExecutionProvider'])
    constants = {tensor.name: NH.to_array(tensor) for tensor in initializers}
    header = ['#pragma once', '#include <stdint.h>']
    for spec in specifications:
        i = spec['index']
        if spec['op'] in ('conv', 'linear'):
            for prefix, ctype in [('weight', 'int8_t'), ('bias', 'int32_t')]:
                values = constants[f'{prefix}_{i}'].flatten().astype(np.int64)
                header.append(f'static const {ctype} {prefix}_{i}[{len(values)}] = {{' + ','.join(map(str, values)) + '};')
        elif spec['op'] == 'relu':
            header.append(f'static const unsigned shift_{i} = {spec["shift"]};')
    (args.output / 'integer_model.h').write_text('\n'.join(header) + '\n')
    binary = (args.output / 'host-reference').resolve()
    subprocess.run(['gcc', '-std=c99', '-O2', '-Wall', '-Wextra', '-Werror', '-I', str(args.output),
                    str(Path(__file__).with_name('host-reference.c')), '-o', str(binary)], check=True)

    def integer_reference(raw):
        x = torch.from_numpy(raw.astype(np.float64))
        outputs = []
        for spec in specifications:
            i = spec['index']; op = spec['op']
            if op in ('conv', 'linear'):
                w = torch.from_numpy(constants[f'weight_{i}'].astype(np.float64))
                b = torch.from_numpy(constants[f'bias_{i}'].astype(np.float64))
                x = torch.nn.functional.conv2d(x, w, b, padding=1) if op == 'conv' else torch.nn.functional.linear(x.flatten(1), w, b)
            elif op == 'relu':
                x = torch.floor(x / 2**spec['shift']).clamp(0, 255)
                outputs.append(x.clone())
            elif op == 'maxpool':
                x = torch.nn.functional.max_pool2d(x, 2); outputs.append(x.clone())
            elif op == 'avgpool':
                x = torch.floor(torch.nn.functional.avg_pool2d(x, 4)); outputs.append(x.clone())
        outputs.append(x.clone())
        return x.numpy(), outputs

    correct = float_correct = disagreements = 0; max_error = 0
    host_inputs, host_expected = [], []
    for index in range(min(len(validation), 512)):
        x, label = validation[index]
        raw = torch.round(x * 255).numpy()[None].astype(np.float32)
        reference, layer_outputs = integer_reference(raw)
        actual = session.run(None, {'0': raw})[0]
        np.testing.assert_array_equal(reference.astype(np.float32), actual)
        host_inputs.append(raw.astype(np.uint8).tobytes())
        host_expected.append(reference.astype(np.int32))
        prediction = int(actual.argmax())
        with torch.no_grad():
            float_prediction = int(model(x[None]).argmax())
        correct += prediction == label; float_correct += float_prediction == label
        disagreements += prediction != float_prediction
        max_error = max(max_error, float(np.abs(reference - actual).max()))
        if index == 0:
            np.savetxt(args.output / 'input.txt', raw.flatten().astype(np.uint8), fmt='%d', delimiter=',')
            for number, output in enumerate(layer_outputs):
                values = output.numpy()
                if values.ndim == 4:
                    values = values.transpose(0, 2, 3, 1)
                np.savetxt(args.output / f'out_layer{number}.txt', values.flatten().astype(np.int64), fmt='%d', delimiter=',')
            np.savez(args.output / 'golden.npz', input=raw, output=actual)
    count = min(len(validation), 512)
    host = subprocess.run([str(binary)], input=b''.join(host_inputs), capture_output=True, check=True)
    host_actual = np.frombuffer(host.stdout, dtype='<i4').reshape(count, 4)
    np.testing.assert_array_equal(host_actual, np.concatenate(host_expected))
    report = {'samples': count, 'synthetic': checkpoint['synthetic'], 'float_accuracy': float_correct/count,
              'integer_accuracy': correct/count, 'prediction_disagreements': disagreements,
              'onnx_integer_reference_max_error': max_error, 'output_scale': scale,
              'classes': checkpoint['classes'], 'layers': specifications,
              'portable_c_integer_match': True,
              'generated_c_validated': False, 'hardware_validated': False}
    (args.output / 'quantization.json').write_text(json.dumps(report, indent=2))
    (args.output / 'config.json').write_text(json.dumps({'BNRelu_bits': 32, 'onnx_file': 'model_int.onnx', 'code reserved space': 132000}, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
