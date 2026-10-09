"""Record concrete artifact hashes and source revisions, without marking completion."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
def run(*args):
    return subprocess.check_output(args, text=True).strip()

revisions = {}
for directory in (root / 'vendor').iterdir():
    if (directory / '.git').exists():
        revisions[directory.name] = {'commit': run('git', '-C', str(directory), 'rev-parse', 'HEAD'),
                                    'dirty': run('git', '-C', str(directory), 'status', '--porcelain')}
paths = ['runs/smoke/model.pt', 'runs/smoke/model.float.onnx', 'runs/quantized/model_int.onnx',
         'firmware/stm32/build/cf2.bin', 'firmware/stm32-cf21bl/build/cf21bl.bin',
         'firmware/tether-camera/BUILD/GAP8_V2/GCC_RISCV_FREERTOS/tether_camera',
         'firmware/tether-golden/BUILD/GAP8_V2/GCC_RISCV_FREERTOS/main']
artifacts = {}
for relative in paths:
    path = root / relative
    if path.is_file():
        artifacts[relative] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
sdk_results = {}
for stage in ('camera', 'golden'):
    result = root / f'gap8-{stage}-result.json'
    if result.exists():
        sdk_results[stage] = json.loads(result.read_text())
report = {'sources': revisions, 'artifacts': artifacts,
          'quantization': json.loads((root/'runs/quantized/quantization.json').read_text()),
          'arm_compiler': run('arm-none-eabi-gcc', '--version').splitlines()[0],
          'sdk_results': sdk_results,
          'gap8_compilation_verified': sdk_results.get('camera', {}).get('passed', False),
          'gap8_golden_input_verified': sdk_results.get('golden', {}).get('passed', False),
          'physical_hardware_verified': False,
          'omarchy_configured': False, 'goal_complete': False}
(root/'evidence-current.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
