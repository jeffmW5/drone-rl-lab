"""Build or simulate with the downloaded official SDK, pinned by image digest.
Does not pull, flash, attach USB, or launch physical flight.
Run via `wsl -d Ubuntu-Laya -u root -- .../bin/python sdk_build.py camera|golden`.
"""
import argparse
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('stage', choices=['camera', 'golden'])
args = p.parse_args()
root = Path(__file__).resolve().parent
pin = root / 'sdk-image.json'
image = json.loads(pin.read_text())['digest'] if pin.exists() else 'bitcraze/aideck:latest'
inspection = subprocess.run(['docker', 'image', 'inspect', image], capture_output=True, text=True, check=True)
info = json.loads(inspection.stdout)[0]
digests = info.get('RepoDigests') or []
if not digests:
    raise RuntimeError('SDK image lacks a registry digest')
digest = next(d for d in digests if d.startswith('bitcraze/aideck@'))
if not pin.exists():
    pin.write_text(json.dumps({'digest': digest, 'image_id': info['Id'], 'created': info['Created']}, indent=2))
stage = 'tether-camera' if args.stage == 'camera' else 'tether-golden'
build_dir = (root / 'firmware' / stage).resolve(strict=True)
command = 'set -e; source /gap_sdk/configs/ai_deck.sh; cd /build'
command += '; make clean; make image CORE=8' + ('' if args.stage == 'camera' else ' platform=gvsoc io=host; make run platform=gvsoc CORE=8 io=host')
log = root / f'gap8-{args.stage}-build.log'
with log.open('w') as output:
    result = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', '/bin/bash',
                             '-v', f'{build_dir}:/build', digest, '-lc', command], stdout=output, stderr=subprocess.STDOUT)
text = log.read_text(errors='replace')
passed = result.returncode == 0
if args.stage == 'golden':
    passed = passed and 'GOLDEN_PASS mismatched_bytes=0' in text
report = {'stage': args.stage, 'image': digest, 'exit_code': result.returncode, 'passed': passed,
          'log': str(log), 'physical_hardware_tested': False}
(root / f'gap8-{args.stage}-result.json').write_text(json.dumps(report, indent=2))
print(text[-6000:])
print(json.dumps(report, indent=2))
raise SystemExit(0 if passed else 1)
