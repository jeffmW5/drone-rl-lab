"""Reproduce this user's isolated PufferLib 3 CUDA runtime under WSL.

Run with the existing crazyflie-tether-venv Python. No system Python packages or
existing training environment are modified. Build only while trainers are stopped.
"""
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path


def download(url, path, sha):
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise RuntimeError('Download checksum mismatch: '+str(path))


def main():
    if os.name != 'posix':
        raise RuntimeError('Run this installer inside Ubuntu-Laya')
    root = Path('/home/n33du/ai')
    original = root/'crazyflie-tether-venv'
    runtime = root/'puffer-tether-venv'
    work = root/'puffer-tether-work'
    toolkit = root/'cuda-12.6-puffer'
    work.mkdir(exist_ok=True)
    if not runtime.exists():
        subprocess.check_call([str(original/'bin/python'), '-m', 'venv', str(runtime)])
    site = runtime/'lib/python3.12/site-packages'
    (site/'existing-training-runtime.pth').write_text(str(original/'lib/python3.12/site-packages')+'\n')
    python = str(runtime/'bin/python')
    # Optional Ocean games/cloud logging are unnecessary for the tether task.
    deps = ['rich==15.0.0','rich-argparse==1.8.0','pyro-ppl==1.9.2','pyro-api==0.1.2',
            'psutil==7.2.2','nvidia-ml-py==13.615.71','gym==0.26.2','gym_notices==0.1.0',
            'pettingzoo==1.24.3','shimmy==1.3.0','ninja==1.13.2','opt_einsum==3.4.0',
            'tqdm==4.70.1','markdown-it-py==4.2.0','mdurl==0.1.2','pygments==2.21.0',
            'pytest==9.1.1','pluggy==1.6.0','iniconfig==2.3.1']
    subprocess.check_call([python,'-m','pip','install','--no-deps',*deps])
    metadata = json.load(urllib.request.urlopen('https://pypi.org/pypi/pufferlib/3.0.0/json'))
    entry = next(e for e in metadata['urls'] if e['packagetype']=='sdist')
    archive = work/'pufferlib-3.0.0.tar.gz'
    download(entry['url'], archive, entry['digests']['sha256'])
    source = work/'pufferlib-3.0.0'
    if not source.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(work, filter='data')
    setup = source/'setup.py'
    text = setup.read_text()
    # Upstream NO_OCEAN=1 packaging bug: an empty list is referenced before assignment.
    text = text.replace('c_extensions = []\nif not NO_OCEAN:',
                        'c_extensions = []\nc_extension_paths = []\nif not NO_OCEAN:')
    setup.write_text(text)
    base = 'https://developer.download.nvidia.com/compute/cuda/redist/'
    manifest = json.load(urllib.request.urlopen(base+'redistrib_12.6.3.json'))
    cuda_work = work/'cuda'
    cuda_work.mkdir(exist_ok=True)
    toolkit.mkdir(exist_ok=True)
    for name in ('cuda_nvcc','cuda_cudart','cuda_cccl'):
        component = manifest[name]['linux-x86_64']
        archive = cuda_work/Path(component['relative_path']).name
        download(base+component['relative_path'], archive, component['sha256'])
        directory = cuda_work/archive.name.removesuffix('.tar.xz')
        if not directory.exists():
            with tarfile.open(archive) as tar:
                tar.extractall(cuda_work, filter='data')
        shutil.copytree(directory, toolkit, dirs_exist_ok=True, symlinks=True)
    environment = dict(os.environ, NO_OCEAN='1', MAX_JOBS='1', CUDA_HOME=str(toolkit),
                       TORCH_CUDA_ARCH_LIST='6.1', PATH=str(toolkit/'bin')+':'+os.environ['PATH'])
    subprocess.check_call([python,'-m','pip','install','--force-reinstall','--no-deps',
                           '--no-build-isolation',str(source)], env=environment)
    print('PufferLib 3 CUDA runtime ready; run adapter tests before training.')


if __name__ == '__main__':
    main()
