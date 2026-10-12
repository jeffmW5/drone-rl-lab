"""Opt-in CUDA particle-tether prototype. Not a compatible training environment."""
import ctypes
import hashlib
from pathlib import Path
import torch


class CudaTether:
    def __init__(self, count=4096, library='tmp/cuda-tether-v001/libtether.so'):
        if count < 1 or not torch.cuda.is_available():
            raise ValueError('Positive batch size and CUDA required')
        for line in Path(library).with_name('sources.sha256').read_text().splitlines():
            digest,source=line.split(maxsplit=1)
            if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('CUDA sources changed; rebuild before running')
        self.count = count
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        torch.cuda.init()
        torch.empty(0,device='cuda')  # Establish the primary context for Driver API.
        self.lib.tether_init.argtypes=[ctypes.c_char_p]
        self.lib.tether_init.restype=ctypes.c_int
        code=self.lib.tether_init(str(Path(library).with_name('physics.cubin').resolve()).encode())
        if code: raise RuntimeError('CUDA module initialization error '+str(code))
        self.lib.tether_step.argtypes = [ctypes.c_void_p]*10 + [ctypes.c_int]*2 + [ctypes.c_float] + [ctypes.c_int]*2 + [ctypes.c_void_p]
        self.lib.tether_step.restype = ctypes.c_int
        self.position = torch.zeros(count,14,3,device='cuda',dtype=torch.float64)
        self.velocity = torch.zeros_like(self.position)
        self.orientation = torch.zeros(count,7,device='cuda')
        self.motors = torch.zeros(count,4,device='cuda')
        self.mass = torch.tensor([.0434]+[.001/12]*12+[.005],device='cuda').repeat(count,1)
        self.params = torch.tensor([.6,.025,.2,.015,.000025,.000025,.00004,0,0],device='cuda').repeat(count,1)
        self.wind = torch.zeros(count,3,device='cuda')
        self.boxes = torch.empty(count,0,6,device='cuda')
        self.failed = torch.zeros(count,dtype=torch.int32,device='cuda')
        self.reset()

    def reset(self):
        self.position.zero_(); self.position[:,:,2] = 2-torch.arange(14,device='cuda')*.6/13
        self.velocity.zero_(); self.orientation.zero_(); self.orientation[:,0]=1
        self.motors[:] = self.mass.sum(1,keepdim=True)*9.81/4
        self.failed.zero_()

    def step(self, actions, dt=.001, substeps=20, iterations=32):
        if dt<=0 or not 1<=substeps<=100 or not 1<=iterations<=256:
            raise ValueError('Invalid integration budget')
        tensors=[self.position,self.velocity,self.orientation,self.motors,actions,
                 self.mass,self.params,self.wind,self.boxes,self.failed]
        shapes=[(self.count,14,3),(self.count,14,3),(self.count,7),(self.count,4),
                (self.count,4),(self.count,14),(self.count,9),(self.count,3),
                (self.count,self.boxes.shape[1],6),(self.count,)]
        for index,(tensor,shape) in enumerate(zip(tensors,shapes)):
            if tensor.device!=self.position.device or not tensor.is_cuda or not tensor.is_contiguous() or tuple(tensor.shape)!=shape or tensor.dtype!=(torch.int32 if index==9 else torch.float64 if index<2 else torch.float32):
                raise ValueError('Wrong tensor shape, dtype, device or layout')
        with torch.cuda.device(self.position.device):
            stream=torch.cuda.current_stream()
            code=self.lib.tether_step(*[t.data_ptr() for t in tensors],self.count,self.boxes.shape[1],dt,substeps,iterations,stream.cuda_stream)
            if code:
                raise RuntimeError('CUDA launch error '+str(code))
        return self.position,self.failed
