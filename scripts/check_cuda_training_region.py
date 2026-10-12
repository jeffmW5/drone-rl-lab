"""Check training-only truncation and unchanged baseline/evaluation behavior."""
import ast
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import torch
from tether_sim.cuda_task import CudaVector
from tether_sim.training_lock import acquire_training_lease

output = Path(sys.argv[1])
assert not output.exists(), 'Preserve check evidence'
lease = acquire_training_lease()
try:
    outcomes = {}
    for name, region in [('baseline', None), ('bounded', {'radius_m': 1., 'ceiling_m': 1.6})]:
        vector = CudaVector(2, direct=True, reward_version=2, phase='takeoff', control_version=2, training_region=region)
        vector.async_reset(51011)
        vector.sim.boxes.zero_()
        # One domain escapes horizontally; the other escapes vertically.
        vector.sim.position[0, :, 0] += 4
        vector.sim.position[1, :, 2] += 2
        vector.send(torch.zeros(2, 4, device='cuda'))
        truncation = vector.truncated.cpu().tolist()
        assert truncation == ([True, True] if region else [False, False])
        if region:
            assert torch.equal(vector.sim.position, vector.start)
            assert not bool(vector.sim.failed.any())
            assert not bool(vector.sim.motors.any())
        outcomes[name] = {'truncated': truncation, 'physics_failed': vector.sim.failed.cpu().tolist()}
    # Evaluation must never receive the training-region override.
    source = Path('scripts/train_tether_cuda.py').read_text()
    evaluation = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'evaluate')
    assert not any(isinstance(node, ast.keyword) and node.arg == 'training_region' for node in ast.walk(evaluation))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(outcomes, indent=2))
    print(json.dumps(outcomes))
finally:
    lease.close()
