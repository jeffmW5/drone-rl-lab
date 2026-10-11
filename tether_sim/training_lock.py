"""Process-held local GPU lease; the OS releases it after process exit."""
import fcntl
from pathlib import Path


def acquire_training_lease(path=Path('tmp/tether-training.lock')):
    path.parent.mkdir(parents=True, exist_ok=True)
    lease = path.open('a')
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        raise RuntimeError('A local training process already owns the GPU lease') from None
    return lease
