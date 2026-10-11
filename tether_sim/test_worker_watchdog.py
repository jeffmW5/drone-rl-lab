import multiprocessing
import os
import time
import pytest
from .worker_watchdog import WorkerWatchdog


def test_dead_worker_interrupts_waiting_parent():
    process = multiprocessing.get_context('fork').Process(target=os._exit, args=(1,))
    process.start()
    process.join(timeout=2)
    watchdog = WorkerWatchdog([process], interval=.02)
    try:
        with pytest.raises(ChildProcessError, match='exited unexpectedly'):
            watchdog.start()
            time.sleep(2)
    finally:
        watchdog.close()


def test_failed_campaign_does_not_skip_to_next_queued_trial(tmp_path, monkeypatch):
    from . import research_loop
    monkeypatch.setattr(research_loop, 'snapshot', lambda root: {'processes': []})
    state = dict(trials=[dict(status='failed'), dict(status='queued')], active_pids=[1])
    result = research_loop.tick(tmp_path, {}, state)
    assert 'failed trial' in result['status']
    assert result['active_pids'] == []
