import numpy as np
from gymnasium.utils.env_checker import check_env
from .task import LiftCarryTask


def test_task_contract_and_seed():
    e=LiftCarryTask()
    check_env(e,skip_render_check=True)
    a,_=e.reset(seed=42)
    b,_=e.reset(seed=42)
    np.testing.assert_array_equal(a,b)


def test_zero_residual_is_pd():
    e=LiftCarryTask()
    e.reset(seed=123)
    for _ in range(600):
        _,reward,done,_,info=e.step(np.zeros(4))
        assert np.isfinite(reward) and not done
    assert info['payload_error_m']<.1
