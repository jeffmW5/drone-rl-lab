import mujoco
import numpy as np
from .generalize import GeneralLiftCarryTask


def test_domain_seed_and_observation():
    e=GeneralLiftCarryTask()
    a,ia=e.reset(seed=20001)
    b,ib=e.reset(seed=20001)
    np.testing.assert_array_equal(a,b)
    assert ia['domain']==ib['domain']
    c,ic=e.reset(seed=20002)
    assert not np.array_equal(b,c)
    assert e.observation_space.contains(c)
    assert e.env.config.payload_mass!=ia['domain']['payload_mass']


def test_real_obstacle_collision():
    e=GeneralLiftCarryTask()
    e.reset(seed=20001)
    o=e.env.config.obstacles[0]
    e.env.data.qpos[:3]=o[:3]
    mujoco.mj_forward(e.env.model,e.env.data)
    assert e.clearance()<0
    _,_,done,_,info=e.step(np.zeros(4))
    assert done and info['obstacle_collision']


def test_zero_residual_route_sweep():
    for seed in (20001,20002,20003,20004,20005):
        e=GeneralLiftCarryTask()
        e.reset(seed=seed)
        for _ in range(800):
            obs,reward,done,_,info=e.step(np.zeros(4))
            assert np.isfinite(obs).all() and np.isfinite(reward)
            assert not done,(seed,info)
        # A PD controller without integral compensation can retain gain-bias error.
        assert info['payload_error_m']<.3
        e.close()
