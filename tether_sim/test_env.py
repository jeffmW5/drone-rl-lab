import numpy as np
import pytest
import mujoco
from gymnasium.utils.env_checker import check_env
from .env import TetherEnv, Config


def test_gym_contract():
    env = TetherEnv()
    check_env(env, skip_render_check=True)
    with pytest.raises(ValueError):
        env.step([np.nan]*4)
    env.close()


def test_freefall():
    env = TetherEnv()
    for _ in range(5):
        env.step(np.zeros(4))
    assert env.data.qpos[2] < 1.17
    assert env.data.qvel[2] < -.8


@pytest.mark.parametrize('scenario',['hover','swing','lift','carry','lift_carry'])
def test_baseline_and_cable(scenario):
    env = TetherEnv()
    env.reset(seed=7,options={'scenario':scenario})
    for _ in range(600):
        obs,_,done,_,info = env.step(env.baseline())
        assert np.isfinite(obs).all()
        assert not done
        # Rigid links cannot extend; drone attachment offset adds 2 cm.
        assert info['endpoint_distance'] <= env.config.length+.020001
    assert np.linalg.norm(env.data.qpos[:3]-env.target) < .08


def test_replay():
    a,b=TetherEnv(),TetherEnv()
    for _ in range(100):
        oa,*_=a.step(a.baseline())
        ob,*_=b.step(b.baseline())
    np.testing.assert_array_equal(oa,ob)


def test_timestep_convergence():
    a,b=TetherEnv(Config(timestep=.002)),TetherEnv(Config(timestep=.001))
    for e in [a,b]:
        e.reset(options={'scenario':'swing'})
        for _ in range(200):
            e.step(e.baseline())
    assert np.linalg.norm(a.data.qpos[:3]-b.data.qpos[:3]) < .01


def test_mass_budget():
    e=TetherEnv()
    assert sum(e.model.body_mass) == pytest.approx(.0494)


def test_static_cable_load():
    e=TetherEnv()
    for _ in range(600):
        e.step(e.baseline())
    assert e.metrics()['attachment_axial_load_N'] == pytest.approx((e.config.payload_mass+e.config.cable_mass)*9.81,rel=.002)


def test_ground_support_and_folding():
    e=TetherEnv()
    e.reset(options={'scenario':'lift'})
    e.data.qpos[7:11] = [np.cos(.05),0,np.sin(.05),0]
    mujoco.mj_forward(e.model,e.data)
    # With motors off the chain must fold onto the ground, not remain taut.
    for _ in range(80):
        e.step(np.zeros(4))
    info=e.metrics()
    assert info['contacts'] > 0
    assert info['endpoint_distance'] < e.config.length*.9
