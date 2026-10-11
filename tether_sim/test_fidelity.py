import mujoco
import numpy as np
import pytest
from .env import Config,TetherEnv
from .navigation_v2 import NavigationTaskV2
from .planning import plan_route


def test_rotor_collision_beyond_body_box():
    obstacle=(.05,.032,1.206,.002,.002,.002,0.)
    old=TetherEnv(Config(obstacles=(obstacle,)))
    new=TetherEnv(Config(obstacles=(obstacle,),rotor_collisions=True,deck_collisions=True))
    assert old.data.ncon==0
    rotor=new.model.geom('rotor_collision0').id
    assert any(rotor in (c.geom1,c.geom2) for c in new.data.contact[:new.data.ncon])
    assert new.model.body_mass.sum()==pytest.approx(old.model.body_mass.sum())
    old.close();new.close()


def test_full_cable_capsule_clearance_detects_mid_link_contact():
    task=NavigationTaskV2();task.reset(seed=71001)
    task.env.close()
    task.env=TetherEnv(Config(obstacles=((.0015,0,1.155,.001,.001,.001,0.),),rotor_collisions=True))
    task.obstacle_ids=task.env.obstacle_geom_ids
    assert task.clearance()<0
    task.close()


def test_obstacle_contact_latched_between_control_samples(monkeypatch):
    e=TetherEnv(Config(obstacles=((1,0,1.2,.05,.05,.05,0.),),substep_collision_checks=True))
    original=mujoco.mj_step
    calls=0
    def transient(model,data):
        nonlocal calls
        calls+=1
        data.qpos[:3]=[1,0,1.2] if calls==2 else [0,0,1.2]
        mujoco.mj_forward(model,data)
        original(model,data)
    monkeypatch.setattr(mujoco,'mj_step',transient)
    e.step(np.zeros(4))
    assert e.data.ncon==0
    assert e.obstacle_collision_in_step
    e.close()


def test_nonintegral_control_clock_rejected():
    with pytest.raises(ValueError,match='integer multiple'):
        TetherEnv(Config(timestep=.003))


def test_v2_seed_and_transfer_observation_contract():
    e=NavigationTaskV2();a,ia=e.reset(seed=71001);b,ib=e.reset(seed=71001)
    np.testing.assert_array_equal(a,b)
    assert e.observation_space.contains(b)
    assert e.domain['planner_margin_m']==.26
    assert e.domain['hardware']=='Crazyflie 2.1 + Flow deck + AI deck'
    assert e.env.data.qpos[:2].any()
    with pytest.raises(ValueError):e.set_goal([3.,0.])
    e.close()


def test_v2_destination_is_not_a_moving_target_success():
    e=NavigationTaskV2();_,info=e.reset(seed=71002)
    assert not info['goal_reached']
    assert info['goal_distance_m']>1
    assert info['route_remaining_m']>1
    e.set_goal([-.5,.5])
    assert e.domain['goal_xy']==[-.5,.5]
    assert e.progress==0
    assert np.linalg.norm(e.env.target[:2]-e.env.data.qpos[:2])<1e-8
    e.close()
