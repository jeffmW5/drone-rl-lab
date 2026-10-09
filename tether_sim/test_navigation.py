import numpy as np
import pytest
from .planning import plan_route
from .navigation import NavigationTask


def test_planner_detours_and_rejects_blocked_goal():
    boxes=[(.6,0,.75,.15,.15,.75,0)]
    path=plan_route([0,0],[1.2,0],boxes)
    assert np.max(np.abs(path[:,1]))>.3
    with pytest.raises(ValueError):
        plan_route([0,0],[.6,0],boxes)
    with pytest.raises(ValueError):
        plan_route([0,0],[1.2,0],[(.6,0,.75,.15,3,.75,0)])


def test_random_navigation_seed_and_collision_free_baseline():
    for seed in (21001,21002,21003):
        e=NavigationTask()
        a,info=e.reset(seed=seed)
        b,_=e.reset(seed=seed)
        np.testing.assert_array_equal(a,b)
        assert len(e.env.config.obstacles)==6
        assert e.observation_space.contains(b)
        assert len(e.route)>2
        for _ in range(1500):
            obs,reward,done,_,info=e.step(np.zeros(4))
            assert np.isfinite(obs).all() and np.isfinite(reward)
            assert not done,(seed,info)
        assert np.linalg.norm(e.env.data.qpos[:2]-e.goal)<.3
        e.close()
