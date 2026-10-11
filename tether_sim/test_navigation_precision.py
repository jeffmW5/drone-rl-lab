from .navigation_precision import PrecisionHold


def test_sustained_precision_and_reset():
    hold = PrecisionHold(seconds=.06)
    assert not hold.update(.0009, .004, 0., False, .02)
    assert not hold.update(.0009, .004, 0., False, .02)
    assert hold.update(.0009, .004, 0., False, .02)
    assert not hold.update(.0011, .004, 0., False, .02)
    assert hold.elapsed == 0


def test_transit_motion_collision_and_nan_cannot_pass():
    for error, speed, remaining, collision in [
        (.0001, .006, 0., False), (.0001, 0., .001, False),
        (.0001, 0., 0., True), (float('nan'), 0., 0., False)]:
        hold = PrecisionHold(seconds=.02)
        assert not hold.update(error, speed, remaining, collision, .02)


def test_precision_task_reports_3d_error_without_changing_policy_shape():
    import numpy as np
    from .generalize import make_task
    task = make_task(dict(task='random_navigation_precision_v1', residual_scale=.03,
                         episode_seconds=90, navigation={}))
    obs, _ = task.reset(seed=51001)
    assert obs.shape == (171,)
    for _ in range(10):
        obs, reward, done, truncated, info = task.step(np.zeros(4))
        assert np.isfinite(reward)
        assert info['payload_goal_error_mm'] == 1000*info['payload_goal_error_m']
        assert not info['precision_success']
        assert info['precision_hold_seconds'] == 0.
        assert not done and not truncated
    task.close()
