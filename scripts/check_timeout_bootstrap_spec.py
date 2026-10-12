"""CPU-only specification for a future timeout bootstrap integration.
This is not a test of the installed PufferLib learner and changes no training.
"""
import json

def target(raw_reward, gamma, final_value, timeout, crash):
    clipped = max(-1.0, min(1.0, raw_reward))
    return clipped + (gamma * final_value if timeout and not crash else 0.0)

def check():
    g = .995
    # Crash dominates a simultaneous artificial cutoff; reset value is irrelevant.
    assert target(-5, g, 100, True, True) == -1
    assert target(.25, g, 2, True, False) == .25 + g * 2
    assert target(.25, g, 2, False, False) == .25
    # Clipping must precede augmentation or positive final values get discarded.
    assert target(5, g, 2, True, False) == 1 + g * 2
    assert target(-5, g, -2, True, False) == -1 - g * 2
    # Keep the reset boundary terminal after augmenting reward: no next-episode GAE.
    r = target(.25, g, 2, True, False)
    current_value = .4
    boundary_advantage = r - current_value
    for reset_value, next_episode_advantage in [(0, 0), (100, -100), (-100, 100)]:
        done = 1
        actual = r + g * (1-done) * reset_value - current_value + g * .95 * (1-done) * next_episode_advantage
        assert abs(actual - boundary_advantage) < 1e-12
    return dict(status='pass', scope='Scalar return specification only; no installed learner or GPU integration tested', checks=8)

if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
