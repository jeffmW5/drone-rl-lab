"""Opt-in destination precision curriculum; V2 experiments stay unchanged."""
import numpy as np
from .navigation_v2 import NavigationTaskV2


class PrecisionHold:
    """Require continuous residence, rather than a lucky final sample."""
    def __init__(self, tolerance=.001, seconds=3., speed=.005):
        if min(tolerance, seconds, speed) <= 0:
            raise ValueError('Precision limits must be positive')
        self.tolerance, self.seconds, self.speed = tolerance, seconds, speed
        self.elapsed = 0.

    def update(self, error, velocity, route_remaining, collision, dt):
        valid = (np.isfinite(error) and np.isfinite(velocity) and
                 error <= self.tolerance and velocity <= self.speed and
                 route_remaining <= 1e-6 and not collision)
        self.elapsed = self.elapsed + dt if valid else 0.
        return self.elapsed + 1e-9 >= self.seconds


class NavigationPrecisionTask(NavigationTaskV2):
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.hold = PrecisionHold(
            self.navigation.get('payload_tolerance_m', .001),
            self.navigation.get('hold_seconds', 3.),
            self.navigation.get('payload_speed_limit_m_s', .005))
        self.last_payload = self.env.data.xpos[self.env.payload].copy()
        self.peak_hold = 0.
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        payload = self.env.data.xpos[self.env.payload].copy()
        speed = float(np.linalg.norm(payload-self.last_payload)/self.env.config.control_dt)
        self.last_payload = payload
        success = self.hold.update(info['payload_goal_error_m'], speed,
                                   info['route_remaining_m'], terminated,
                                   self.env.config.control_dt)
        self.peak_hold = max(self.peak_hold, self.hold.elapsed)
        info.update(precision_success=bool(success and not terminated),
                    precision_hold_seconds=self.hold.elapsed,
                    peak_precision_hold_seconds=self.peak_hold,
                    payload_speed_m_s=speed,
                    payload_goal_error_mm=1000*info['payload_goal_error_m'])
        # Smooth logarithmic shaping retains gradients at 10 cm, 1 cm and 1 mm.
        # Only apply it at the destination, so transit is not punished as hover.
        if info['route_remaining_m'] <= 1e-6:
            reward -= .1*float(np.log1p(info['payload_goal_error_m']/self.hold.tolerance))
            reward -= .05*min(speed/self.hold.speed, 10.)
        # End after an actual sustained success; evaluation records this reason.
        return obs, float(reward), terminated, bool(truncated or success), info
