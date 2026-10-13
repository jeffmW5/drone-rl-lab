"""Opt-in evaluation terminal evidence captured before CudaVector auto-reset.

The caller supplies the existing terminal observer's five tensors plus the
one-based absolute evaluation step and the vector's pre-reset episode counters.
This module neither controls simulation nor changes evaluation metrics.
"""
import copy
import math
import operator


def _positive_integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a positive integer')
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f'{name} must be a positive integer') from error
    if result < 1:
        raise ValueError(f'{name} must be a positive integer')
    return result


def _cpu_values(value, shape, name):
    if tuple(value.shape) != shape:
        raise ValueError(f'{name} has shape {tuple(value.shape)}, expected {shape}')
    return value.detach().cpu().tolist()


def _finite_json(value):
    """Replace only nonfinite numbers with null, preserving an explicit count."""
    if isinstance(value, float) and not math.isfinite(value):
        return None, 1
    if isinstance(value, list):
        result = [_finite_json(item) for item in value]
        return [item for item, count in result], sum(count for item, count in result)
    if isinstance(value, dict):
        result = {key: _finite_json(item) for key, item in value.items()}
        return {key: item for key, (item, count) in result.items()}, sum(
            count for item, count in result.values())
    return value, 0


class EvalTerminalTrace:
    """Record done events, with a finite bound of evaluation_steps * num_envs.

    absolute_step is one-based: the first send call is step 1. episode_steps
    must come from vec.steps inside the pre-reset callback, after send increments
    it. Decoded states inherit the policy observation's precision (typically FP32),
    not the full-FP64 solver state.
    """
    version = 1

    def __init__(self, num_envs, evaluation_steps, control_dt=.02):
        self.num_envs = _positive_integer(num_envs, 'num_envs')
        self.evaluation_steps = _positive_integer(evaluation_steps, 'evaluation_steps')
        self.control_dt = float(control_dt)
        if not math.isfinite(self.control_dt) or self.control_dt <= 0:
            raise ValueError('control_dt must be finite and positive')
        self.max_events = self.num_envs * self.evaluation_steps
        self._events = []
        self._first = [None] * self.num_envs
        self._seen = set()
        self._last_absolute_step = 0

    def record(self, obs, done, truncated, failed, reward, *, absolute_step,
               episode_steps):
        """Consume detached copies of the pre-reset tensors; never mutate them."""
        absolute_step = _positive_integer(absolute_step, 'absolute_step')
        if absolute_step > self.evaluation_steps:
            raise ValueError('absolute_step exceeds the evaluation bound')
        if absolute_step < self._last_absolute_step:
            raise ValueError('terminal chronology must be nondecreasing')
        observations = _cpu_values(obs, (self.num_envs, 98), 'obs')
        dones = _cpu_values(done, (self.num_envs,), 'done')
        truncations = _cpu_values(truncated, (self.num_envs,), 'truncated')
        failures = _cpu_values(failed, (self.num_envs,), 'failed')
        rewards = _cpu_values(reward, (self.num_envs,), 'reward')
        counters = _cpu_values(episode_steps, (self.num_envs,), 'episode_steps')
        pending = []
        for lane, is_done in enumerate(dones):
            if not is_done:
                continue
            key = (absolute_step, lane)
            if key in self._seen:
                raise ValueError('duplicate terminal event for the same step and lane')
            row = observations[lane]
            episode_step = _positive_integer(counters[lane], 'episode_steps')
            try:
                failure = operator.index(failures[lane])
            except TypeError as error:
                raise ValueError('failed codes must be integers') from error
            if failure < 0:
                raise ValueError('failed codes must be nonnegative')
            is_truncated = bool(truncations[lane])
            reason = ('failure_with_truncation' if is_truncated else 'failure') if failure else (
                'truncation' if is_truncated else 'terminal_without_failure_flag')
            goal = row[95:98]
            positions = [[row[3 * particle + axis] + goal[axis] for axis in range(3)]
                         for particle in range(14)]
            event = dict(lane=lane, absolute_step=absolute_step,
                         absolute_time_s=absolute_step * self.control_dt,
                         episode_step=episode_step,
                         episode_time_s=episode_step * self.control_dt,
                         done=True, truncated=is_truncated, failed=failure,
                         reward=float(rewards[lane]), reason=reason,
                         positions=positions,
                         velocities=[row[42 + 3 * particle:45 + 3 * particle]
                                     for particle in range(14)],
                         orientation=row[84:91], motors=row[91:95], goal=goal)
            event['nonfinite_observation_indices'] = [index for index, value in enumerate(row)
                                                       if not math.isfinite(float(value))]
            event['nonfinite_reward'] = not math.isfinite(float(rewards[lane]))
            event['has_nonfinite'] = bool(event['nonfinite_observation_indices'] or event['nonfinite_reward'])
            event, nonfinite_count = _finite_json(event)
            event['nonfinite_output_value_count'] = nonfinite_count
            pending.append((key, event))
        if len(self._events) + len(pending) > self.max_events:
            raise ValueError('terminal event count exceeds the evaluation bound')
        # Commit only after all rows validate, so a malformed batch is atomic.
        for key, event in pending:
            self._seen.add(key)
            self._events.append(event)
            if self._first[event['lane']] is None:
                self._first[event['lane']] = event
        self._last_absolute_step = absolute_step

    def as_dict(self):
        """Return independent JSON-ready evidence, including first/all chronology."""
        return copy.deepcopy(dict(version=self.version, num_envs=self.num_envs,
                                  evaluation_steps=self.evaluation_steps,
                                  control_dt_s=self.control_dt, max_events=self.max_events,
                                  terminal_count=len(self._events),
                                  first_terminal_by_lane=self._first, terminals=self._events,
                                  scope='Decoded pre-reset 98D policy observations, typically FP32; '
                                        'not a full-FP64 solver snapshot. Steps/timestamps are exact, '
                                        'but state precision is bounded by the observation. Flag-only '
                                        'reasons do not infer contact cause or controller intent. '
                                        'No metrics, rank, policy or dynamics changes.'))


