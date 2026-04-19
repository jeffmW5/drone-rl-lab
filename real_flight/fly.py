#!/usr/bin/env python3.11
"""Deploy trained RL policies to a real Crazyflie drone.

Usage:
    python3.11 fly.py check                          # Verify connection, battery, positioning
    python3.11 fly.py hover                           # PID hover test (no RL)
    python3.11 fly.py fly                             # Deploy RL policy
    python3.11 fly.py fly --checkpoint path/to/ckpt   # Override checkpoint
    python3.11 fly.py fly --no-gates                  # Hover-only policy test (no gate obs)

Setup:
    python3.11 -m pip install --user cflib pyyaml     # cflib may only be on 3.10 currently
    # torch should already be on 3.11

Requires:
    - Crazyflie 2.1 with Lighthouse V2 or Flow Deck V2
    - Crazyradio PA USB dongle
    - cflib + torch on Python 3.11

If running in a VM (VirtualBox), you need USB passthrough for the Crazyradio:
    VBox Manager -> Settings -> USB -> Add Filter -> Bitcraze Crazyradio PA
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fly")

LAB_DIR = Path(__file__).resolve().parent.parent

# ─── State container ───────────────────────────────────────────────────────────

@dataclass
class DroneState:
    """Thread-safe container for state estimates from the Crazyflie logging framework."""
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    quat: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1], dtype=np.float32))
    ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    battery: float = 0.0
    pos_var: float = 1.0  # Kalman filter position variance
    timestamp: float = 0.0
    _lock: Lock = field(default_factory=Lock)

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.timestamp = time.time()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "pos": self.pos.copy(),
                "vel": self.vel.copy(),
                "quat": self.quat.copy(),
                "ang_vel": self.ang_vel.copy(),
                "battery": self.battery,
                "pos_var": self.pos_var,
                "timestamp": self.timestamp,
            }


# ─── Config loading ───────────────────────────────────────────────────────────

def load_config(path: str | None = None) -> dict:
    config_path = Path(path) if path else Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─── Policy loader ────────────────────────────────────────────────────────────

def load_policy(cfg: dict):
    """Load trained policy from checkpoint. Returns (agent, obs_normalizer, obs_dim).

    Auto-detects obs_dim and hidden_size from checkpoint weight shapes so the architecture
    always matches, regardless of which wrappers were used during training.
    """
    import torch
    import torch.nn as nn

    pol = cfg["policy"]
    ckpt_path = LAB_DIR / pol["checkpoint"]

    log.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Handle checkpoints that include obs normalizer stats alongside model weights
    obs_normalizer = None
    if "obs_norm_mean" in ckpt:
        obs_normalizer = {
            "mean": ckpt.pop("obs_norm_mean").astype(np.float32),
            "var": ckpt.pop("obs_norm_var").astype(np.float32),
            "count": ckpt.pop("obs_norm_count"),
        }
        log.info(f"Loaded obs normalizer (count={obs_normalizer['count']:.0f})")

    # Auto-detect architecture from checkpoint weight shapes
    obs_dim = ckpt["actor_mean.0.weight"].shape[1]
    hidden_size = ckpt["actor_mean.0.weight"].shape[0]
    act_dim = ckpt["actor_mean.4.weight"].shape[0]

    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    class Agent(nn.Module):
        def __init__(self, obs_dim, act_dim, hidden):
            super().__init__()
            self.critic = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
                layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
                layer_init(nn.Linear(hidden, 1), std=1.0),
            )
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
                layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
                layer_init(nn.Linear(hidden, act_dim), std=0.01), nn.Tanh(),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        def get_action(self, x):
            return self.actor_mean(x)

    agent = Agent(obs_dim, act_dim, hidden_size)
    agent.load_state_dict(ckpt)
    agent.eval()
    log.info(f"Policy loaded: obs_dim={obs_dim}, hidden={hidden_size}, act_dim={act_dim}, params={sum(p.numel() for p in agent.parameters()):,}")

    return agent, obs_normalizer, obs_dim


# ─── Observation construction ─────────────────────────────────────────────────

class ObservationBuilder:
    """Constructs flattened observation vectors matching the training format."""

    def __init__(self, cfg: dict, obs_dim: int):
        pol = cfg["policy"]
        self.n_obs = pol.get("n_obs", 2)
        self.n_gates = len(pol.get("gates", []))
        self.n_obstacles = len(pol.get("obstacles", []))
        self.obs_dim = obs_dim

        # Pre-compute gate arrays
        self.gates_pos = np.zeros((self.n_gates, 3), dtype=np.float32)
        self.gates_quat = np.zeros((self.n_gates, 4), dtype=np.float32)
        for i, g in enumerate(pol.get("gates", [])):
            self.gates_pos[i] = g["pos"]
            yaw_rad = np.deg2rad(g.get("yaw", 0.0))
            cy, sy = np.cos(yaw_rad / 2), np.sin(yaw_rad / 2)
            self.gates_quat[i] = [0, 0, sy, cy]  # [x, y, z, w] quaternion

        self.obstacles_pos = np.zeros((self.n_obstacles, 3), dtype=np.float32)
        for i, o in enumerate(pol.get("obstacles", [])):
            self.obstacles_pos[i] = o["pos"]

        # Stacked observation buffer
        self.prev_obs = np.zeros((self.n_obs, 13), dtype=np.float32)
        self.last_action = np.zeros(4, dtype=np.float32)
        self.target_gate = 0
        self.gates_visited = np.zeros(self.n_gates, dtype=np.float32)
        self.obstacles_visited = np.zeros(self.n_obstacles, dtype=np.float32)
        self._dim_logged = False

    def build(self, state: dict) -> np.ndarray:
        """Build flattened obs vector from drone state dict.

        The exact obs format depends on which wrappers were used during training
        (make_race_envs on the GPU training server). We construct the standard race env
        observation and auto-adapt to the checkpoint's expected obs_dim:
        - If our obs matches obs_dim exactly, use it as-is
        - If our obs is longer, truncate (training wrappers may exclude some fields)
        - If our obs is shorter, zero-pad (training wrappers may add fields we don't have)
        This allows incremental testing while the exact format is being matched.
        """
        pos = state["pos"]
        quat = state["quat"]
        vel = state["vel"]
        ang_vel = state["ang_vel"]

        # Check gate passage (simple distance threshold)
        if 0 <= self.target_gate < self.n_gates:
            gate_pos = self.gates_pos[self.target_gate]
            dist = np.linalg.norm(pos[:2] - gate_pos[:2])
            if dist < 0.45:
                self.gates_visited[self.target_gate] = 1.0
                self.target_gate += 1
                if self.target_gate >= self.n_gates:
                    self.target_gate = -1
                log.info(f"Gate {self.target_gate - 1} passed! Next: {self.target_gate}")

        # Basic obs for stacking
        basic_obs = np.concatenate([pos, quat, vel, ang_vel])

        # Assemble observation in standard race env order (FlattenJaxObservation order)
        parts = [
            pos,                                        # 3
            quat,                                       # 4
            vel,                                        # 3
            ang_vel,                                    # 3
            np.array([self.target_gate], dtype=np.float32),  # 1
            self.gates_pos.flatten(),                   # n_gates * 3
            self.gates_quat.flatten(),                  # n_gates * 4
            self.gates_visited,                         # n_gates
            self.obstacles_pos.flatten(),               # n_obstacles * 3
            self.obstacles_visited,                     # n_obstacles
        ]

        obs = np.concatenate(parts).astype(np.float32)

        # Auto-adapt to checkpoint's expected obs_dim
        if obs.shape[0] > self.obs_dim:
            obs = obs[: self.obs_dim]
        elif obs.shape[0] < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - obs.shape[0]))

        if not self._dim_logged:
            raw_dim = sum(p.size for p in parts)
            log.info(f"Obs constructed: raw={raw_dim}D, checkpoint expects={self.obs_dim}D, "
                     f"{'exact match' if raw_dim == self.obs_dim else f'auto-adapted ({raw_dim}->{self.obs_dim})'}")
            self._dim_logged = True

        # Update stacked obs buffer
        self.prev_obs = np.roll(self.prev_obs, -1, axis=0)
        self.prev_obs[-1] = basic_obs
        return obs

    def record_action(self, action: np.ndarray):
        self.last_action = action.copy()


# ─── Flight data logger ───────────────────────────────────────────────────────

class FlightLogger:
    def __init__(self, cfg: dict):
        self.enabled = cfg.get("logging", {}).get("enabled", True)
        self.records = []
        if self.enabled:
            out_dir = LAB_DIR / cfg.get("logging", {}).get("output_dir", "real_flight/logs")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.path = out_dir / f"flight_{ts}.npz"

    def log(self, t: float, state: dict, obs: np.ndarray | None = None, action: np.ndarray | None = None):
        if not self.enabled:
            return
        record = {"t": t, **{k: v for k, v in state.items() if isinstance(v, np.ndarray)}}
        if obs is not None:
            record["obs"] = obs
        if action is not None:
            record["action"] = action
        self.records.append(record)

    def save(self):
        if not self.enabled or not self.records:
            return
        arrays = {}
        keys = self.records[0].keys()
        for k in keys:
            arrays[k] = np.array([r[k] for r in self.records])
        np.savez_compressed(self.path, **arrays)
        log.info(f"Flight data saved: {self.path} ({len(self.records)} samples)")


# ─── Safety monitor ───────────────────────────────────────────────────────────

class SafetyMonitor:
    def __init__(self, cfg: dict):
        geo = cfg["safety"]["geofence"]
        self.bounds_low = np.array([geo["x_min"], geo["y_min"], geo["z_min"]])
        self.bounds_high = np.array([geo["x_max"], geo["y_max"], geo["z_max"]])
        self.max_roll = np.deg2rad(cfg["safety"]["max_roll"])
        self.max_pitch = np.deg2rad(cfg["safety"]["max_pitch"])
        self.max_yaw_rate = np.deg2rad(cfg["safety"]["max_yaw_rate"])
        self.min_thrust_pwm = cfg["safety"]["min_thrust"]
        self.max_thrust_pwm = cfg["safety"]["max_thrust"]
        self.min_battery = cfg["safety"]["min_battery_voltage"]
        self.max_pos_var = cfg["safety"]["max_position_variance"]

    def check_state(self, state: dict) -> str | None:
        """Return error message if state is unsafe, else None."""
        pos = state["pos"]
        if np.any(pos < self.bounds_low) or np.any(pos > self.bounds_high):
            return f"GEOFENCE VIOLATION: pos={pos}"
        if state["battery"] > 0 and state["battery"] < self.min_battery:
            return f"LOW BATTERY: {state['battery']:.2f}V"
        if state["pos_var"] > self.max_pos_var and state["timestamp"] > 0:
            return f"HIGH POSITION VARIANCE: {state['pos_var']:.6f}"
        return None

    def clamp_action(self, roll: float, pitch: float, yaw_rate: float, thrust_pwm: int) -> tuple:
        """Clamp action within safety limits."""
        roll = np.clip(roll, -self.max_roll, self.max_roll)
        pitch = np.clip(pitch, -self.max_pitch, self.max_pitch)
        yaw_rate = np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)
        thrust_pwm = int(np.clip(thrust_pwm, self.min_thrust_pwm, self.max_thrust_pwm))
        return roll, pitch, yaw_rate, thrust_pwm


# ─── Action scaling ───────────────────────────────────────────────────────────

def scale_action(action: np.ndarray, thrust_min: float, thrust_max: float) -> tuple:
    """Convert [-1,1] policy output to physical commands.

    Returns: (roll_rad, pitch_rad, yaw_rate_rad, thrust_newtons)
    """
    action = np.clip(action, -1.0, 1.0)
    roll = action[0] * (np.pi / 2)
    pitch = action[1] * (np.pi / 2)
    yaw_rate = 0.0  # Zeroed in training
    thrust = action[3] * (thrust_max - thrust_min) / 2 + (thrust_max + thrust_min) / 2
    return roll, pitch, yaw_rate, thrust


def thrust_to_pwm(thrust_newtons: float, thrust_max: float, pwm_max: int = 65535) -> int:
    """Convert thrust in Newtons to PWM value.

    Matches force2pwm from drone_models: linear mapping from [0, thrust_max] to [0, pwm_max].
    """
    ratio = np.clip(thrust_newtons / thrust_max, 0.0, 1.0)
    return int(ratio * pwm_max)


# ─── Crazyflie connection ─────────────────────────────────────────────────────

def connect_crazyflie(cfg: dict):
    """Connect to Crazyflie and set up logging. Returns (scf, drone_state)."""
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.syncLogger import SyncLogger

    cflib.crtp.init_drivers()
    uri = cfg["radio"]["uri"]
    log.info(f"Connecting to {uri}...")

    cf = Crazyflie(rw_cache="./cf_cache")
    scf = SyncCrazyflie(uri, cf=cf)
    scf.open_link()
    log.info("Connected!")

    # Apply firmware settings
    fw = cfg["firmware"]
    cf = scf.cf
    cf.param.set_value("stabilizer.estimator", fw["estimator"])
    time.sleep(0.1)
    cf.param.set_value("stabilizer.controller", fw["controller"])
    cf.param.set_value("supervisor.tmblChckEn", 1 if fw["tumble_check"] else 0)
    cf.param.set_value("flightmode.stabModeRoll", 1)  # angle mode
    cf.param.set_value("flightmode.stabModePitch", 1)
    cf.param.set_value("flightmode.stabModeYaw", 0)   # rate mode for yaw
    time.sleep(0.1)
    log.info("Firmware parameters set (Mellinger controller, Kalman estimator)")

    # Set up state logging
    state = DroneState()
    log_freq = cfg["control"]["log_freq"]

    lg_state = LogConfig(name="State", period_in_ms=max(10, 1000 // log_freq))
    lg_state.add_variable("stateEstimate.x", "float")
    lg_state.add_variable("stateEstimate.y", "float")
    lg_state.add_variable("stateEstimate.z", "float")
    lg_state.add_variable("stateEstimate.vx", "float")
    lg_state.add_variable("stateEstimate.vy", "float")
    lg_state.add_variable("stateEstimate.vz", "float")

    lg_att = LogConfig(name="Attitude", period_in_ms=max(10, 1000 // log_freq))
    lg_att.add_variable("stabilizer.qx", "float")
    lg_att.add_variable("stabilizer.qy", "float")
    lg_att.add_variable("stabilizer.qz", "float")
    lg_att.add_variable("stabilizer.qw", "float")
    lg_att.add_variable("gyro.x", "float")
    lg_att.add_variable("gyro.y", "float")
    lg_att.add_variable("gyro.z", "float")

    lg_sys = LogConfig(name="System", period_in_ms=500)
    lg_sys.add_variable("pm.vbat", "float")
    lg_sys.add_variable("kalman.varPX", "float")

    def _state_cb(timestamp, data, logconf):
        state.update(
            pos=np.array([data["stateEstimate.x"], data["stateEstimate.y"], data["stateEstimate.z"]], dtype=np.float32),
            vel=np.array([data["stateEstimate.vx"], data["stateEstimate.vy"], data["stateEstimate.vz"]], dtype=np.float32),
        )

    def _att_cb(timestamp, data, logconf):
        state.update(
            quat=np.array([data["stabilizer.qx"], data["stabilizer.qy"], data["stabilizer.qz"], data["stabilizer.qw"]], dtype=np.float32),
            ang_vel=np.deg2rad(np.array([data["gyro.x"], data["gyro.y"], data["gyro.z"]], dtype=np.float32)),
        )

    def _sys_cb(timestamp, data, logconf):
        state.update(
            battery=data["pm.vbat"],
            pos_var=data["kalman.varPX"],
        )

    cf.log.add_config(lg_state)
    cf.log.add_config(lg_att)
    cf.log.add_config(lg_sys)
    lg_state.data_received_cb.add_callback(_state_cb)
    lg_att.data_received_cb.add_callback(_att_cb)
    lg_sys.data_received_cb.add_callback(_sys_cb)
    lg_state.start()
    lg_att.start()
    lg_sys.start()

    warmup = cfg["control"]["warmup_seconds"]
    log.info(f"Waiting {warmup}s for estimator convergence...")
    time.sleep(warmup)

    return scf, state


# ─── Emergency stop ───────────────────────────────────────────────────────────

def emergency_stop(cf):
    """Send emergency stop and kill motors immediately."""
    try:
        cf.commander.send_stop_setpoint()
        pk = struct.pack("<B", 0)  # EMERGENCY_STOP
        from cflib.crtp.crtpstack import CRTPPacket, CRTPPort
        from cflib.crazyflie import Localization
        packet = CRTPPacket()
        packet.port = CRTPPort.LOCALIZATION
        packet.channel = Localization.GENERIC_CH
        packet.data = struct.pack("<B", Localization.EMERGENCY_STOP)
        cf.send_packet(packet)
    except Exception:
        pass
    log.warning("EMERGENCY STOP SENT")


# ─── Flight modes ─────────────────────────────────────────────────────────────

def cmd_check(cfg: dict):
    """Check connection, battery, and positioning."""
    scf, state = connect_crazyflie(cfg)
    try:
        snap = state.snapshot()
        print("\n=== Crazyflie Status ===")
        print(f"  Battery:  {snap['battery']:.2f} V")
        print(f"  Position: [{snap['pos'][0]:.3f}, {snap['pos'][1]:.3f}, {snap['pos'][2]:.3f}]")
        print(f"  Velocity: [{snap['vel'][0]:.3f}, {snap['vel'][1]:.3f}, {snap['vel'][2]:.3f}]")
        print(f"  Quaternion: [{snap['quat'][0]:.3f}, {snap['quat'][1]:.3f}, {snap['quat'][2]:.3f}, {snap['quat'][3]:.3f}]")
        print(f"  Pos variance: {snap['pos_var']:.6f}")

        min_bat = cfg["safety"]["min_battery_voltage"]
        max_var = cfg["safety"]["max_position_variance"]
        ok = True
        if snap["battery"] < min_bat:
            print(f"  [FAIL] Battery below {min_bat}V")
            ok = False
        if snap["pos_var"] > max_var:
            print(f"  [WARN] Position variance {snap['pos_var']:.6f} > {max_var} — estimator may not have converged")
            print(f"         Check Lighthouse / Flow Deck setup")
        if abs(snap["pos"][2]) > 0.1:
            print(f"  [WARN] Z={snap['pos'][2]:.3f} — drone may not be on flat ground, or estimator is drifting")

        if ok:
            print("\n  [OK] Ready to fly!")
        print()
    finally:
        scf.close_link()


def cmd_hover(cfg: dict):
    """Basic PID hover test — no RL policy, uses high-level commander."""
    scf, state = connect_crazyflie(cfg)
    cf = scf.cf
    safety = SafetyMonitor(cfg)
    flight_log = FlightLogger(cfg)

    height = cfg["control"]["hover_height"]
    duration = cfg["control"]["hover_duration"]

    # Kill switch
    abort = Event()
    def _sigint(sig, frame):
        abort.set()
    signal.signal(signal.SIGINT, _sigint)

    try:
        snap = state.snapshot()
        violation = safety.check_state(snap)
        if violation:
            log.error(f"Pre-flight safety check failed: {violation}")
            return

        log.info(f"Hovering at {height}m for {duration}s...")
        log.info("Press Ctrl+C to abort")

        # Use high-level commander for hover (PID, no RL)
        cf.param.set_value("commander.enHighLevel", "1")
        cf.platform.send_arming_request(True)
        cf.high_level_commander.takeoff(height, 2.0)

        t_start = time.time()
        while (time.time() - t_start) < duration + 2.0 and not abort.is_set():
            snap = state.snapshot()
            violation = safety.check_state(snap)
            if violation:
                log.error(f"Safety violation during hover: {violation}")
                break
            flight_log.log(time.time() - t_start, snap)
            time.sleep(0.05)

        log.info("Landing...")
        cf.high_level_commander.land(0.05, 3.0)
        time.sleep(3.5)

    except Exception as e:
        log.error(f"Error during hover: {e}")
        emergency_stop(cf)
    finally:
        cf.commander.send_stop_setpoint()
        flight_log.save()
        scf.close_link()


def cmd_fly(cfg: dict, checkpoint: str | None = None, no_gates: bool = False):
    """Deploy trained RL policy on real drone."""
    import torch

    if checkpoint:
        cfg["policy"]["checkpoint"] = checkpoint

    # Load policy
    agent, obs_normalizer, obs_dim = load_policy(cfg)

    if no_gates:
        cfg["policy"]["gates"] = []
        cfg["policy"]["obstacles"] = []
        agent, obs_normalizer, obs_dim = load_policy(cfg)

    obs_builder = ObservationBuilder(cfg, obs_dim)
    safety = SafetyMonitor(cfg)
    flight_log = FlightLogger(cfg)

    # Drone params for action scaling (from config, cf21B_500)
    drone = cfg.get("drone", {})
    thrust_min = drone.get("thrust_min", 0.08545)
    thrust_max = drone.get("thrust_max", 0.8)

    freq = cfg["control"]["freq"]
    hover_height = cfg["control"]["hover_height"]
    hover_duration = cfg["control"]["hover_duration"]
    dt = 1.0 / freq

    scf, state = connect_crazyflie(cfg)
    cf = scf.cf

    abort = Event()
    def _sigint(sig, frame):
        abort.set()
    signal.signal(signal.SIGINT, _sigint)

    try:
        # Pre-flight check
        snap = state.snapshot()
        violation = safety.check_state(snap)
        if violation:
            log.error(f"Pre-flight safety check failed: {violation}")
            return

        log.info(f"Phase 1: High-level takeoff to {hover_height}m...")
        cf.param.set_value("commander.enHighLevel", "1")
        cf.platform.send_arming_request(True)
        cf.high_level_commander.takeoff(hover_height, 2.0)
        time.sleep(2.5)

        # Stabilize at hover before switching to RL
        log.info(f"Phase 2: Stabilizing hover for {hover_duration}s...")
        t_hover_start = time.time()
        while (time.time() - t_hover_start) < hover_duration and not abort.is_set():
            snap = state.snapshot()
            violation = safety.check_state(snap)
            if violation:
                log.error(f"Safety violation during stabilization: {violation}")
                emergency_stop(cf)
                return
            time.sleep(0.05)

        if abort.is_set():
            log.info("Aborted during hover phase")
            cf.high_level_commander.land(0.05, 3.0)
            time.sleep(3.5)
            return

        # Switch to low-level attitude control for RL
        log.info("Phase 3: Switching to RL policy control...")
        cf.param.set_value("commander.enHighLevel", "0")
        cf.commander.send_setpoint(0, 0, 0, 0)  # Unlock thrust protection

        # Warm up observation buffer
        for _ in range(obs_builder.n_obs + 1):
            snap = state.snapshot()
            obs_builder.build(snap)

        log.info("RL policy active! Press Ctrl+C to land.")
        t_start = time.time()
        step = 0

        while not abort.is_set():
            t_loop = time.time()

            snap = state.snapshot()
            violation = safety.check_state(snap)
            if violation:
                log.error(f"Safety violation: {violation}")
                break

            # Build observation
            obs = obs_builder.build(snap)

            # Normalize if trained with obs normalization
            if obs_normalizer is not None:
                mean = obs_normalizer["mean"]
                var = obs_normalizer["var"]
                obs = np.clip((obs - mean) / np.sqrt(var + 1e-8), -10.0, 10.0)

            # Policy inference
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = agent.get_action(obs_tensor).squeeze(0).numpy()

            # Scale and clamp
            roll, pitch, yaw_rate, thrust = scale_action(action, thrust_min, thrust_max)
            thrust_pwm = thrust_to_pwm(thrust, thrust_max)
            roll, pitch, yaw_rate, thrust_pwm = safety.clamp_action(roll, pitch, yaw_rate, thrust_pwm)

            # Send command (cflib expects degrees for roll/pitch, deg/s for yaw)
            cf.commander.send_setpoint(
                np.rad2deg(roll),
                np.rad2deg(pitch),
                np.rad2deg(yaw_rate),
                thrust_pwm,
            )

            obs_builder.record_action(action)
            flight_log.log(time.time() - t_start, snap, obs=obs, action=action)

            step += 1
            if step % (freq * 2) == 0:
                log.info(f"  t={time.time()-t_start:.1f}s pos=[{snap['pos'][0]:.2f}, {snap['pos'][1]:.2f}, {snap['pos'][2]:.2f}] bat={snap['battery']:.2f}V gate={obs_builder.target_gate}")

            # Maintain control frequency
            elapsed = time.time() - t_loop
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Landing
        log.info("Landing...")
        cf.commander.send_stop_setpoint()
        cf.commander.send_notify_setpoint_stop()
        cf.param.set_value("commander.enHighLevel", "1")
        cf.platform.send_arming_request(True)
        cf.high_level_commander.land(0.05, cfg["control"]["landing_duration"])
        time.sleep(cfg["control"]["landing_duration"] + 0.5)

    except Exception as e:
        log.error(f"Error during flight: {e}")
        import traceback
        traceback.print_exc()
        emergency_stop(cf)
    finally:
        cf.commander.send_stop_setpoint()
        flight_log.save()
        scf.close_link()
        log.info("Connection closed.")


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deploy RL policies to Crazyflie")
    parser.add_argument("mode", choices=["check", "hover", "fly"], help="Flight mode")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Override policy checkpoint path")
    parser.add_argument("--no-gates", action="store_true", help="Ignore gate observations")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "check":
        cmd_check(cfg)
    elif args.mode == "hover":
        cmd_hover(cfg)
    elif args.mode == "fly":
        cmd_fly(cfg, checkpoint=args.checkpoint, no_gates=args.no_gates)


if __name__ == "__main__":
    main()
