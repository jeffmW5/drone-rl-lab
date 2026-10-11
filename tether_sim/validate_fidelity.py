"""Repeatable model checks, not a hardware-calibration certificate."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import mujoco
from .env import Config,model_xml
from .navigation_v2 import NavigationTaskV2


def pendulum(config):
    model=mujoco.MjModel.from_xml_string(model_xml(config).replace('<freejoint/>',''))
    data=mujoco.MjData(model)
    angle=.06
    data.qpos[:4]=[np.cos(angle/2),0,np.sin(angle/2),0]
    mujoco.mj_forward(model,data)
    payload=model.body('payload').id
    times=[];positions=[]
    for _ in range(round(6/config.timestep)):
        mujoco.mj_step(model,data)
        times.append(float(data.time));positions.append(float(data.xpos[payload,0]))
    x=np.array(positions);t=np.array(times)
    crossings=[]
    for i in np.flatnonzero((x[:-1]<=0)&(x[1:]>0)):
        crossings.append(float(t[i]+(-x[i])/(x[i+1]-x[i])*(t[i+1]-t[i])))
    period=float(np.mean(np.diff(crossings))) if len(crossings)>1 else None
    # Rigid-rod comparison is a diagnostic, not the exact flexible-chain solution.
    ds=config.length/config.segments
    rod_I=config.payload_mass*config.length**2+sum(config.cable_mass/config.segments*((i+.5)*ds)**2 for i in range(config.segments))
    first_moment=config.payload_mass*config.length+config.cable_mass*config.length/2
    reference=float(2*np.pi*np.sqrt(rod_I/(9.81*first_moment)))
    return dict(segments=config.segments,timestep_s=config.timestep,period_s=period,
                rigid_rod_reference_period_s=reference,positive_crossings_s=crossings,
                relative_reference_difference=abs(period-reference)/reference if period else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    checks=[pendulum(Config(segments=n,timestep=dt)) for n,dt in ((6,.002),(12,.002),(12,.001),(24,.001))]
    for c in checks:
        if c['period_s'] is None or not np.isfinite(c['period_s']):raise RuntimeError('No finite pendulum period')
    diff=abs(checks[1]['period_s']-checks[2]['period_s'])/checks[2]['period_s']
    result=dict(kind='simulator consistency; not hardware validation',pendulum=checks,
                timestep_period_relative_difference=diff,
                hardware='Crazyflie 2.1 with Flow deck and AI deck (user confirmed 2026-10-11)',
                nominal_config=asdict(Config()),
                calibration_required=['weighed assembled vehicle including battery','payload and cable mass/length/diameter','thrust vs command and battery voltage','inertia/attachment geometry','free swing decay and slack/contact recordings','camera intrinsics and measured capture/control latency'],
                limitations=['rigid segmented links can transmit compression','drag/thrust/inertia not fitted to real flights','primitive rotor/deck volumes','navigation uses privileged state and known static obstacles'])
    if diff>.02:raise RuntimeError('Pendulum timestep sensitivity exceeds 2 percent')
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'period_12links_s':checks[1]['period_s'],'timestep_relative_difference':diff,'output':str(path)}))

if __name__=='__main__':main()
