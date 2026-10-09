"""Official Bitcraze meshes plus illustrative AI/Flow deck geometry."""
from pathlib import Path


def visual_assets():
    root = Path(__file__).with_name('assets')/'bitcraze'
    names = ['cf_body','4_motormounts','4_motors','battery','battery_holder','2_pinheaders','cw_prop','ccw_prop']
    return ''.join(f'<mesh name="{n}" file="{(root/(n+".stl")).as_posix()}"/>' for n in names)


def visual_drone(motors):
    colors={'cf_body':'.05 .25 .12 1','4_motormounts':'.08 .09 .11 1','4_motors':'.65 .68 .72 1','battery':'.2 .25 .65 1','battery_holder':'.12 .13 .15 1','2_pinheaders':'.12 .12 .13 1'}
    geoms=''.join(f'<geom type="mesh" mesh="{n}" pos="0 0 -.016" rgba="{color}" contype="0" conaffinity="0" group="2" mass="0"/>' for n,color in colors.items())
    for i,(x,y) in enumerate(motors):
        geoms+=f'<geom name="prop{i}" type="mesh" mesh="{("cw_prop" if i%2 else "ccw_prop")}" pos="{x} {y} .006" rgba=".8 .83 .86 1" contype="0" conaffinity="0" mass="0" group="2"/><geom name="blur{i}" type="cylinder" pos="{x} {y} .006" size=".023 .0002" rgba=".7 .8 .9 .12" contype="0" conaffinity="0" mass="0" group="2"/>'
    geoms+='''<geom type="box" pos="0 0 .022" size=".015 .014 .001" rgba=".06 .35 .18 1" contype="0" conaffinity="0" mass="0" group="2"/><geom type="box" pos=".004 0 .025" size=".006 .006 .0015" rgba=".09 .1 .12 1" contype="0" conaffinity="0" mass="0" group="2"/><geom type="box" pos="0 0 -.016" size=".014 .014 .001" rgba=".05 .3 .17 1" contype="0" conaffinity="0" mass="0" group="2"/><geom type="cylinder" pos=".02 0 -.02" size=".003 .003" rgba=".06 .07 .09 1" contype="0" conaffinity="0" mass="0" group="2"/>'''
    return geoms
