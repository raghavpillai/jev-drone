from types import SimpleNamespace

import numpy as np

from jev_drone.planning.visibility_memory import VisibilityMemory, sample_depth
from jev_drone.world.layout import HOUSE


def frame(name='east',depth=2.,time=1.):
    return SimpleNamespace(camera_name=name,time=time,depth=np.full((12,16),depth),
        origin=np.array([19.,9.,1.6]),forward=np.array([1.,0,0]),right=np.array([0.,-1,0]),
        up=np.array([0.,0,1]),focal=8.)


def test_front_visibility_remembers_near_space_without_marking_hidden_or_rear_space():
    f=frame();samples=np.array([[20.,9,1.6],[21.,9,1.6],[22.,9,1.6],[18.,9,1.6]])
    free,surface=sample_depth(samples,f.depth,f.origin,f.forward,f.right,f.up,f.focal)
    assert free.tolist()==[True,False,False,False]
    assert surface.tolist()==[False,True,False,False]


def test_auxiliary_depth_and_missing_pixels_do_not_supply_search_memory():
    memory=VisibilityMemory(HOUSE);initial=memory.summary('workshop')
    memory.observe([frame('west'),frame(depth=np.nan)])
    assert memory.summary('workshop')==initial
    memory.observe([frame(time=2.)])
    assert sum(row['free_samples'] for row in memory.summary('workshop'))>0
    assert all(row['free_samples']==0 for row in memory.summary('library'))
    memory.observe([frame(time=1.,depth=10.)])
    assert memory.stamps['workshop']==2.
