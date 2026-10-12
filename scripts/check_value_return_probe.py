import json
from tether_sim.value_return_probe import completed_returns,summary
assert completed_returns([1,2,3,4],[False,True,False,False],.5)==[2.,2.,None,None]
assert completed_returns([1,2,3,4],[False,True,False,True],.5)==[2.,2.,5.,4.]
assert completed_returns([1],[False],.995)==[None]
assert completed_returns([1],[True],.995)==[1.]
assert summary([(1,1),(2,2)])['rmse']==0
assert summary([])['count']==0
assert summary([(2,1),(2,1)])['explained_variance'] is None
print(json.dumps(dict(status='pass',scope='CPU return folding with terminal resets, censored tails, summary zero/constant cases; GPU probe pending'),indent=2))
