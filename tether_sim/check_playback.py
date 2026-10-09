"""Verify reloaded policy reproduces its saved evaluation, including normalization."""
import argparse
import json
import numpy as np
from .playback import PolicyPlayback


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',default='tmp/tether-runs/tether_v002')
    args=p.parse_args()
    player=PolicyPlayback(args.run)
    results=[]
    for seed in player.config['eval_seeds']:
        player.seed=seed
        player.reset()
        errors=[]
        for _ in range(600):
            _,_,done,truncated,info=player.step()
            assert not done
            errors.append(info['payload_error_m'])
            if truncated:
                break
        results.append(float(np.sqrt(np.mean(np.square(errors)))))
    actual=float(np.mean(results))
    evaluations=[json.loads(p.read_text()) for p in player.run.glob('eval_*.json')]
    expected=min(e['PPO_residual']['mean_payload_rmse_m'] for e in evaluations if e['PPO_residual']['crashes']==0)
    assert abs(actual-expected)<1e-7,(actual,expected)
    print(json.dumps(dict(checkpoint='best',reloaded_rmse_m=actual,recorded_rmse_m=expected,inferences=player.inferences,matched=True)))
    player.env.close()


if __name__=='__main__':
    main()
