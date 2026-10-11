"""Artifact-monitor regressions: crashed runs must never appear live."""
import json
import tempfile
import unittest
from pathlib import Path
from tether_sim.monitor_snapshot import snapshot


class MonitorTests(unittest.TestCase):
    def test_stale_status_is_not_training_and_incomplete_pair_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'interrupted';run.mkdir()
            (run/'config.json').write_text('{}')
            (run/'status.json').write_text(json.dumps({'status':'training','timesteps':1234}))
            (run/'best.zip').touch();(run/'final.zip').touch()
            report=snapshot(Path(tmp))['runs'][0]
            self.assertEqual(report['effective'],'interrupted / stale')
            self.assertFalse(report['active'])
            self.assertFalse(report['playback_ready'])
            (run/'best_normalize.pkl').touch();(run/'final_normalize.pkl').touch()
            self.assertTrue(snapshot(Path(tmp))['runs'][0]['playback_ready'])

    def test_partial_status_and_nonfinite_diagnostics_do_not_break_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'partial';run.mkdir()
            (run/'config.json').write_text('{}')
            (run/'status.json').write_text('{')
            (run/'progress.csv').write_text('time/total_timesteps,train/value_loss\n512,nan\n1024,0.2\n')
            report=snapshot(Path(tmp))
            json.dumps(report,allow_nan=False)
            self.assertEqual(len(report['runs'][0]['curves']),2)


if __name__=='__main__': unittest.main()
