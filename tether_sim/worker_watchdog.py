"""Fail the parent promptly if a CPU simulator process exits unexpectedly."""
import os
import signal
import threading


class WorkerWatchdog:
    def __init__(self, processes, interval=1.):
        self.processes, self.interval = processes, interval
        self.stopped = threading.Event()
        self.failure = None
        self.thread = None

    def start(self):
        self.previous_handler = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, self._raise)
        def watch():
            while not self.stopped.wait(self.interval):
                dead = [(p.pid, p.exitcode) for p in self.processes if p.exitcode is not None]
                if dead:
                    self.failure = dead
                    os.kill(os.getpid(), signal.SIGUSR1)
                    return
        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()

    def _raise(self, signum, frame):
        raise ChildProcessError('Simulator worker exited unexpectedly: '+str(self.failure))

    def close(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=2)
            signal.signal(signal.SIGUSR1, self.previous_handler)
