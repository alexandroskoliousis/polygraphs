"""
Monitoring infrastructure
"""
import torch

from . import timer


class MonitorHook:
    """
    Periodic monitor for performance measurements
    """

    def __init__(self, interval=1, atend=True):
        self._interval = interval
        self._atend = atend
        # Timer that starts after the first step
        self._clock = timer.Timer()
        # Last processed step (to avoid duplicate runs at end)
        self._last = None

    def _isvalid(self, step):
        return step == 1 or step % self._interval == 0

    def _islast(self, step):
        return self._last and self._last == step

    def _run(self, step, polygraph):
        # Store last processed step
        self._last = step

        # Compute throughput
        if not self._clock.isrunning():
            # Ensure this is the first step
            assert step == 1
            # Start the clock
            self._clock.start()
            # Report 0 steps/s
            throughput = 0.0
        else:
            # Time elapsed since clock started
            dt = self._clock.lap()  # pylint: disable=invalid-name
            throughput = (step - 1) / dt / 1000.0

        # Number of nodes that believe action A (resp. B) is better
        beliefs = polygraph.ndata["beliefs"]
        a, b = torch.sum(
            torch.le(beliefs, 0.5)
        ), torch.sum(  # pylint: disable=invalid-name
            torch.gt(beliefs, 0.5)
        )
        # print(beliefs)
        # Log progress
        msg = "[MON]"
        msg = f"{msg} step {step:04d}"
        msg = f"{msg} Ksteps/s {throughput:6.2f}"
        msg = f"{msg} A/B {a / (a + b):4.2f}/{b / (a + b):4.2f}"
        print(msg)

    def mayberun(self, step, beliefs):
        """
        Monitors progress at given simulation step.
        """
        if not self._isvalid(step):
            return
        self._run(step, beliefs)

    def conclude(self, step, polygraph):
        """
        Concludes monitoring.
        """
        if not self._atend or self._islast(step):
            return
        self._run(step, polygraph)
