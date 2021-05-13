"""
Utility for time measurements
"""
from collections import deque
import time


class Timer:
    # pylint: disable=invalid-name
    # dt, t0, etc.
    """
    Utility for timing measurements
    """

    def __init__(self):
        self._clock = deque()

    def start(self):
        """
        Starts a new clock.
        """
        self._clock.append(time.time())

    def isrunning(self):
        """
        Returns true if there is at least one clock running.
        """
        return len(self._clock) > 0

    def dt(self):
        """
        Stops the last added clock and returns elapsed time (in seconds).
        """
        assert self.isrunning()
        t0 = self._clock.pop()
        return time.time() - t0

    def lap(self):
        """
        Returns time elapsed since the last clock started.
        """
        assert self.isrunning()
        # Peek last added clock
        t0 = self._clock[-1]
        return time.time() - t0
