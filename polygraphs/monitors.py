"""
Monitoring infrastructure
"""
import os
import abc
import torch
import h5py

from . import timer


class BasicHook(metaclass=abc.ABCMeta):
    """
    Abstract periodic monitor
    """

    def __init__(self, interval=1, atend=True):
        super().__init__()
        self._interval = interval
        self._atend = atend
        # Last processed step (to avoid duplicate runs at end)
        self._last = None

    def _isvalid(self, step):
        return step == 1 or step % self._interval == 0

    def _islast(self, step):
        return self._last and self._last == step

    def _run(self, step, polygraph):
        raise NotImplementedError

    def mayberun(self, step, polygraph):
        """
        Monitors progress at given simulation step.
        """
        if not self._isvalid(step):
            return
        # Store last processed step
        self._last = step
        # User-defined run method
        self._run(step, polygraph)

    def conclude(self, step, polygraph):
        """
        Concludes monitoring.
        """
        if not self._atend or self._islast(step):
            return
        self._run(step, polygraph)


class MonitorHook(BasicHook):
    """
    Periodic monitor for performance measurements
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Timer that starts after the first step
        # used to measure throughput (steps/s)
        self._clock = timer.Timer()

    def _run(self, step, polygraph):
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
            throughput = ((step - 1) / dt) / 1000.0
            
        # Number of nodes that believe action A (resp. B) is better
        beliefs = polygraph.ndata["beliefs"]
        a, b = torch.sum(  # pylint: disable=invalid-name
            torch.le(beliefs, 0.5)
        ), torch.sum(torch.gt(beliefs, 0.5))
        # print(beliefs)
        # Log progress
        msg = "[MON]"
        msg = f"{msg} step {step:04d}"
        msg = f"{msg} Ksteps/s {throughput:6.2f}"
        msg = f"{msg} A/B {a / (a + b):4.2f}/{b / (a + b):4.2f}"
        print(msg)


class SnapshotHook(BasicHook):
    """
    Periodic logger for agent beliefs
    """

    def __init__(self, messages=False, location=None, filename="data.hd5", **kwargs):
        super().__init__(**kwargs)
        # Store snapshots in user-specified directory
        assert location and os.path.isdir(location)
        # Construct HDF5 filename
        self._filename = os.path.join(location, filename)
        # Whether to snapshot messages or not
        self._messages = messages

    def _run(self, step, polygraph):
        # Create dataset file, or read/write if exists
        f = h5py.File(self._filename, "a")  # pylint: disable=invalid-name

        # Store beliefs
        beliefs = polygraph.ndata["beliefs"].cpu().numpy()
        # Create or modify group
        grp = f.require_group("beliefs")
        # Create new dataset
        grp.create_dataset(str(step), data=beliefs)
        
        # Store messages
        if self._messages:
            payoffs = polygraph.ndata["payoffs"].cpu().numpy()
            # Create or modify group
            grp = f.require_group("payoffs")
            # Create new dataset
            grp.create_dataset(str(step), data=payoffs)

        # Close file
        f.close()
