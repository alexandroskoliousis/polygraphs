"""
A collection of metadata associated with PolyGraph simulations
"""
import os
from collections import deque
import six
import pandas as pd


_default_columns = (
    "steps",
    "duration",
    "action",
    "undefined",
    "converged",
    "polarized",
)


def merge(*results):
    """
    Merge two or more instances of `PolyGraphSimulation` into a single data frame.
    Optionally, write data frame to stream.
    """
    assert len(results) > 0
    assert all(isinstance(result, PolyGraphSimulation) for result in results)
    if len(results) > 1:
        # Create list of data frames
        frames = [result.frame for result in results]
        # Assert all frames have the same columns
        assert all(frames[0].columns.equals(frame.columns) for frame in frames[1:])
        # Concat all frames
        df = pd.concat(frames, ignore_index=True)  # pylint: disable=invalid-name
        result = PolyGraphSimulation.fromframe(df)
    else:
        (result,) = results
    return result


class PolyGraphSimulation:
    """
    A collection of PolyGraph simulation results
    """

    @classmethod
    def fromframe(cls, frame):
        """
        Returns a `PolyGraphSimulation` container of given data frame.
        """
        obj = cls(*frame.columns)
        obj._frame = frame  # pylint: disable=protected-access
        return obj

    def __init__(self, *cols, uid=None, **meta):
        # Column names
        if not cols:
            cols = _default_columns
        assert cols and all(isinstance(column, str) for column in cols)
        self._columns = cols
        # Collection of simulation results
        self._queue = deque()
        # Metadata associated with collection
        if meta:
            # Ensure there are no duplicate columns
            assert not any(column in meta for column in cols)
            # Validate metadata values
            assert all(
                isinstance(value, (int, float, bool, str)) for value in meta.values()
            )
        self._meta = meta
        # Experiment unique identifier
        self._uid = uid
        # Data frame of simulation results. Once results are converted
        # to a data frame, the collection becomes read-only.
        self._frame = None

    @property
    def frame(self):
        """
        Returns data frame of PolyGraph simulation results.
        """
        return self._export()

    def _export(self):
        """
        Exports collection or results to a data frame.
        """
        if self._frame is None:
            # Create data frame from collection
            self._frame = pd.DataFrame(self._queue, columns=self._columns)
            if self._meta:
                # Append metadata as new columns
                for key, value in six.iteritems(self._meta):
                    self._frame[key] = value
            if self._uid:
                # Append uuid as a new column
                self._frame["uid"] = self._uid
        return self._frame

    def add(self, *values):
        """
        Adds a PolyGraph simulation result to the current collection.
        Note that column order follows order in values.
        """
        # Do not add new results once collection has been exported
        assert self._frame is None
        # Ensure that number of values equals number of columns
        assert len(self._columns) == len(values)
        self._queue.append(values)

    def store(self, directory=None, filename=None, overwrite_ok=True):
        """
        Stores collection to disk.
        """
        if filename is None:
            destination = "data.csv"
        else:
            destination = filename
        if directory is not None:
            # Ensure that the output directory exists
            assert os.path.isdir(directory)
            # Set destination path
            destination = os.path.join(directory, destination)
        # Check for overwrites
        assert not (not overwrite_ok and os.path.exists(destination))
        # Export collection to data frame
        _ = self._export()
        # Store data frame to a csv file
        self._frame.to_csv(destination, index=False)
