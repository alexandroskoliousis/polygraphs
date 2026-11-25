"""
A collection of metadata associated with PolyGraph simulations
"""
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SimulationRecord:
    steps: int
    duration: float
    action: str
    undefined: bool
    converged: bool


def _metadata_columns(columns, uid, meta):
    """
    Returns {name: value} for the columns that are constant across a
    collection (metadata and `uid`), after validating them against the
    record's own `columns`.
    """
    for key, value in meta.items():
        if not isinstance(value, (int, float, bool, str)):
            raise TypeError(f"metadata {key!r} must be int, float, bool or str")

    metadata = dict(meta)
    if uid is not None:
        metadata["uid"] = uid

    # A constant column would silently overwrite a real results column
    if clash := set(columns) & set(metadata):
        raise ValueError(f"names clash with record fields: {sorted(clash)}")
    return metadata


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

    def __init__(self, record=SimulationRecord, *, uid=None, **meta):
        self._record = record
        # Column field names returned for each sim from op
        self._columns = tuple(f.name for f in fields(record))

        # Metadata columns such as uid
        self._metadata = _metadata_columns(self._columns, uid, meta)
        self._results = []
        self._frame = None  # cache

    def __len__(self):
        return len(self._results)

    def __iter__(self):
        return iter(self._results)        

    @property
    def record(self):
        """The dataclass type of the results in this collection."""
        return self._record

    def add(self, result):
        """Adds a single result, which must be an instance of `record`."""
        if not isinstance(result, self._record):
            raise TypeError(
                f"expected {self._record.__name__}, got {type(result).__name__}"
            )
        self._results.append(result)
        self._frame = None

    def extend(self, results):
        """Adds every result in an iterable."""
        for result in results:
            self.add(result)        

    @property
    def frame(self):
        """Data frame of results, with metadata appended as columns."""
        if self._frame is None:
            df = pd.DataFrame(
                [asdict(r) for r in self._results], columns=list(self._columns)
            )
            for key, value in self._metadata.items():
                df[key] = value
            self._frame = df
        return self._frame

    def store(self, directory, filename="data.csv", *, overwrite=True):
        """Stores the collection to a csv file."""
        path = Path(directory, filename)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        self.frame.to_csv(path, index=False)


def merge(*simulations):
    """
    Merges two or more `PolyGraphSimulation` instances into a single data frame.
    All simulations must produce the same columns.
    """
    if not simulations:
        raise ValueError("nothing to merge")
    frames = [sim.frame for sim in simulations]
    if not all(frames[0].columns.equals(f.columns) for f in frames[1:]):
        raise ValueError("simulations have different columns")
    return pd.concat(frames, ignore_index=True)
