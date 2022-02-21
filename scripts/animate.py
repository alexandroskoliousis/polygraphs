import sys
import os
import argparse
import collections

import dgl
import h5py

from polygraphs import visualisations as viz


def _argparse(argv):
    return argv


def _getid(directory):
    return directory


def _load(directory, id):
    # Load graph(s)
    graphs, _ = dgl.load_graphs(os.path.join(directory, f"{id}.bin"))
    graph = graphs[0]
    # Load beliefs
    fp = h5py.File(os.path.join(directory, f"{id}.hd5"), "r")
    _keys = [int(key) for key in fp.keys()]
    _keys = sorted(_keys)
    frames = collections.deque()
    frames.append(graph.ndata['beliefs'].numpy())
    for key in _keys:
        frames.append(fp[str(key)][:])
    return graph, frames


if __name__ == "__main__":
    # Parse command-line arguments
    args = _argparse(sys.argv)
    # Ensure result directory exists
    assert os.path.isdir(args.directory)
    # Extract id(s)
    identifiers = _getid(args.directory)
    for id in identifiers:
        # Load graphs
        graph, frames = _load(args.directory, id)
        # Animate
        _ = viz.animate(graph,
                        frames,
                        filename="test.mp4",
                        layout='circular')
    print('Bye.')
