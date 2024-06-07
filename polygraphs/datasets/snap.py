"""
PolyGraph SNAP datasets
"""

from collections import deque, defaultdict
import gzip
from urllib.parse import urljoin
import numpy as np
import torch
import dgl

from .dataset import PolyGraphDataset


_SNAP = "https://snap.stanford.edu"


class SNAPDataset(PolyGraphDataset):
    """
    SNAP dataset
    """

    def __init__(self, folder, directed=True, edges=None, **extra):

        # Lookup table for normalising node identifier to 0 to N
        self.tbl = defaultdict(lambda: len(self.tbl))

        super().__init__(folder=folder, directed=directed, edges=edges, **extra)

    @property
    def collection(self):
        return "snap"

    def __read_edges(self):
        # pylint: disable=no-member
        """
        Reads edges (u, v) from dataset file and returns two lists,
        U and V for source and destination nodes, respectively.
        """
        src, dst = deque(), deque()

        # Read gzip file as txt
        with gzip.open(
            self.edges.origin, mode="rt"
        ) as txt:  # pylint: disable=no-member
            for line in txt:
                # Ignore comments
                if line.startswith("#"):
                    continue
                # Each line has two numbers, the source (u) and destination node
                # id (v), we catch timestamps for temporal graphs with t
                u, v, *t = list(map(int, line.split()))  # pylint: disable=invalid-name
                src.append(u)
                dst.append(v)

        # Normalise node identifiers (from 0 to N)
        src = [self.tbl[node] for node in src]
        dst = [self.tbl[node] for node in dst]

        return src, dst

    def read(self):
        """
        Reads dataset into memory as DGL graph.
        """
        # Fetch all dataset files
        self.fetchall()

        # Create two symmetric lists for source and destination nodes, respectively,
        # representing edges from src[i] to dst[i]
        src, dst = self.__read_edges()

        # Create DGL graph from edges
        return dgl.graph(
            (torch.Tensor(src).to(torch.int64), torch.Tensor(dst).to(torch.int64))
        )


class Twitter2010(SNAPDataset):
    """
    Twitter follower network from 'snap.stanford.edu/data/twitter-2010.html'

    Basic dataset statistics:

        Nodes:    41,652,230
        Edges: 1,468,364,884

    Other information:

        - The network is directed.
    """

    def __init__(self):
        super().__init__(
            folder="twitter-2010", edges=urljoin(_SNAP, "data/twitter-2010.txt.gz")
        )


class EgoTwitter(SNAPDataset):
    """
    Social circles: Twitter from 'snap.stanford.edu/data/ego-Twitter.html'

    Basic dataset statistics:

        Nodes:    81,306
        Edges: 1,768,149

    Other information:

        - The network is directed.
    """

    def __init__(self):
        super().__init__(
            folder="ego-twitter", edges=urljoin(_SNAP, "data/twitter_combined.txt.gz")
        )


class EgoFacebook(SNAPDataset):
    """
    Social circles: Facebook from 'snap.stanford.edu/data/ego-Facebook.html'

    Basic dataset statistics:

        Nodes:  4,039
        Edges: 88,234

    Other information:

        - The network is undirected.
    """

    def __init__(self):
        super().__init__(
            folder="ego-facebook",
            directed=False,
            edges=urljoin(_SNAP, "data/facebook_combined.txt.gz"),
        )


class LiveJournal1(SNAPDataset):
    """
    LiveJournal social network from
    'https://snap.stanford.edu/data/soc-LiveJournal1.html'

    Basic dataset statistics:

        Nodes:  4,847,571
        Edges: 68,993,773

    Other information:

        - The network is directed.

    """

    def __init__(self):
        super().__init__(
            folder="soc-livejournal",
            edges=urljoin(_SNAP, "data/soc-LiveJournal1.txt.gz"),
        )


class EmailEUCore(SNAPDataset):
    """
    email-Eu-core network containing only links within the
    insitution from 'snap.stanford.edu/data/email-Eu-core.html'

    Basic dataset statistics:

        Nodes:  1,005
        Edges: 25,571

    Other information:

        - The network is directed.
    """

    def __init__(self):
        super().__init__(
            folder="email-eu-core",
            edges=urljoin(_SNAP, "data/email-Eu-core.txt.gz"),
        )


class EmailEUAll(SNAPDataset):
    """
    EU email communication network from
    'snap.stanford.edu/data/email-EuAll.html'

    Basic dataset statistics:

        Nodes: 265,214
        Edges: 420,045

    Other information:

        - The network is directed.
    """

    def __init__(self):
        super().__init__(
            folder="email-eu-all",
            edges=urljoin(_SNAP, "data/email-EuAll.txt.gz"),
        )


class CollegeMsg(SNAPDataset):
    """
    Messages on a Facebook-like platform at UC-Irvine
    'snap.stanford.edu/data/CollegeMsg.html'

    Basic dataset statistics:

        Nodes:  1,899
        Edges: 20,296

    Other information:

        - The network is directed.
    """

    def __init__(self):
        super().__init__(
            folder="college-msg",
            edges=urljoin(_SNAP, "data/CollegeMsg.txt.gz"),
        )


class LiveJournal(SNAPDataset):
    """
    LiveJournal social network and ground-truth communities from
    'https://snap.stanford.edu/data/com-LiveJournal.html'

    Basic dataset statistics:

        Nodes:  3,997,962
        Edges: 34,681,189

    Other information:

        - The network is undirected.
    """

    def __init__(self):
        # pylint: disable=line-too-long
        super().__init__(
            folder="com-livejournal",
            directed=False,
            edges=urljoin(_SNAP, "data/bigdata/communities/com-lj.ungraph.txt.gz"),
            top5K=urljoin(_SNAP, "data/bigdata/communities/com-lj.top5000.cmty.txt.gz"),
        )

    def read(self):
        # pylint: disable=no-member
        # Generate graph
        graph = super().read()

        # A sparse matrix of N rows and 5,000 columns, indicating
        # whether a node (i-th row) belongs to a community or not
        # (j-th column)
        group = np.zeros((graph.num_nodes(), 5000), dtype=np.ubyte)

        # Read top-5000 communities
        with gzip.open(self.top5K.origin, mode="rt") as txt:
            for j, line in enumerate(txt):
                # Ignore comments
                if line.startswith("#"):
                    continue
                # Each line contains a list of numbers, indicating which nodes
                # belong to the j-th community
                community = list(map(int, line.split()))
                for node in community:
                    # All node ids should be present in the translation table
                    # assert node in self.tbl
                    # Get canonical index
                    i = self.tbl[node]
                    # Each entry should be unique (thus set once)
                    # assert not group[i][j]
                    group[i][j] = 1

        # Set node features
        graph.ndata["group"] = torch.from_numpy(group)

        return graph


def getbyname(name):
    """
    Returns SNAP dataset by name.
    """
    assert name and isinstance(name, str)

    def _find():
        for obj in SNAPDataset.__subclasses__():
            if name.lower() == obj.__name__.lower():
                return obj
        return None

    datasetcls = _find()
    if datasetcls is None:
        raise Exception(f"Invalid SNAP dataset name: {name}")
    return datasetcls()
