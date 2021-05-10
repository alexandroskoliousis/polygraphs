"""
PolyGraph datasets
"""
import os
import sys
from collections import deque, defaultdict

# For download management
import urllib

# For gzip file management
import gzip
from io import TextIOWrapper

# For DGL graph creation
import torch
import dgl


from .logger import getlogger


log = getlogger()

# Data cache for all datasets
_DATACACHE = '~/polygraphs-cache/data'


class _ProgressBar:  # pylint: disable=too-few-public-methods
    """
    Reports download progress.
    """
    def __init__(self, slots=10):
        # Maximum number of slots
        self.slots = slots
        # Previous slot
        self.previous = -1

    def update(self, nb, bs, fs):  # pylint: disable=invalid-name
        """
        Report hook, called once on establishment of the network connection and once after
        each block read thereafter.

        The function will be passed three arguments; a count of blocks transferred so far,
        a block size in bytes, and the total size of the file.
        """
        # Check that the file size is a valid number (excl. older FTP servers)
        assert fs > 0
        # Download progress thus far, a number between 0 and 1
        progress = min(float(nb * bs) / float(fs), 1.)
        # We report progress every in fixed increments, determined by the number of slots
        slot = int(progress * self.slots)
        if slot > self.previous:
            report = '[{:10s}] {:5.1f}\n'.format('=' * slot, 100. * progress)
            sys.stdout.write(report)
            self.previous = slot


class SNAPDataset:
    """
    Base class from which all SNAP datasets are derived
    """
    def __init__(self, origin, destination, filename=None):
        # The origin URL
        self._origin = origin

        # The destination directory
        if os.path.isabs(destination):
            self._destination = destination
        else:
            # Cache data directory for SNAP datasets
            cache = os.path.join(os.path.expanduser(_DATACACHE), 'snap')
            # Set normalised path
            self._destination = os.path.normpath(os.path.join(cache, destination))

        # Check that the destination directory if free of any extensions
        # (indicating a file rather than a directory)
        _, ext = os.path.splitext(self._destination)
        assert not ext

        # Parse URL, extracting components such as scheme (e.g. 'http'),
        # location, and path. The latter is of interest
        components = urllib.parse.urlparse(self._origin)
        # A very simple case of a malformed URL
        assert components.path
        # Get origin filename from URL's path
        basename = os.path.basename(components.path)
        # Another case of a malformed URL
        assert basename

        # The destination file
        if filename:
            assert isinstance(filename, str)
            self._filename = filename
        else:
            self._filename = basename

        # The destination file, combined
        self._filepath = os.path.join(self._destination, self._filename)

    def download(self):
        """
        Downloads dataset.
        """
        if os.path.exists(self._filepath):
            # Assume that the file is already downloaded
            log.info('File \'%s\' already exists', self._filepath)
            return
        # Create directory if not exists
        if not os.path.isdir(self._destination):
            log.info('Creating directory \'%s\'', self._destination)
            os.makedirs(self._destination)

        # Pretty print download message
        components = urllib.parse.urlparse(self._origin)
        log.info('Downloading \'%s\' from %s', os.path.basename(components.path), components.netloc)

        # Create progress bar
        reporter = _ProgressBar()

        def reporthook(nb, bs, fs):  # pylint: disable=invalid-name
            reporter.update(nb, bs, fs)

        # Try download
        try:
            urllib.request.urlretrieve(self._origin, self._filepath, reporthook=reporthook)
        except KeyboardInterrupt:
            # Try remove downloaded file
            if os.path.exists(self._filepath):
                os.remove(self._filepath)
            raise
        log.info('Download complete')

    @staticmethod
    def read_(txt):
        """
        Returns graph edges as two lists, one for source and one for destination nodes.
        """
        assert isinstance(txt, TextIOWrapper)
        # Lists of source-destination pairs
        src = deque()
        dst = deque()
        for line in txt:
            # Ignore comments
            if line.startswith('#'):
                continue
            # Each line has two numbers,
            # the source and destination
            # node id
            u, v = list(map(int, line.split()))  # pylint: disable=invalid-name
            src.append(u)
            dst.append(v)
        # Normalise node identifiers (from 0 to N)
        tbl = defaultdict(lambda: len(tbl))
        src = [tbl[node] for node in src]
        dst = [tbl[node] for node in dst]
        # Create DGL graph from edges
        edges = torch.Tensor(src).to(torch.int64), torch.Tensor(dst).to(torch.int64)
        return dgl.graph(edges)

    def load(self):
        """
        Reads dataset into memory as DGL graph.
        """
        # Try download file
        self.download()
        # Check if downloaded file is a valid tar archive
        # assert tarfile.is_tarfile(self._filepath)

        with gzip.open(self._filepath, mode='rt') as txt:
            graph = type(self).read_(txt)
        return graph
