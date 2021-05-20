"""
PolyGraph datasets and dataset files
"""
import os
import abc
import urllib
import six

from . import utils as datautils

# Cache data directory for all datasets
_DATACACHE = "~/polygraphs-cache/data"


class PolyGraphDatasetFile:
    """
    A dataset file (either remote, local, or a local copy)
    """

    def __init__(self, origin):
        # File origin must be a string
        assert isinstance(origin, str)

        self._origin = origin
        self._remote = bool(urllib.parse.urlparse(origin).scheme)

        # Validate origin
        if self._remote:
            # Ensure origin is a valid remote location
            try:
                ctx = urllib.request.urlopen(origin)
                ctx.close()
            except urllib.error.URLError as err:
                raise Exception("Invalid file origin: {}".format(origin)) from err
        else:
            # Ensure origin is a valid file
            if not os.path.isfile(origin):
                raise Exception("Invalid file origin: {}".format(origin))

    @property
    def origin(self):
        """
        Returns (possibly remote) origin.
        """
        return self._origin

    @property
    def remote(self):
        """
        Returns whether origin is remote.
        """
        return self._remote

    @property
    def local(self):
        """
        Returns whether origin is local.
        """
        return not self._remote

    def fetch(self, folder):
        """
        Downloads remote origin to local folder.
        """
        if self.local:
            # Assume that the file has already been fetch
            return
        # Ensure local folder exists
        assert os.path.isdir(folder)
        # Construct destination file
        filename = os.path.join(
            folder, os.path.basename(urllib.parse.urlparse(self.origin).path)
        )
        # Maybe download file
        datautils.download(self.origin, filename)
        # File no longer considered remote
        self._origin = filename
        self._remote = False
        return


class PolyGraphDataset(metaclass=abc.ABCMeta):
    """
    Base class from which all datasets are derived
    """

    def __init__(self, folder=None, directed=True, **kwargs):

        # Ensure local folder is set and it does not start with a tilde
        folder = os.path.expanduser(folder or ".")

        # Expand local folder, if path is relative
        if not os.path.isabs(folder):
            # Default cache for dataset collection
            cache = os.path.join(os.path.expanduser(_DATACACHE), self.collection)
            # Normalise path
            folder = os.path.normpath(os.path.join(cache, folder))

        # Create dataset folder, if not exists
        os.makedirs(folder, exist_ok=True)

        # Set dataset folder
        self.folder = folder

        # Set network property (directed or not)
        self.directed = directed

        # The rest of the keyword argument are named dataset files;
        # let's parse them
        self.files = {}

        for name, value in six.iteritems(kwargs):
            # Value must be a string
            assert value and isinstance(value, str)
            # Name must not correspond to an attribute (e.g. 'self.folder' or 'self.files')
            assert not hasattr(self, name)
            # Add (or update) dataset file
            self.files[name] = PolyGraphDatasetFile(value)

        # Make named datasets accessible with the dot notation
        self.__dict__.update(self.files)

    def fetchall(self):
        """
        Downloads all dataset files.
        """
        for _, value in six.iteritems(self.files):
            value.fetch(self.folder)

    @abc.abstractproperty
    def collection(self):
        """
        Returns collection to which dataset belongs (e.g. 'snap' or 'ogb').
        """
        return None

    @abc.abstractmethod
    def read(self):
        """
        Reads dataset into memory as a DGL graph.
        """
        raise NotImplementedError
