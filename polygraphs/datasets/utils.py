"""
PolyGraph dataset utils
"""
import os
import sys

# For downloads
import urllib

# For unzip
import zipfile

# For copy
import shutil

# Import timer
from ..timer import Timer

# Import logger
from ..logger import getlogger

log = getlogger()


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
        progress = min(float(nb * bs) / float(fs), 1.0)
        # We report progress every in fixed increments, determined by the number of slots
        slot = int(progress * self.slots)
        if slot > self.previous:
            report = "[{:10s}] {:5.1f}\n".format("=" * slot, 100.0 * progress)
            sys.stdout.write(report)
            self.previous = slot


def download(url, filename):
    """
    Downloads a remote file, denoted by given URL, to a local file.
    """
    if os.path.isfile(filename):
        # File already exists; do not download
        log.info("File '%s' already exists", filename)
        return

    # Pretty print download message
    components = urllib.parse.urlparse(url)
    log.info(
        "Downloading '%s' from %s", os.path.basename(components.path), components.netloc
    )

    # Create rudimentary progress bar (and report hook)
    reporter = _ProgressBar()

    def reporthook(nb, bs, fs):  # pylint: disable=invalid-name
        reporter.update(nb, bs, fs)

    clock = Timer()
    clock.start()
    # Try download
    try:
        urllib.request.urlretrieve(url, filename, reporthook=reporthook)
    except KeyboardInterrupt:
        # Try remove downloaded file
        if os.path.exists(filename):
            os.remove(filename)
        raise
    log.info("Download complete (%.2fs)", clock.dt())


def unzip(filename, folder=None):
    """
    Extracts contents of ZIP archive to folder.
    """
    # Check that filename is a valid ZIP file
    assert zipfile.is_zipfile(filename)

    # Get [directory]/[file].zip structure
    head, tail = os.path.split(filename)

    if not folder:
        # Extract zip archive in place
        folder = head

    # Create destination folder, if not exists
    if not os.path.isdir(folder):
        log.info("Creating directory %s", folder)
        os.makedirs(folder)

    # Pretty print unzip message
    log.info("Extracting '%s'", tail)

    # Extract contents to destination
    clock = Timer()
    clock.start()
    with zipfile.ZipFile(filename, "r") as ctx:
        ctx.extractall(folder)
    log.info("Unzip complete (%.2fs)", clock.dt())


def copy(source, destination):
    """
    Copy source file to destination file locally.
    """
    # Check that origin is a valid file
    assert os.path.isfile(source)
    # Create master's folder, if not exists
    head, _ = os.path.split(destination)
    if head:
        if not os.path.isdir(head):
            os.makedirs(head)
    # Copy
    shutil.copy(source, destination)
