"""
Hyper-parameter settings for PolyGraph simulations
"""
import os
import itertools
import copy
import json
import yaml
import six


class HyperParameters:
    """
    Hyper-parameter settings
    """

    def __init__(self, **kwargs):
        self.ht = kwargs  # pylint: disable=invalid-name
        self.__dict__.update(self.ht)

    def __repr__(self):
        body = json.dumps(self.ht, default=lambda x: x.ht, indent=4)
        return "{}({})".format(type(self).__name__, body)

    def __setattr__(self, name, value):
        if name == "ht":
            super().__setattr__(name, value)
        else:
            self.update(**{name: value})

    def getattr(self, name):
        """
        Something like __getattr__...
        """
        head, *tail = name.split(".", 1)
        if not hasattr(self, head):
            raise AttributeError("Attribute not found: {}".format(head))
        value = self.ht.get(head)
        if isinstance(value, HyperParameters):
            # There must be at least another level to explore
            return value.getattr(*tail)
        return value

    def __contains__(self, key):
        return key in self.ht

    @staticmethod
    def _unflatten(source, destination, separator):
        for name, value in source.items():
            *prefix, key = name.split(separator)
            target = destination
            for part in prefix:
                if part in target:
                    target = target[part]
                else:
                    target[part] = {}
                    target = target[part]
            if isinstance(value, dict):
                if key not in target:
                    target[key] = {}
                    HyperParameters._unflatten(value, target[key], separator)
            else:
                if key in target:
                    raise ValueError("Duplicate key found: {}".format(key))
                target[key] = value
        return destination

    @staticmethod
    def unflatten(data, separator="."):
        """
        Unflattens a '.'-structured hyper-parameter setting (e.g., `a.b.c`).
        """
        return HyperParameters._unflatten(data, {}, separator)

    def _isvalid(self, value):
        """
        Checks if provided value has a valid data type.
        """
        if value is None:
            return True
        # Tuple of supported types
        types = (int, float, bool, str, list, dict, HyperParameters)
        if not isinstance(value, types):
            return False
        # If the value is a dictionary, check:
        # a) All keys are strings
        # b) All values are valid
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value.keys()):
                return False
            if not all(self._isvalid(v) for v in value.values()):
                return False
        # If the value is a list, check all values are valid
        if isinstance(value, list):
            if not all(self._isvalid(v) for v in value):
                return False
        return True

    def _update(self, name, value):
        """
        Updates attribute with given name with the provided value.
        """
        if not hasattr(self, name):
            raise AttributeError("Attribute not found: {}".format(name))
        if not self._isvalid(value):
            raise TypeError("Invalid value type: {}".format(value.__class__.__name__))
        self.ht[name] = value
        self.__dict__.update(self.ht)

    def _write_to(self, directory, filename, suffix, exists_ok=False):
        """
        Returns destination file name.
        """
        if filename is None:
            destination = "{}.{}".format(self.__class__.__name__.lower(), suffix)
        else:
            destination = filename
        if directory is not None:
            # If directory does not exist, create it
            if not os.path.isdir(directory):
                os.makedirs(directory)
            destination = os.path.join(directory, destination)
        # Check if destination path already exists
        if os.path.exists(destination) and not exists_ok:
            raise Exception("File already exists: {}".format(destination))
        return destination

    @classmethod
    def _merge(cls, dst, src):
        # pylint: disable=protected-access
        if isinstance(dst, HyperParameters):
            if src is None:
                return dst
            if isinstance(src, dict):
                for key, value in src.items():
                    dst._update(key, cls._merge(dst.ht[key], value))
                return dst
            raise ValueError(src)
        if isinstance(dst, dict):
            if src is None:
                return dst
            if isinstance(src, dict):
                for key, value in src.items():
                    dst[key] = cls._merge(dst.get(key), value)
                return dst
            raise ValueError(src)
        return src

    def toYAML(self, directory=None, filename=None):  # pylint: disable=invalid-name
        """
        Writes hyper-parameters to a .yaml file.
        """
        filename = self._write_to(directory, filename, "yaml")
        with open(filename, "w") as fstream:
            yaml.dump(self.ht, fstream, Dumper=yaml.CDumper)
        return filename

    def toJSON(
        self, directory=None, filename=None, exists_ok=False
    ):  # pylint: disable=invalid-name
        """
        Writes hyper-parameters to a .json file.
        """
        filename = self._write_to(directory, filename, "json", exists_ok=exists_ok)
        with open(filename, "w") as fstream:
            json.dump(self.ht, fstream, default=lambda x: x.ht, indent=4)
        return filename

    @classmethod
    def fromYAML(cls, filename, dest=None):  # pylint: disable=invalid-name
        """
        Reads hyper-parameters from a .yaml file.
        """
        with open(filename, "r") as fstream:
            try:
                # yaml.CLoader may raise an AttributeError
                data = yaml.load(fstream, Loader=yaml.CLoader)
            except AttributeError:
                data = yaml.load(fstream, Loader=yaml.Loader)
            data = HyperParameters.unflatten(data, separator=".")
        if not dest:
            dest = cls()
        return cls._merge(dest, data)

    @classmethod
    def fromJSON(cls, filename, dest=None):  # pylint: disable=invalid-name
        """
        Reads hyper-parameters from a .json file.
        """
        with open(filename, "r") as fstream:
            data = json.load(fstream)
        if not dest:
            dest = cls()
        return cls._merge(dest, data)

    @classmethod
    def fromJSON_(cls, fstream):
        data = json.load(fstream)
        dest = cls()
        return cls._merge(dest, data)

    @classmethod
    def load(cls, filenames):
        """
        Reads configuration from one or more files
        """
        params = None
        for filename in filenames:
            # Check file exists
            if not os.path.exists(filename):
                raise Exception("File not found: {}".format(filename))
            _, ext = os.path.splitext(filename)
            if ext in [".yaml", ".yml"]:
                params = cls.fromYAML(filename, params)
            elif ext in [".json"]:
                params = cls.fromJSON(filename, params)
            else:
                raise Exception("Invalid file name: {}".format(filename))
        return params

    def add(self, **kwargs):
        """
        Adds one or more attributes.
        """
        for name, value in six.iteritems(kwargs):
            if hasattr(self, name):
                raise AttributeError("Attribute already exists: {}".format(name))
            if not self._isvalid(value):
                raise TypeError(
                    "Invalid value type: {}".format(value.__class__.__name__)
                )
            self.ht[name] = value
            self.__dict__.update(self.ht)

    def update(self, **kwargs):
        """
        Updates the value of one or more attributes.
        """
        for name, value in six.iteritems(kwargs):
            if not hasattr(self, name):
                raise ValueError
            if self.ht[name] is None:
                if not self._isvalid(value):
                    raise TypeError
                self.ht[name] = value
            else:
                if isinstance(self.ht[name], HyperParameters):
                    self.ht[name] = value
                else:
                    if value is None:
                        self.ht[name] = value
                    else:
                        self.ht[name] = type(self.ht[name])(value)
            self.__dict__.update(self.ht)

    def delete(self, name):
        """
        Deletes attribute of given name.
        """
        if not hasattr(self, name):
            return
        del self.ht[name]
        delattr(self, name)

    def keys(self):
        """
        Returns list of attributes.
        """
        return self.ht.keys()

    @classmethod
    def expand(cls, params, options):
        """
        Expands a configuration given options.
        """
        # Copy an instance of hyper-parameters
        assert isinstance(params, cls)
        # Options are a dictionary
        assert isinstance(options, dict)
        # All values in the dictionary are lists (more general, iterable)
        assert all(isinstance(v, list) for v in options.values())
        # Generate combinations for each key-value pair
        lst = list(itertools.product((key,), value) for key, value in options.items())
        configurations = []
        for combination in itertools.product(*lst):
            dest = copy.deepcopy(params)
            data = cls.unflatten(dict(combination), separator=".")
            dest = cls._merge(dest, data)
            configurations.append(dest)
        return configurations


class LoggingHyperParameters(HyperParameters):
    """
    Configuration parameters include:

        params.enabled
        params.interval
    """

    def __init__(self):
        super().__init__()
        self.add(enabled=True)
        self.add(interval=1)


class SnapshotHyperParameters(HyperParameters):
    """
    Configuration parameters include:

        params.enabled
        params.interval
        params.messages
    """

    def __init__(self):
        super().__init__()
        self.add(enabled=False)
        self.add(interval=1)
        self.add(messages=False)


class NetworkHyperParameters(HyperParameters):
    """
    Configuration parameters include:

        params.kind
        params.size
        params.directed
        params.selfloop

        params.random.seed
        params.random.probability

        params.wattsstrogatz.knn
        params.wattsstrogatz.seed
        params.wattsstrogatz.tries
        params.wattsstrogatz.probability

        params.barabasialbert.attachments
        params.barabasialbert.seed

        params.snap.name
        params.snap.extras

        params.ogb.name
        params.ogb.extras
    """

    def __init__(self):
        super().__init__()
        self.add(kind=None)
        self.add(size=None)
        self.add(directed=False)
        # Whether to connect each vertex to itself or not
        self.add(selfloop=True)
        # Network-specific configurations
        self.add(random=HyperParameters(seed=None, tries=100, probability=1.0))
        self.add(
            wattsstrogatz=HyperParameters(knn=2, seed=None, tries=100, probability=1.0)
        )
        self.add(barabasialbert=HyperParameters(attachments=1, seed=None))
        # Adding support for datasets
        self.add(snap=HyperParameters(name=None))
        self.add(ogb=HyperParameters(name="collab"))
        self.add(gml=HyperParameters(name=None, path=None, directed=False))


class InitHyperParameters(HyperParameters):
    """
    Configuration parameters include:

        params.kind
        params.value
    """

    def __init__(self):
        super().__init__()
        self.add(kind="uniform")
        # Uniform initializer
        self.add(uniform=HyperParameters(lower=0.0, upper=1.0))
        # Gaussian initializer
        self.add(
            gaussian=HyperParameters(
                mean=0.0, std=1.0, lower=-2.0, upper=2.0, attempts=4
            )
        )
        # Constant initializer
        self.add(constant=HyperParameters(value=None))
        # Specific node beliefs after initialisation
        self.add(beliefs=HyperParameters())


class SimulationHyperParameters(HyperParameters):
    """
    Configuration parameters include:

        params.results
        params.repeats
        params.steps
    """

    def __init__(self):
        super().__init__()
        self.add(results="auto")
        self.add(repeats=1)
        self.add(steps=0)


class PolyGraphHyperParameters(HyperParameters):
    """
    Configuration parameters include:

        params.device
        params.op
        params.seed
        params.epsilon
        params.trials
        params.lowerupper
        params.upperlower

        params.mistrust
        params.antiupdating

        params.init.kind
        params.init.value

        params.network.kind
        params.network.size
        params.network.directed
        params.network.selfloop
        params.network.random.*
        params.network.wattsstrogatz.*
        params.network.snap.*
        params.network.ogb.*

        params.logging.enabled
        params.logging.interval

        params.simulation.results
        params.simulation.repeats
        params.simulation.steps
    """

    def __init__(self):
        super().__init__()

        # Target device
        self.add(device="cpu")

        # Operator name
        self.add(op=None)

        # Parameter related to randomness
        self.add(seed=0)

        # Parameters related to learning from neighbours
        self.add(epsilon=0.0)
        self.add(trials=10)

        # Parameters related to convergence (upper and lower bounds for beliefs)
        self.add(lowerupper=0.99)
        self.add(upperlower=0.50)

        # Parameters related to polarisation
        self.add(mistrust=0.0)
        self.add(antiupdating=False)

        # Parameters related to testimonials and epistemic injustice
        self.add(reliability=1.0)
        self.add(trust=0.0)
        self.add(unreliablenodes=[])

        # Parameters related to belief initilisation
        self.add(init=InitHyperParameters())
        # Logging configuration
        self.add(logging=LoggingHyperParameters())
        # Snapshot configuration
        self.add(snapshots=SnapshotHyperParameters())
        # Network properties (e.g. size, type)
        self.add(network=NetworkHyperParameters())
        # Metadata configuration
        self.add(simulation=SimulationHyperParameters())
