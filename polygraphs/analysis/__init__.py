import os
from .graph_converter import GraphConverter, Graphs
from .belief_processor import BeliefProcessor, Beliefs
from .simulation_processor import SimulationProcessor
from .utils import *

# Cache data directory for all results
_RESULTCACHE = os.getenv("POLYGRAPHS_CACHE") or "~/polygraphs-cache/results"


class Processor(SimulationProcessor):
    """
    Processor class for performing analysis on simulation data.

    This class inherits from SimulationProcessor classes allowing it to process
    simulation data and add attributes to it.

    Keyword arguments:
    - root_folder_path (str or list): The path to the root folder containing simulation data.
    - include (dict): Dictionary specifying key-value pairs to include directories based on config.json.
    - exclude (dict): Dictionary specifying key-value pairs to exclude directories based on config.json.
    - ignore_config (bool): Check config folder location in simulation.results
    - graph_converter (Graphs, optional): An instance of Graphs class for graph conversion.
        If not provided, a new instance will be created.
    - belief_processor (Beliefs, optional): An instance of Beliefs class for belief processing.
        If not provided, a new instance will be created.

    This class initializes with the specified root folder path, along with optional
    instances of Graphs and Beliefs classes. It then processes the simulations
    in the root folder path.
    """
    normalise_gml = staticmethod(utils.normalise_gml)

    def __init__(
        self,
        root_folder_path=_RESULTCACHE,
        include=None,
        exclude=None,
        config_check=True,
        graph_converter=None,
        belief_processor=None,
    ):
        # Initialize with default Graphs and Beliefs instances if not provided
        if graph_converter is None:
            graph_converter = GraphConverter()
        if belief_processor is None:
            belief_processor = BeliefProcessor()
        # Call the constructor of parent classes with specified instances
        super().__init__(include, exclude, config_check)
        # Process simulations in the specified root folder path
        self.process_simulations(root_folder_path)
        # Objects to store loaded beliefs and graphs
        self.graphs = Graphs(self.dataframe, graph_converter)
        self.beliefs = Beliefs(self.dataframe, belief_processor, self.graphs)

    def add(self, *methods):
        """
        Decorator to add custom columns to the DataFrame.

        This method takes a variable number of methods and applies a decorator
        to each method, allowing it to be called to add custom columns to the DataFrame.
        """

        def column(func):
            def wrapper(*args, **kwargs):
                func(*args, **kwargs)

            return wrapper

        for method in methods:
            column(method)

    @property
    def sims(self):
        """Get the processed DataFrame."""
        return self.dataframe

    def get(self):
        return self.sims
