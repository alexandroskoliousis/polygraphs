import os

from .graph_converter import Graphs
from .belief_processor import Beliefs
from .add_attributes import AddAttributes
from .simulation_processor import SimulationProcessor


# Cache data directory for all results
_RESULTCACHE = os.getenv("POLYGRAPHS_CACHE") or "~/polygraphs-cache/results"


class Processor(SimulationProcessor, AddAttributes):
    """
    Processor class for performing analysis on simulation data.

    This class inherits from SimulationProcessor and AddAttributes classes,
    allowing it to process simulation data and add attributes to it.

    Keyword arguments:
    - root_folder_path (str): The path to the root folder containing simulation data.
    - graph_converter (Graphs, optional): An instance of Graphs class for graph conversion.
        If not provided, a new instance will be created.
    - belief_processor (Beliefs, optional): An instance of Beliefs class for belief processing.
        If not provided, a new instance will be created.

    This class initializes with the specified root folder path, along with optional
    instances of Graphs and Beliefs classes. It then processes the simulations
    in the root folder path.
    """

    def __init__(
        self, root_folder_path=_RESULTCACHE, graph_converter=None, belief_processor=None
    ):
        # Initialize with default Graphs and Beliefs instances if not provided
        if graph_converter is None:
            graph_converter = Graphs()
        if belief_processor is None:
            belief_processor = Beliefs()
        # Call the constructor of parent classes with specified instances
        super().__init__(graph_converter, belief_processor)
        # Process simulations in the specified root folder path
        self.process_simulations(root_folder_path)
        # Remeber all_beliefs and all_graphs
        self.all_beliefs = []
        self.all_graphs = []

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

    def get(self):
        """
        Get the processed DataFrame.

        Returns:
        - pandas.DataFrame: The DataFrame containing processed simulation data.
        """
        return self.dataframe

    def get_beliefs(self, row=None):
        """
        Get the beliefs for all similation or for a single simualtion

        Keyword arguments:
        - row: return the beliefs for a specific simulation row from results dataframe

        Returns:
        - pandas.DataFrame: The DataFrame containing processed simulation data.
        """
        if isinstance(row, int) and row >= 0:
            bin_file_path = self.dataframe.iloc[row]["bin_file_path"]
            hd5_file_path = self.dataframe.iloc[row]["hd5_file_path"]
            return self.belief_processor.get_beliefs(
                hd5_file_path, bin_file_path, self.graph_converter
            )
        else:
            # Check if we already have beliefs for all simulations
            if len(self.all_beliefs) == len(self.dataframe):
                return self.all_beliefs

            # Loop through each bin file in dataframe and extract beliefs
            _ = []
            for hd5_file_path, bin_file_path in zip(
                self.dataframe["hd5_file_path"], self.dataframe["bin_file_path"]
            ):
                iterations = self.belief_processor.get_beliefs(
                    hd5_file_path, bin_file_path, self.graph_converter
                )
                _.append(iterations)

            self.all_beliefs = _
            return self.all_beliefs

    def get_graphs(self, row=None):
        """
        Get the graphs for all similation or for a single simualtion

        Keyword arguments:
        - row: return the graph for a specific simulation row from results dataframe

        Returns:
        - nx.Graph: A NetworkX Graph of the simulation
        """
        if isinstance(row, int) and row >= 0:
            bin_file_path = self.dataframe.iloc[row]["bin_file_path"]
            return self.graph_converter.get_networkx_object(bin_file_path)
        else:
            # Check if we already have graphs for all simulations
            if len(self.all_graphs) == len(self.dataframe):
                return self.all_graphs

            # Get graph from each simulation
            self.all_graphs = [
                self.graph_converter.get_networkx_object(bin_file_path)
                for bin_file_path in self.dataframe["bin_file_path"]
            ]

            return self.all_graphs
