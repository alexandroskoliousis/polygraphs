# Import necessary classes/modules
from graphs.graph_converter import Graphs  # Import Graphs class for graph conversion
from beliefs.belief_processor import (
    Beliefs,
)  # Import Beliefs class for belief processing
from attributes.add_attributes import (
    AddAttributes,
)  # Import AddAttributes class for adding attributes
from simulations.simulation_processor import (
    SimulationProcessor,
)  # Import SimulationProcessor class for simulation processing


class PolygraphAnalysis(SimulationProcessor, AddAttributes):
    """
    PolygraphAnalysis class for performing analysis on simulation data.

    This class inherits from SimulationProcessor and AddAttributes classes,
    allowing it to process simulation data and add attributes to it.

    Parameters:
    - root_folder_path (str): The path to the root folder containing simulation data.
    - graph_converter (Graphs, optional): An instance of Graphs class for graph conversion.
        If not provided, a new instance will be created.
    - belief_processor (Beliefs, optional): An instance of Beliefs class for belief processing.
        If not provided, a new instance will be created.

    This class initializes with the specified root folder path, along with optional
    instances of Graphs and Beliefs classes. It then processes the simulations
    in the root folder path.
    """

    def __init__(self, root_folder_path, graph_converter=None, belief_processor=None):
        # Initialize with default Graphs and Beliefs instances if not provided
        if graph_converter is None:
            graph_converter = Graphs()
        if belief_processor is None:
            belief_processor = Beliefs()
        # Call the constructor of parent classes with specified instances
        super().__init__(graph_converter, belief_processor)
        # Process simulations in the specified root folder path
        self.process_simulations(root_folder_path)

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
