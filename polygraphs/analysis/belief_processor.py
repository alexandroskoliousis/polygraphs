import pandas as pd  # Importing pandas library for data manipulation
import h5py  # Importing h5py library for working with HDF5 files


class BeliefProcessor:
    def get_beliefs(self, hd5_file_path, graph):
        # Open the HDF5 file in read mode
        with h5py.File(hd5_file_path, "r") as fp:
            # Extract the keys (iteration numbers) from the 'beliefs' group in the HDF5 file
            _keys = sorted(map(int, fp["beliefs"].keys()))
            # Initialize a list to store iteration number and corresponding beliefs
            # with the initial beliefs from the .bin file graph
            iterations = [(0, graph.pg["ndata"]["beliefs"].tolist())]

            # Iterate over each key (iteration number) in the HDF5 file
            for key in _keys:
                # Retrieve beliefs data for the current iteration
                beliefs = fp["beliefs"][str(key)]

                # Append the iteration number and beliefs data to the list
                iterations.append((key, list(beliefs)))

        # Create a MultiIndex for DataFrame indexing with iteration number and node as indices
        index = pd.MultiIndex.from_product(
            [[0, *_keys], list(graph.nodes)], names=["iteration", "node"]
        )

        # Create an empty DataFrame with the defined MultiIndex
        iterations_df = pd.DataFrame(index=index, columns=["beliefs"], dtype="Float32")

        # Populate the DataFrame with beliefs data for each iteration
        for key, beliefs in iterations:
            iterations_df.loc[key, "beliefs"] = beliefs

        # Return the populated DataFrame containing beliefs data for each iteration
        return iterations_df


class Beliefs:
    """
    The Beliefs class stores the beliefs of simulations that have been
    explicitly loaded for analysis using the Belief Processor

    This class provides an iterator and get item to access beliefs
    """

    def __init__(self, dataframe, belief_processor, graphs):
        self.hd5_file_path = dataframe["hd5_file_path"]
        self.belief_processor = belief_processor
        self.graphs = graphs
        self.beliefs = [None] * len(dataframe)
        self.index = 0

    def __len__(self):
        return len(self.beliefs)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.get(i) for i in range(*index.indices(len(self)))]
        return self.get(index)

    def __iter__(self):
        for i in range(len(self)):
            yield self.get(i)

    def get(self, index):
        """Return the beliefs dataframe for a simulation, loading it if needed."""
        try:
            cached = self.beliefs[index]
        except IndexError:
            raise IndexError("Simulation index out of range") from None

        if cached is None:
            cached = self.belief_processor.get_beliefs(
                self.hd5_file_path[index],
                self.graphs[index],
            )
            self.beliefs[index] = cached
        return cached
