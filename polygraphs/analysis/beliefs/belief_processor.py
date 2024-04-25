import pandas as pd  # Importing pandas library for data manipulation
import h5py  # Importing h5py library for working with HDF5 files


class Beliefs:
    def get_beliefs(self, hd5_file_path, bin_file_path, graph_converter):
        # Retrieve graph object from bin file using the provided graph_converter
        graph = graph_converter.get_graph_object(bin_file_path)
        # Convert the graph to a NetworkX graph
        G = graph_converter.convert_graph_networkx(graph)

        # Open the HDF5 file in read mode
        with h5py.File(hd5_file_path, "r") as fp:
            # Extract the keys (iteration numbers) from the 'beliefs' group in the HDF5 file
            _keys = sorted(map(int, fp["beliefs"].keys()))
            # Initialize a list to store iteration number and corresponding beliefs
            iterations = [(0, graph[0].ndata["beliefs"].tolist())]

            # Iterate over each key (iteration number) in the HDF5 file
            for key in _keys:
                # Retrieve beliefs data for the current iteration
                beliefs = fp["beliefs"][str(key)]
                # Append the iteration number and beliefs data to the list
                iterations.append((key, list(beliefs)))

        # Create a MultiIndex for DataFrame indexing with iteration number and node as indices
        index = pd.MultiIndex.from_product(
            [[0, *_keys], list(G.nodes())], names=["iteration", "node"]
        )
        # Create an empty DataFrame with the defined MultiIndex
        iterations_df = pd.DataFrame(index=index, columns=["beliefs"])

        # Populate the DataFrame with beliefs data for each iteration
        for key, beliefs in iterations:
            iterations_df.loc[key, "beliefs"] = beliefs

        # Return the populated DataFrame containing beliefs data for each iteration
        return iterations_df

    def get_majority(self, iterations):
        # Calculate the average beliefs for each iteration
        average_by_iteration = iterations.groupby(level="iteration").mean()
        # Filter iterations where average belief is above 0.5 (majority belief)
        iterations_above_threshold = average_by_iteration[
            average_by_iteration["beliefs"] > 0.5
        ]
        # Return the index of the first iteration above the threshold
        return iterations_above_threshold.index.tolist()[0]
