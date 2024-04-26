import os  # Importing os module for interacting with the operating system
import pandas as pd  # Importing pandas library for data manipulation
import re  # Importing re module for regular expressions
import json  # Importing json module for working with JSON data


class SimulationProcessor:
    def __init__(self, graph_converter, belief_processor):
        # Initialize SimulationProcessor with required objects and variables
        self.path = ""  # Path to the root folder of simulation data
        self.dataframe = None  # DataFrame to store processed simulation data
        self.graph_converter = graph_converter  # Object to convert graphs
        self.belief_processor = belief_processor  # Object to process beliefs

    def _is_valid_uuid_folder(self, dirname):
        # Check if the folder name is a valid UUID format
        if len(dirname) == 32 and all(c in "0123456789abcdefABCDEF" for c in dirname):
            return True
        return False

    def extract_params(self, config_json_path):
        # Extract relevant parameters from a configuration JSON file
        with open(config_json_path, "r") as f:
            config_data = json.load(f)
        return (
            config_data.get("trials"),
            config_data.get("network", {}).get("size"),
            config_data.get("network", {}).get("kind"),
            config_data.get("op"),
            config_data.get("epsilon"),
        )

    def process_simulations(self, path):
        """
        Process simulation data from the specified path.

        Parameters:
        - path (str): The path to the root folder containing simulation data.

        Returns:
        - None

        This method walks through the directory tree starting from the specified path,
        identifies subfolders that represent individual simulation runs (identified by
        valid UUID folder names), processes each subfolder using the `process_subfolder`
        method, and aggregates the results into a single DataFrame stored in `self.dataframe`.
        """
        if path:
            # Expand the path to handle user home directory (~)
            self.path = os.path.expanduser(path)

        # Initialize a list to store paths of subfolders representing individual simulations
        uuid_folders = []
        try:
            # Walk through the directory tree starting from the specified path
            for dirpath, dirnames, filenames in os.walk(self.path):
                for dirname in dirnames:
                    # Construct the full path of each subfolder
                    folder_path = os.path.join(dirpath, dirname)
                    # Check if the folder name represents a valid UUID
                    if self._is_valid_uuid_folder(dirname):
                        # If it's a valid UUID folder, add its path to the list
                        uuid_folders.append(folder_path)
        except (FileNotFoundError, PermissionError) as e:
            # Handle exceptions if there are issues accessing folders
            print(f"Error accessing folder: {e}")

        # Initialize an empty DataFrame to store processed simulation data
        result_df = pd.DataFrame()

        # Process each subfolder and concatenate the results into the result DataFrame
        for folder in uuid_folders:
            subfolder_df = self.process_subfolder(folder)
            result_df = pd.concat([result_df, subfolder_df], ignore_index=True)

        # Store the aggregated DataFrame in the class attribute `self.dataframe`
        self.dataframe = result_df

    def process_subfolder(self, subfolder_path):
        """
        Process each subfolder in the root folder.

        Parameters:
        - subfolder_path (str): The path to the subfolder to be processed.

        Returns:
        - pandas.DataFrame: DataFrame containing processed data from the subfolder.

        This method is responsible for processing each subfolder in the root folder of
        simulation data. It extracts relevant information such as paths to binary and HDF5
        files, configuration JSON file, and optional CSV file. It then constructs a DataFrame
        containing this information for further analysis.
        """
        # Get a list of files in the subfolder
        files = os.listdir(subfolder_path)
        # Filter binary files and sort them based on their numerical order
        bin_files = sorted(
            [f for f in files if f.endswith(".bin")],
            key=lambda x: int(re.search(r"(\d+)\.bin", x).group(1)),
        )
        # Filter HDF5 files and sort them based on their numerical order
        hd5_files = sorted(
            [f for f in files if f.endswith(".hd5")],
            key=lambda x: int(re.search(r"(\d+)\.hd5", x).group(1)),
        )
        # Check if there is a configuration JSON file in the subfolder
        config_file = [f for f in files if f == "configuration.json"]
        # Check if there is a CSV file in the subfolder
        csv_file = [f for f in files if f.endswith(".csv")]

        # Initialize an empty DataFrame to store processed data
        df = pd.DataFrame()
        # Add paths to binary files to the DataFrame
        df["bin_file_path"] = [os.path.join(subfolder_path, f) for f in bin_files]
        # Add paths to HDF5 files to the DataFrame
        df["hd5_file_path"] = [os.path.join(subfolder_path, f) for f in hd5_files]

        # Process configuration JSON file if it exists
        if config_file:
            config_json_path = os.path.join(subfolder_path, config_file[0])
            df["config_json_path"] = config_json_path
            # Extract parameters from the configuration JSON file
            trials, network_size, network_kind, op, epsilon = self.extract_params(
                config_json_path
            )
            df["trials"] = trials
            df["network_size"] = network_size
            df["network_kind"] = network_kind
            df["op"] = op
            df["epsilon"] = epsilon
        else:
            # If configuration JSON file doesn't exist, set the corresponding column to None
            df["config_json_path"] = None

        # Process CSV file if it exists
        if csv_file:
            csv_path = os.path.join(subfolder_path, csv_file[0])
            csv_df = pd.read_csv(csv_path)
            # Determine the number of files to match with the CSV data
            num_files = min(len(bin_files), len(hd5_files))
            if len(csv_df) != num_files:
                # Print a warning if the number of rows in CSV doesn't match the number of binary and HDF5 files
                print(
                    "Warning: Number of rows in data.csv does not match the number of bin and hd5 files."
                )
            # Concatenate the CSV data with the existing DataFrame
            df = pd.concat([df[:num_files], csv_df], axis=1)
        else:
            # If CSV file doesn't exist, set the corresponding columns to None
            df[
                ["steps", "duration", "action", "undefined", "converged", "polarized"]
            ] = None
            # Extract unique identifier (UID) from the subfolder path
            df["uid"] = subfolder_path.split("/")[-1]

        # Return the processed DataFrame
        return df
