import os  # Importing os module for interacting with the operating system
import pandas as pd  # Importing pandas library for data manipulation
import re  # Importing re module for regular expressions
import json  # Importing json module for working with JSON data


class SimulationProcessor:

    def __init__(self, include=None, exclude=None):
        """
        Initialize SimulationProcessor with optional include and exclude parameters.

        Parameters:
        - include (dict): Dictionary specifying key-value pairs to include directories based on config.json.
        - exclude (dict): Dictionary specifying key-value pairs to exclude directories based on config.json.
        """
        self.path = ""  # Path to the root folder of simulation data
        self.dataframe = pd.DataFrame()  # DataFrame to store processed simulation data
        self.configs = {}  # Dictionary to save config files
        self.include = include if include else {}
        self.exclude = exclude if exclude else {}

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

    def load_config(self, config_json_path):
        with open(config_json_path, "r") as f:
            config_data = json.load(f)
        return config_data

    def match_criteria(self, config_data, criteria):
        for key, value in criteria.items():
            keys = key.split(".")
            data = config_data
            # Get the corresponding key
            for k in keys:
                data = data.get(k, None)
                if data is None:
                    return False
            # Check value against retreived key
            if data != value:
                return False
        return True

    def should_include(self, config_data):
        if self.include:
            return self.match_criteria(config_data, self.include)
        return True

    def should_exclude(self, config_data):
        if self.exclude:
            return self.match_criteria(config_data, self.exclude)
        return False

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
        # Include the root folder as one of the folder to search for simulations
        folders = [self.path]
        try:
            # Walk through the directory tree starting from the specified path
            for dirpath, dirnames, filenames in os.walk(self.path):
                for dirname in dirnames:
                    # Construct the full path of each subfolder
                    folder_path = os.path.join(dirpath, dirname)
                    folders.append(folder_path)
        except (FileNotFoundError, PermissionError) as e:
            # Handle exceptions if there are issues accessing folders
            print(f"Error accessing folder: {e}")

        # Initialize an empty DataFrame to store processed simulation data
        result_df = pd.DataFrame(
            columns=["bin_file_path", "hd5_file_path", "config_json_path"]
        )

        # Process each subfolder and concatenate the results into the result DataFrame
        for folder in folders:
            subfolder_df = self.process_subfolder(folder)
            # Add folder if we get a dataframe with simulations
            if isinstance(subfolder_df, pd.DataFrame):
                result_df = pd.concat(
                    [result_df, subfolder_df.dropna(axis=1, how="all")],
                    ignore_index=True,
                )

        # Store the aggregated DataFrame in the class attribute `self.dataframe`
        self.dataframe = result_df

    def process_subfolder(self, subfolder_path):
        """
        Process each subfolder in the root folder.

        Parameters:
        - subfolder_path (str): The path to the subfolder to be processed.

        Returns:
        - pandas.DataFrame: DataFrame containing processed data from the subfolder, or None if the subfolder
        does not meet the inclusion/exclusion criteria or does not contain relevant files.
        """
        # Get a list of files in the subfolder
        files = os.listdir(subfolder_path)

        # Check if there is a configuration JSON file in the subfolder
        config_file = [f for f in files if f == "configuration.json"]

        # If no configuration file is found, skip processing this subfolder
        if not config_file:
            return

        # Reslove path to config file
        config_path = os.path.join(subfolder_path, config_file[0])

        # Check if the subfolder meets the inclusion/exclusion criteria
        if self.include or self.exclude:
            # Load the configuration JSON file
            config_data = self.load_config(config_path)

            # Skip directory if it meets criteria
            if not self.should_include(config_data) or self.should_exclude(config_data):
                return

            # Store config in configs if we didnt skip directory
            self.configs[config_path] = config_data

        # Filter and sort HDF5 files based on their numerical order
        _hd5_files = sorted([f for f in files if f.endswith(".hd5")])

        # If no HDF5 files are found, skip processing this subfolder
        if len(_hd5_files) == 0:
            return

        # Initialize lists to store paths to binary and HDF5 files
        hd5_files = []
        bin_files = []
        for sim in _hd5_files:
            # Find corresponding .bin files for each .hd5 file
            _bin_file = sim.replace("hd5", "bin")
            if _bin_file in files:
                hd5_files.append(sim)
                bin_files.append(_bin_file)

        # Initialize an empty DataFrame to store processed data
        df = pd.DataFrame()
        # Add paths to binary files to the DataFrame
        df["bin_file_path"] = [os.path.join(subfolder_path, f) for f in bin_files]
        # Add paths to HDF5 files to the DataFrame
        df["hd5_file_path"] = [os.path.join(subfolder_path, f) for f in hd5_files]

        # Add configuration JSON file path to the DataFrame
        df["config_json_path"] = config_path
        # Extract parameters from the configuration JSON file
        trials, network_size, network_kind, op, epsilon = self.extract_params(
            config_path
        )
        df["trials"] = trials
        df["network_size"] = network_size
        df["network_kind"] = network_kind
        df["op"] = op
        df["epsilon"] = epsilon

        # Check if there is a data.csv file in the subfolder
        csv_file = [f for f in files if f == "data.csv"]

        if csv_file:
            csv_path = os.path.join(subfolder_path, csv_file[0])
            csv_df = pd.read_csv(csv_path)
            num_files = len(hd5_files)

            # Raise an error if the number of rows in CSV doesn't match the number of binary and HDF5 files
            if len(csv_df) != num_files:
                raise ValueError(
                    f"Number of rows in data.csv does not match the number of bin and hd5 files in: {subfolder_path}"
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

    def add_config(self, *key_paths):
        """
        Add values from a specified key_paths in JSON config files to the dataframe
        """
        # Loop through each key path
        for key_path in key_paths:
            values = []
            for config_path in self.dataframe["config_json_path"]:
                # Check if we have already read config file
                if config_path in self.configs:
                    current_obj = self.configs[config_path]
                else:
                    json_obj = self.load_config(config_path)
                    self.configs[config_path] = json_obj
                    current_obj = json_obj

                # Get data for key paths
                keys = key_path.split(".")
                value = None
                for key in keys:
                    if key in current_obj:
                        current_obj = current_obj[key]
                        value = current_obj
                    else:
                        value = None
                        break
                values.append(value)
            # Format column name
            column_name = key_path.replace(".", "_").replace(" ", "")
            # Add values as a new column in the dataframe
            self.dataframe[column_name] = values
