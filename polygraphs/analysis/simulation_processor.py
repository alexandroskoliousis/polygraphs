import pandas as pd
import json
from pathlib import Path, PosixPath, PurePath
import warnings


class SimulationProcessor:

    def __init__(self, include=None, exclude=None, config_check=True):
        """
        Initialize SimulationProcessor with optional include and exclude parameters.

        Parameters:
        - include (dict): Dictionary specifying key-value pairs to include directories based on config.json.
        - exclude (dict): Dictionary specifying key-value pairs to exclude directories based on config.json.
        - ignore_config (bool): Check config folder location in simulation.results
        """
        self.dataframe = pd.DataFrame()  # DataFrame to store processed simulation data
        self.configs = {}  # Dictionary to save config files
        self.include = include if include else {}
        self.exclude = exclude if exclude else {}
        self.config_check = config_check
        self.initial_columns = ["bin_file_path", "hd5_file_path", "config_json_path"]

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

    def expand_path(self, path):
        """Expand the path to handle user home directory (~) or relative paths"""
        if path.startswith("~"):
            return PosixPath(path).expanduser()
        else:
            return Path(path).resolve()

    def process_simulations(self, path):
        """
        Process simulation data from the specified path.

        Parameters:
        - path (str or list): The path to the root folder containing simulation data.

        Returns:
        - None

        This method walks through the directory tree starting from the specified path,
        identifies subfolders that represent individual simulation runs (identified by
        valid UUID folder names), processes each subfolder using the `process_subfolder`
        method, and aggregates the results into a single DataFrame stored in `self.dataframe`.
        """
        # Initialize a list to store paths of subfolders representing individual simulations
        # Include the root folder as one of the folder to search for simulations
        folders = []
        if isinstance(path, list):
            _folders = []
            for _path in path:
                _ = self.expand_path(_path)
                _folders.extend((_, *[x for x in _.rglob("*/")]))

            # Maks sure that folders are unique
            _folders_set = set(str(f) for f in _folders)
            folders = [Path(f) for f in _folders_set]
        else:
            _ = self.expand_path(path)
            folders = [_, *[x for x in _.rglob("*/")]]

        # Initialize an empty DataFrame to store processed simulation data
        result_df = pd.DataFrame(columns=self.initial_columns)

        # Process each subfolder and concatenate the results into the result DataFrame
        for folder in folders:
            try:
                subfolder_df = self.process_subfolder(folder)
                # Add folder if we get a dataframe with simulations
                if isinstance(subfolder_df, pd.DataFrame):
                    result_df = pd.concat(
                        [result_df, subfolder_df.dropna(axis=1, how="all")],
                        ignore_index=True,
                    )
            except (FileNotFoundError, PermissionError) as e:
                # Handle exceptions if there are issues accessing folders
                warnings.warn(f"Error accessing folder: {e}", RuntimeWarning)

        # Store the aggregated DataFrame in the class attribute `self.dataframe`
        self.dataframe = result_df
        self.format_known_column_types()
        self.reorder_columns()

    def process_subfolder(self, subfolder_path):
        """
        Process each subfolder in the root folder.

        Parameters:
        - subfolder_path (str): The path to the subfolder to be processed.

        Returns:
        - pandas.DataFrame: DataFrame containing processed data from the subfolder, or None if the subfolder
        does not meet the inclusion/exclusion criteria or does not contain relevant files.
        """

        # Resolve path to config file using pathlib
        config_path = subfolder_path / "configuration.json"

        # If no configuration file is found, skip processing this subfolder
        if not config_path.exists():
            return

        # Load the configuration JSON file
        config_data = self.load_config(config_path)

        # Find base directory (UUID) of simulation directory in configuration file
        config_directory = config_data.get("simulation", {}).get("results", "")
        config_base_dir = PurePath(config_directory).parts[-1]

        # Check that the configuration file directory matches the directory name
        # if the config_check parameter is true skip directory
        if config_base_dir != subfolder_path.name:
            warnings.warn(
                f"Results folder does not match configuration.json: {subfolder_path}",
                UserWarning,
            )
            if self.config_check == True:
                return

        # Check if the subfolder meets the inclusion/exclusion criteria
        if self.include or self.exclude:
            # Skip directory if it meets criteria
            if not self.should_include(config_data) or self.should_exclude(config_data):
                return

        # Filter and sort HDF5 files based on their numerical order
        _hd5_files = sorted(subfolder_path.glob("*.hd5"))

        # If no HDF5 files are found, skip processing this subfolder
        if len(_hd5_files) == 0:
            return

        # Initialize lists to store paths to binary and HDF5 files
        hd5_files = []
        bin_files = []
        for sim in _hd5_files:
            # Find corresponding .bin files for each .hd5 file
            _bin_file = sim.with_suffix(".bin")
            if _bin_file.exists():
                hd5_files.append(str(sim))
                bin_files.append(str(_bin_file))

        # Initialize an empty DataFrame to store processed data
        df = pd.DataFrame()
        # Add paths to binary files to the DataFrame
        df["bin_file_path"] = bin_files
        # Add paths to HDF5 files to the DataFrame
        df["hd5_file_path"] = hd5_files
        # Add configuration JSON file path to the DataFrame
        df["config_json_path"] = config_path

        # Extract parameters from the configuration JSON file
        df["trials"] = config_data.get("trials")
        df["network_size"] = config_data.get("network", {}).get("size")
        df["network_kind"] = config_data.get("network", {}).get("kind")
        df["op"] = config_data.get("op")
        df["epsilon"] = config_data.get("epsilon")

        # Check if there is a data.csv file in the subfolder
        csv_file = subfolder_path / "data.csv"

        if csv_file.exists():
            csv_df = pd.read_csv(csv_file)
            num_files = len(hd5_files)

            # Skip folder if rows in CSV doesn't match the number of binary and HDF5 files
            if len(csv_df) != num_files:
                warnings.warn(
                    f"Number of rows in data.csv did not match bin and hd5 files: {subfolder_path}",
                    UserWarning,
                )
                return

            # Concatenate the CSV data with the existing DataFrame
            df = pd.concat([df[:num_files], csv_df], axis=1)
        else:
            # If CSV file doesn't exist, set the corresponding columns to None
            df[
                ["steps", "duration", "action", "undefined", "converged", "polarized"]
            ] = None
            # Extract unique identifier (UID) from the subfolder path
            df["uid"] = subfolder_path.name

        # Store config in self.configs if we didnt skip directory
        self.configs[config_path] = config_data

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
            self.reorder_columns()

    def reorder_columns(self):
        """Moves file columns to end of DataFrame"""
        new_column_order = [
            col for col in self.dataframe.columns if col not in self.initial_columns
        ] + self.initial_columns
        self.dataframe = self.dataframe[new_column_order]

    def format_known_column_types(self):
        """Convert known column types"""
        known_columns = {
            "trials": "int",
            "network_size": "int",
            "steps": "int",
            "network_kind": "category",
            "op": "category",
            "action": "category",
            "undefined": "bool",
            "converged": "bool",
            "polarized": "bool"
        }

        for col, _type in known_columns.items():
            if col in self.dataframe.columns:
                try:
                    self.dataframe[col] = self.dataframe[col].astype(_type)
                except:
                    pass
