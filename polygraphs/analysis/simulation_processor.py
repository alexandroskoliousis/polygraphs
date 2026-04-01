import pandas as pd
import json
from pathlib import Path, PosixPath, PurePath
import warnings

# Check for DGL availability once at import time
try:
    import dgl  # noqa: F401

    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False


class SimulationProcessor:

    def __init__(self, include=None, exclude=None, config_check=True):
        """
        Initialize SimulationProcessor with optional include and exclude parameters.

        Parameters:
        - include (dict): Dictionary specifying key-value pairs to include directories based on config.json.
        - exclude (dict): Dictionary specifying key-value pairs to exclude directories based on config.json.
        - config_check (bool): Check config folder location in simulation.results
        """
        self.dataframe = pd.DataFrame()
        self.configs = {}
        self.include = include if include else {}
        self.exclude = exclude if exclude else {}
        self.config_check = config_check
        self.initial_columns = [
            "bin_file_path",
            "hd5_file_path",
            "config_json_path",
        ]

    def load_config(self, config_json_path):
        with open(config_json_path, "r") as f:
            config_data = json.load(f)
        return config_data

    def match_criteria(self, config_data, criteria):
        for key, value in criteria.items():
            keys = key.split(".")
            data = config_data
            for k in keys:
                data = data.get(k, None)
                if data is None:
                    return False
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

    def _discover_graph_files(self, subfolder_path, hd5_files):
        """
        For each .hd5 file, find a matching graph file.

        Prefers .pt files. Falls back to .bin files only if DGL is installed.

        Returns:
            tuple: (graph_files, hd5_files, file_type) where file_type is "pt" or "bin",
                   or ([], [], None) if no matching pairs are found.
        """
        graph_files = []
        matched_hd5 = []

        # First pass: try .pt files
        for sim in hd5_files:
            pt_file = sim.with_suffix(".pt")
            if pt_file.exists():
                graph_files.append(str(pt_file))
                matched_hd5.append(str(sim))

        if graph_files:
            return graph_files, matched_hd5, "pt"

        # Second pass: fall back to .bin if DGL is available
        if DGL_AVAILABLE:
            for sim in hd5_files:
                bin_file = sim.with_suffix(".bin")
                if bin_file.exists():
                    graph_files.append(str(bin_file))
                    matched_hd5.append(str(sim))

            if graph_files:
                return graph_files, matched_hd5, "bin"

        return [], [], None

    def process_simulations(self, path):
        """
        Process simulation data from the specified path.

        Parameters:
        - path (str or list): The path to the root folder containing simulation data.
        """
        folders = []
        if isinstance(path, list):
            _folders = []
            for _path in path:
                _ = self.expand_path(_path)
                _folders.extend((_, *[x for x in _.rglob("*/")]))

            _folders_set = set(str(f) for f in _folders)
            folders = [Path(f) for f in _folders_set]
        else:
            _ = self.expand_path(path)
            folders = [_, *[x for x in _.rglob("*/")]]

        result_df = pd.DataFrame(columns=self.initial_columns)

        for folder in folders:
            try:
                subfolder_df = self.process_subfolder(folder)
                if isinstance(subfolder_df, pd.DataFrame):
                    result_df = pd.concat(
                        [result_df, subfolder_df.dropna(axis=1, how="all")],
                        ignore_index=True,
                    )
            except (FileNotFoundError, PermissionError) as e:
                warnings.warn(f"Error accessing folder: {e}", RuntimeWarning)

        self.dataframe = result_df
        self.format_known_column_types()
        self.reorder_columns()

    def process_subfolder(self, subfolder_path):
        """
        Process each subfolder in the root folder.

        Parameters:
        - subfolder_path (str): The path to the subfolder to be processed.

        Returns:
        - pandas.DataFrame or None
        """
        config_path = subfolder_path / "configuration.json"

        if not config_path.exists():
            return

        config_data = self.load_config(config_path)

        config_directory = config_data.get("simulation", {}).get("results", "")
        config_base_dir = PurePath(config_directory).parts[-1]

        if config_base_dir != subfolder_path.name:
            warnings.warn(
                f"Results folder does not match configuration.json: {subfolder_path}",
                UserWarning,
            )
            if self.config_check == True:
                return

        if self.include or self.exclude:
            if not self.should_include(config_data) or self.should_exclude(config_data):
                return

        # Filter and sort HDF5 files
        _hd5_files = sorted(subfolder_path.glob("*.hd5"))

        if len(_hd5_files) == 0:
            return

        # Discover graph files (.pt preferred, .bin fallback)
        graph_files, hd5_files, file_type = self._discover_graph_files(
            subfolder_path, _hd5_files
        )

        if not graph_files:
            return

        df = pd.DataFrame()
        df["bin_file_path"] = graph_files
        df["hd5_file_path"] = hd5_files
        df["config_json_path"] = config_path

        df["trials"] = config_data.get("trials")
        df["network_size"] = config_data.get("network", {}).get("size")
        df["network_kind"] = config_data.get("network", {}).get("kind")
        df["op"] = config_data.get("op")
        df["epsilon"] = config_data.get("epsilon")

        csv_file = subfolder_path / "data.csv"

        if csv_file.exists():
            csv_df = pd.read_csv(csv_file)
            num_files = len(graph_files)

            if len(csv_df) != num_files:
                warnings.warn(
                    f"Number of rows in data.csv did not match bin/pt and hd5 files: {subfolder_path}",
                    UserWarning,
                )
                return

            df = pd.concat([df[:num_files], csv_df], axis=1)
        else:
            df[
                ["steps", "duration", "action", "undefined", "converged", "polarized"]
            ] = None
            df["uid"] = subfolder_path.name

        self.configs[config_path] = config_data

        return df

    def add_config(self, *key_paths):
        """
        Add values from a specified key_paths in JSON config files to the dataframe
        """
        for key_path in key_paths:
            values = []
            for config_path in self.dataframe["config_json_path"]:
                if config_path in self.configs:
                    current_obj = self.configs[config_path]
                else:
                    json_obj = self.load_config(config_path)
                    self.configs[config_path] = json_obj
                    current_obj = json_obj

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

            column_name = key_path.replace(".", "_").replace(" ", "")
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
            "polarized": "bool",
        }

        for col, _type in known_columns.items():
            if col in self.dataframe.columns:
                try:
                    self.dataframe[col] = self.dataframe[col].astype(_type)
                except Exception:
                    pass
