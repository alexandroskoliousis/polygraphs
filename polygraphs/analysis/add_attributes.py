import networkx as nx  # Importing networkx library for working with graphs
import json  # Importing json library for working with JSON data


class AddAttributes:
    def __init__(self, graph_converter, belief_processor):
        # Initialize AddAttributes with graph_converter and belief_processor objects
        self.graph_converter = graph_converter
        self.belief_processor = belief_processor

    def density(self, dataframe):
        # Calculate density of each graph in dataframe and add it as a new column
        density_list = [
            nx.density(self.graph_converter.get_networkx_object(bin_file_path))
            for bin_file_path in dataframe["bin_file_path"]
        ]
        self.dataframe["density"] = density_list

    def add_config(self, *key_paths):
        # Add values from a specified key_path in JSON config files to the dataframe
        configs = {}
        for key_path in key_paths:
            values = []
            for config_path in self.dataframe["config_json_path"]:
                # Check if we have already read config file
                if config_path in configs:
                    current_obj = configs[config_path]
                else:
                    with open(config_path, "r") as file:
                        json_str = file.read()
                        json_obj = json.loads(json_str)
                        configs[config_path] = json_obj
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
