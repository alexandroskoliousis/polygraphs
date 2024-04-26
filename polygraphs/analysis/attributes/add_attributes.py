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

    def add_config(self, key_path):
        # Add values from a specified key_path in JSON config files to the dataframe
        values = []
        for config_path in self.dataframe["config_json_path"]:
            with open(config_path, "r") as file:
                json_str = file.read()
            json_obj = json.loads(json_str)
            keys = key_path.split(".")
            value = None
            current_obj = json_obj
            for key in keys:
                if key in current_obj:
                    current_obj = current_obj[key]
                    value = current_obj
                else:
                    value = None
                    break
            values.append(value)
        column_name = key_path.replace(".", "_").replace(" ", "")  # Format column name
        self.dataframe[column_name] = (
            values  # Add values as a new column in the dataframe
        )
