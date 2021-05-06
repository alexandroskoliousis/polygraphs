import matplotlib.pyplot as plt
import networkx as nx
import dgl
from enum import Enum


class Layout(Enum):

    CIRCULAR = nx.circular_layout

    PLANAR = nx.planar_layout
    RANDOM = nx.random_layout
    SPRING = nx.spring_layout
    SPIRAL = nx.spiral_layout


def _getlayout(graph, layout):
    if layout:
        return layout(graph)
    else:
        return None


def draw(graph, layout=None):
    """
    Draws a graph.
    """
    g = dgl.to_networkx(graph)
    _, ax = plt.subplots()
    nx.draw(g,
            pos=_getlayout(g, layout),
            node_color='#EBF1DD',
            edgecolors='black',
            node_size=600,
            with_labels=True,
            ax=ax)


def animate(graph):
    """
    Animate a graph.
    """
    return None
