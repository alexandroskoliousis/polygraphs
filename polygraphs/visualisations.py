"""
PolyGraph visualisations
"""
import os
import inspect

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

import networkx as nx
import dgl

import numpy as np


def _get_layout(graph, name, **kwargs):
    if name is None:
        return None
    assert isinstance(name, str)
    # Create friendly dictionary from list of (name, function) tuples
    layouts = dict(inspect.getmembers(nx.drawing.layout, inspect.isfunction))
    # Find layout routine from name
    layout = layouts.get(f'{name.lower()}_layout')
    if layout is None:
        raise Exception(f'Invalid graph layout: {name}')
    return layout(graph, **kwargs)


def draw(graph, figsize=None, layout=None, **kwargs):
    """
    Draws a PolyGraph.
    """
    # Export beliefs from graph node attributes, if present
    if 'belief' in graph.ndata:
        beliefs = graph.ndata.get('belief').numpy()
    else:
        print('Beliefs not found')
        beliefs = np.zeros(graph.num_nodes())
    print(beliefs)
    # Convert DGL graph to NetworkX graph
    G = dgl.to_networkx(graph)  # pylint: disable=invalid-name
    # Plot
    fig, ax = plt.subplots(figsize=figsize)  # pylint: disable=invalid-name
    nx.draw(G,
            pos=_get_layout(G, layout, **kwargs),
            edgecolors='black',
            with_labels=True,
            node_color=beliefs,
            vmin=0.0,
            vmax=1.0,
            cmap=plt.cm.coolwarm,  # pylint: disable=no-member
            ax=ax)
    # Show frame
    ax.axis('on')
    # Map scalars to RGBA
    mapper = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm)  # pylint: disable=no-member
    # Show colorbar
    fig.colorbar(mapper)
    return fig, ax


def animate(graph, frames, filename=None, fps=1, figsize=None, layout=None, **kwargs):
    """
    Animate a graph.
    """
    # Sort out destination file
    if not filename:
        # Generate a GIF by default in local directory
        filename = 'animation.gif'

    # Get file extension
    _, ext = os.path.splitext(filename)
    if ext not in ['.gif', '.mp4']:
        raise Exception("Invalid file name: {}".format(filename))

    # Movie writer
    writer = PillowWriter(fps=fps) if ext == 'gif' else FFMpegWriter(fps=fps)

    # Export beliefs from graph node attributes, if present
    if 'belief' in graph.ndata:
        beliefs = graph.ndata.get('belief').numpy()
    else:
        beliefs = np.zeros(graph.num_nodes())
    # Add latest beliefs to the right side of the deque
    frames.append(beliefs)
    # Convert DGL graph to NetworkX graph
    G = dgl.to_networkx(graph)  # pylint: disable=invalid-name
    # Set plot
    fig, ax = plt.subplots(figsize=figsize)  # pylint: disable=invalid-name
    # Map scalars to RGBA
    mapper = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm)  # pylint: disable=no-member
    # Show colorbar
    fig.colorbar(mapper)

    def update(frame):
        # Redraw graph
        nx.draw(G,
                pos=_get_layout(G, layout, **kwargs),
                edgecolors='black',
                with_labels=True,
                node_color=frame,
                vmin=0.0,
                vmax=1.0,
                cmap=plt.cm.coolwarm,  # pylint: disable=no-member
                ax=ax)
        ax.axis('on')

    # Create animation
    animation = FuncAnimation(fig, update, frames=frames, blit=False)
    # Save animation to file
    animation.save(filename, writer=writer)
    return animation
