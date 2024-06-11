# Custom Operations
The core functionality of each [PolyGraph Operation](https://github.com/alexandroskoliousis/polygraphs/blob/main/polygraphs/ops/core.py) works using a [PyTorch forward method](https://pytorch.org/docs/stable/generated/torch.autograd.Function.forward.html) which specifies what happens at each iteration of the simulation. This method calls the `experiment()` method to run experiments and generate some results, filters the nodes that eligible to send messages using `filterfn()` and [sends and receives](https://docs.dgl.ai/en/2.1.x/generated/dgl.DGLGraph.send_and_recv.html) messages on the graph using the `messagefn()`, `reducefn()` and `applyfn()` methods.

| Function   | Method         | Description                                                                                                        |
|------------|----------------|--------------------------------------------------------------------------------------------------------------------|
| Experiment | `experiment()` | Generates evidence communicated to edges                                                                           |
| Sampler    | `sample()`     | Draws a sample from a distribution                                                                                 |
| Trials     | `trials()`     | Returns the number of trials by default                                                                            |
| Filter     | `filterfn()`   | Filters edges that are eligible to communicate evidence                                                            |
| Message    | `message()`    | Creates the messages that are sent along the edges                                                                 |
| Reduce     | `reducefn()`   | Specifies how the evidence received from all neighbours should be aggregated by each node                          |
| Apply      | `applyfn()`    | Updates the beliefs on the nodes after the messages from neighbours have been aggregated using the reduce function |

To create a custom op, create a class that inherits from either the [PolyGraphOp](https://github.com/alexandroskoliousis/polygraphs/blob/main/polygraphs/ops/core.py) or extends and existing op such as the [BalaGoyalOp](https://github.com/alexandroskoliousis/polygraphs/blob/main/polygraphs/ops/common.py) and override any of the methods above to change the simulation behaviour.
