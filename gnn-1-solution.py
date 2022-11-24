# %% [markdown]
# # GNN Warmeup
# For this exercise we will learn to use `pytorch_geometric` (PyG) to run GNNs.
# The library comes with a comprehensive [documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html) and not only provides tools to handle graphs but also provides a large set of GNN specific layers and dataset.

# %% [markdown]
# ## Task 1.1
# [Data Handling of Graphs](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs) offers a nice introduction into handling graphs.
# Provide an adjacency matric for cyclic graph (each nodes connects to the next) with 5 nodes. Convert the adjacency matric to an edge_index. With this edge_index implement a graph with features [0,1,...,n-1]:
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

adj = torch.tensor(
    [
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ]
)

cycle = Data(
    x=torch.tensor(range(5)),
    edge_index=dense_to_sparse(adj)[0],
)
# %% [markdown]
# ## Task 1.2
# Implement binary tree with 7 nodes over three levels directly constructing the edge_index.:
#    0
#  1   2
# 3 4 5 6

# %%
tree = Data(
    x=torch.tensor(range(7)),
    edge_index=torch.tensor([[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]]).T,
)

# %% [markdown]
# ## Task 1.3
# Convert the implemented PyG graphs to `networkx` graphs and plot them.
# %%
from torch_geometric.utils import to_networkx
import networkx as nx

nx.draw(to_networkx(tree))
# %%
nx.draw(to_networkx(cycle))
# %% [markdown]
# ## Task 1.4
# Have a look at the documentation on [Mini-batches](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#mini-batches)
# Batch the two graphs together. And plot the batch with networkx. What do you see ?

# %%
batch = Batch.from_data_list([tree, cycle])
nx.draw(to_networkx(batch))
# -> The Graphs are combined to a larger graph.


# %% [markdown]
# ## Task 1.5
# Inspect the properties of the batch. 
# You can use the `inspect` method of the rich library or a simple `print`.
# What do the `batch` and `ptr` attributes do?
# %%
import rich
rich.inspect(batch)
# -> `batch` provides the index for teh feature  matrix of each graph in the feature vector of the batch
# -> `ptr` does the same, but with a range
