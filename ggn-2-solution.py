# %% [markdown]
# # Task 2
# ## Signal Classification using Dynamic Graph Convolutional Neural Networks
# After a long journey through the universe before reaching the earth, the cosmic particles interact with the galactic magnetic field $B$.
# As these particles carry a charge $q$ they are deflected in the field by the Lorentz force $F = q \cdot v × B$.
# Sources of cosmic particles are located all over the sky, thus arrival distributions of the cosmic particles are isotropic in general. However, particles originating from the same source generate on top of the isotropic
# arrival directions, street-like patterns from galactic magnetic field deflections.
#
# In this tasks we want to classify whether a simulated set of $500$ arriving cosmic particles contains street-like patterns (signal), or originates from an isotropic background.
#
# Training graph networks can be computationally demanding, thus, we recommend to use a GPU for this task.

# %%
import torch
from torch import nn
from torch_geometric.data import Data, Batch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils import CosmicRayDS

ds = CosmicRayDS(".")
n_test = 10000
ds_train, ds_test = ds[:-n_test], ds[-n_test:]
# %%


# %% [markdown]
# # Task 2.1
# Extract a single event from the test dataset and inspect it.
# Plot an example sky map using the `skymap` function from `utils`
#%%
from utils import skymap
event0 = ds_test[0]
fig=skymap(
    event0.pos, c=event0.x, zlabel="Energy (normed)", title="Event 0"
)

# %% [markdown]
# # Task 2.2
# Generate edges for the event using `knn_graph`. 
# Plot the edges by passing the `edge_index` to the `skymap` function. How does the number of edges scale with the $k$?
# %%
from torch_geometric.nn import knn_graph


fig=skymap(
    event0.pos, c=event0.x, edge_index=knn_graph(event0.pos, k=3), zlabel="Energy (normed)", title="Event 0"
)
# -> k*num_nodes


# %% [markdown]
# # Task 2.3
# Write a class to return a simple Feed-Forward-Network (FFN) for a given number inputs and outputs. (3 layers, 20 hidden nodes, BatchNorm, LeakyReLU)
# %%
class FFN(nn.Module):
    def __init__(self, n_in, n_out, n_hidden=20):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, *args, **kwargs):
        return self.seq(*args, **kwargs)


# %% [markdown]
# GNNs classifiers are frequently build in a two step process: First MessagePassingLayers( aka Graph [Convolutional Layers](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#convolutional-layers) ) update the nodes. These exploit the local information. Then, the nodes are aggregated using [Pooling Layers](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#pooling-layers), reducing the graph to a single feature vector. This feature vector is then passed through a FFN to get the classification output.
# Have a look at the documentation of [EdgeConv][(https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.EdgeConv) and [DynamicEdgeConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.DynamicEdgeConv).

# # Build complete graph network model
# In the first layer, it might be advantageous to choose the next neighbors using the coordinates of the cosmic ray but perform the convolution using their energies also.
# Thus, we input `y = EdgeConv(...)[points_input, feats_input]` into the first EdgeConv layer.
# If we later want to perform a dynamic EdgeConv (we want to update the graph), we simply input `z = EdgeConv(...)(y)`.
#
# To specify the size of the "convolutional filter", make use of the `next_neighbors` argument (searches for $k$ next neighbors for each cosmic ray).

# %%
# points_input = layers.Input((500, 3))
# feats_input = layers.Input((500, 4))

# x = EdgeConv(lambda a: kernel_nn(a, nodes=8), next_neighbors=8)([points_input, feats_input])  # conv with fixed graph
# x = layers.Activation("relu")(x)
# x = EdgeConv(lambda a: kernel_nn(a, nodes=16), next_neighbors=8)([points_input, x])  # conv with fixed graph
# x = layers.Activation("relu")(x)
# x = EdgeConv(lambda a: kernel_nn(a, nodes=32), next_neighbors=8)([x, x])  # conv with dynamic graph
# x = layers.Activation("relu")(x)
# x = layers.GlobalAveragePooling1D(name="embedding")(x)
# out = layers.Dense(2, name="classification", activation="softmax")(x)

# model = keras.models.Model([points_input, feats_input], out)
# print(model.summary())

from torch_geometric.nn import knn_graph, EdgeConv, global_add_pool


class GNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = EdgeConv(FFN(4 * 2, 10))
        self.conv2 = EdgeConv(FFN(10 * 2, 10))
        self.conv3 = EdgeConv(FFN(10 * 2, 5))
        self.out = FFN(5, 1)

    def forward(self, batch: Batch):
        # We run knn on the positions
        # knn needs to know about the batches, otherwise it connects
        # points from different events
        edge_index = knn_graph(batch.pos, batch=batch.batch, k=10)
        x = torch.hstack([batch.x, batch.pos])
        x = self.conv1(x, edge_index=edge_index)
        x = self.conv2(x, edge_index=edge_index)
        edge_index = knn_graph(x, batch=batch.batch, k=10)
        x = self.conv3(x, edge_index=edge_index)
        x = global_add_pool(x, batch.batch)
        x = self.out(x)
        return x.squeeze()


# %%
from torch_geometric.loader import DataLoader
from utils import metric_aggr

device = torch.device("cuda")
loader = DataLoader(ds_train, batch_size=64, shuffle=True)
model = GNN().to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_f = nn.BCEWithLogitsLoss()

for iepoch in range(4):
    for batch in tqdm(loader):
        optim.zero_grad()
        batch = batch.to(device)
        output = model(batch)
        loss = loss_f(output, batch.y)
        loss.backward()
        optim.step()
        metric_aggr("loss", float(loss.detach().cpu()))

