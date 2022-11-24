# %% [markdown]
# # Exercise 10.1 - Solution
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

# %% [markdown]
# ### Download Data

# %%
from  utils import CosmicRayDS
ds = CosmicRayDS(".")
n_test=10000
ds_train, ds_test= ds[:-n_test], ds[-n_test:]
# %%

# Data Handling of Graphs [https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html]
# A graph is used to model pairwise relations (edges) between objects (nodes). A single graph in PyG is described by an instance of torch_geometric.data.Data, which holds the following attributes by default:
#     data.x: Node feature matrix with shape [num_nodes, num_node_features]
#     data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
#     data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
#     data.y: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]
#     data.pos: Node position matrix with shape [num_nodes, num_dimensions]

# %% [markdown]
# Extract a single event from the test dataset.
# Inspect are the properties of the event accessable?
# You can use the `inspect` method of the rich library or a simple `print`

example_map = ds_test[0]
# import rich
# rich.inspect(example_map)
# %%
# %% [markdown]
# Plot an example sky map using the `skymap` function from `utils`
from  utils import skymap

fig = skymap(example_map.pos.T, c=example_map.x, zlabel="Energy (normed)", title = "Event 0")

# %%


# %% [markdown]
# ### Design DGCNN

# %% [markdown]
# #### Start with defining a kernel network
# Design a kernel network. The input to the kernel network is the central pixel coordinate and the neighborhood pixel coordinates.
# Hint: using `layers.BatchNormalization` can help to stabilize the training process of a DGCNN.
# 
# You can make use of the code snippet below.
# 
# Note that the output of the DNN should be `(None, nodes)`, where `None` is a placeholder for the batch size.
# 
# <em> In this case, we perform subtraction and concatenate the result with the central pixel value to combine translational invariance with local information. </em>

# %%
# def kernel_nn(data, nodes=16):
#     d1, d2 = data  # get xi ("central" pixel) and xj ("neighborhood" pixels)

#     dif = layers.Subtract()([d1, d2])  # perform substraction for translational invariance
#     x = layers.Concatenate(axis=-1)([d1, dif])  # add information on the absolute pixel value

#     x = layers.Dense(nodes, use_bias=False, activation="relu")(x)
#     x = layers.BatchNormalization()(x)

#     x = layers.Dense(nodes, use_bias=False, activation="relu")(x)
#     x = layers.BatchNormalization()(x)

#     x = layers.Dense(nodes, use_bias=False, activation="relu")(x)
#     x = layers.BatchNormalization()(x)
#     return x

class FFN(nn.Module):
    def __init__(self,n_in, n_out, n_hidden=20) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(n_in,n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(n_hidden,n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(n_hidden,n_out),
            nn.BatchNorm1d(n_out),
            nn.LeakyReLU(0.1),
        )
    
    def forward(self,*args, **kwargs):
        return self.seq(*args, **kwargs)

# %% [markdown]
# #### Build complete graph network model
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
        self.conv1=EdgeConv(FFN(4*2,10))
        self.conv2=EdgeConv(FFN(10*2,5))
        self.out = FFN(5,1)
    
    def forward(self,batch:Batch):
        # We run knn on the positions 
        # knn needs to know about the batches, otherwise it connects 
        # points from different events
        edge_index = knn_graph(batch.pos, batch=batch.batch, k=10)
        x = torch.hstack([batch.x, batch.pos])
        x = self.conv1(x,edge_index=edge_index)
        x = self.conv2(x,edge_index=edge_index)
        x= global_add_pool(x, batch.batch)
        x = self.out(x)
        return x.squeeze()


# %%
from torch_geometric.loader import DataLoader
loader = DataLoader(ds_train, batch_size=32)
model = GNN()
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_f = nn.BCEWithLogitsLoss()

for ibatch, batch in enumerate(loader):
    optim.zero_grad()
    loss = loss_f(model(batch), batch.y)
    loss.backward()
    optim.step()
    if ibatch % 10 ==0:
        print(model.conv1.nn.seq[0].weight[:3])
        print(f"{ibatch}: {loss}")
#%%
# %%
# %% [markdown]
# You can inspect the kernel network using:

# %%
model.layers[2].kernel_func.summary()

# %% [markdown]
# <em> The kernel network maps the energies an positions of 2 cosmic rays (the central and the neighbor comsic ray) to 8 features. </em>

# %% [markdown]
# The kernel network in the third layer maps from 16 extracted features (of 2 cosmic rays) to 32 new features and looks like this:

# %%
model.layers[6].kernel_func.summary()

# %% [markdown]
# ### Train the model

# %%
model.compile(loss="binary_crossentropy",
              optimizer=keras.optimizers.Adam(3E-3, decay=1E-4),
              metrics=['acc'])

# %% [markdown]
# If you don't have `networkx` or `sklearn` install it by executing:

# %%
history = model.fit(train_input_data, y_train, batch_size=64, epochs=4)

# %% [markdown]
# ## Visualization of the underlying graph
# To inspect the changing neighborhood relation (we used a dynamic layer) of the nodes, we visualize the underlying graph structure.
# 
# Note that plotting may take some time, so be a bit patient.

# %% [markdown]
# To perform the relative complex plotting, we make use of networkx and sklearn.  
# If you don't have installed the packages yet, run the cell below.

# %%
import sys
# !{sys.executable} -m pip install sklearn
# !{sys.executable} -m pip install networkx

# %%
import tensorflow.keras.backend as K
from sklearn.neighbors import kneighbors_graph
import networkx as nx

edge_layers = [l for l in model.layers if "edge_conv" in l.name]
coord_mask = [np.sum(np.linalg.norm(inp_d[test_id], axis=-1)) == 500 for inp_d in train_input_data]
assert True in coord_mask, "For plotting the spherical graph at least one input has to have 3 dimensions XYZ"
fig, axes = plt.subplots(ncols=len(edge_layers), figsize=(5 * len(edge_layers), 5))

for i, e_layer in enumerate(edge_layers):
    points_in, feats_in = model.inputs
    coordinates = e_layer.get_input_at(0)
    functor = K.function(model.inputs, coordinates)
    sample_input = [inp[np.newaxis, test_id] for inp in train_input_data]

    if type(e_layer.input) == list:
        layer_points, layer_features = functor(sample_input)
    else:
        layer_points = functor(sample_input)

    layer_points = np.squeeze(layer_points)
    adj = kneighbors_graph(layer_points, e_layer.next_neighbors)
    g = nx.DiGraph(adj)

    for c, s in zip(coord_mask, sample_input):
        if c == True:
            pos = s
            break

    axes[i].set_title("Graph in %s" % e_layer.name)
    nx.draw(g, cmap=plt.get_cmap('viridis'), pos=pos.squeeze()[:, :-1],
            node_size=10, width=0.5, arrowsize=5, ax=axes[i])
    axes[i].axis('equal')


