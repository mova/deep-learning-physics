import torch
import os
import gdown
from torch_geometric.data import InMemoryDataset, Data, Batch
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


class CosmicRayDS(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["cr_sphere.npz"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    # def len(self):
    #     return len(self.data)

    # def get(self, idx):
    #     return self.data[idx]

    def download(self):
        url = "https://drive.google.com/u/0/uc?export=download&confirm=HgGH&id=1XKN-Ik7BDyMWdQ230zWS2bNxXL3_9jZq"
        if os.path.exists(self.raw_file_names[0]) == False:
            gdown.download(url, self.raw_file_names[0], quiet=True)

    def process(self):
        f = np.load(self.raw_file_names[0])
        x = torch.tensor(f["data"]).float()
        y = torch.tensor(f["label"]).float()
        n_events, n_points, _ = x.shape

        data_list = []
        for idx in range(len(x)):
            data_list.append(
                Data(x=x[idx, :, 3].reshape(-1, 1), pos=x[idx, :, :3], y=y[idx])
            )
            # data_list[-1].num_nodes=n_points

        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]

        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def vec2ang(v):
    x, y, z = np.asarray(v).T
    phi = np.arctan2(y, x)
    theta = np.arctan2(z, (x * x + y * y) ** 0.5)
    return np.vstack([phi, theta]).T


def skymap(v, c=None, edge_index=None, zlabel="", title="", **kwargs):
    pos_ang= vec2ang(v)
    lons, lats = pos_ang.T
    lons = -lons
    fig = plt.figure(figsize=kwargs.pop("figsize", [12, 6]))
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.9], projection="hammer")
    events = ax.scatter(lons, lats, c=c, s=12, lw=2)
    
    
    if edge_index is not None:
        x = pos_ang[:,0][edge_index]
        y = pos_ang[:,1][edge_index]
        ax.plot(-x,y, linestyle='-', linewidth=.5)

    plt.colorbar(
        events, orientation="horizontal", shrink=0.85, pad=0.05, aspect=30, label=zlabel
    )
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    return fig


def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()


def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()



from collections import defaultdict


class MetricAggr:
    def __init__(self, log_interval: int = 20) -> None:
        self.log_interval = log_interval
        self.storage = defaultdict(list)

    def __call__(self, val_name: str, val):
        val_store = self.storage[val_name]
        val_store.append(val)
        if len(val_store) == self.log_interval:
            print(f"{val_name}: {np.mean(val_store)}")
            self.storage[val_name] = []
metric_aggr = MetricAggr()