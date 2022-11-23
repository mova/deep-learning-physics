import torch
import os
import gdown
from torch_geometric.data import InMemoryDataset


class CosmicRayDS(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['cr_sphere.npz']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        url = "https://drive.google.com/u/0/uc?export=download&confirm=HgGH&id=1XKN-Ik7BDyMWdQ230zWS2bNxXL3_9jZq"
        if os.path.exists(self.raw_file_names[0]) == False:
            gdown.download(url, self.raw_file_names[0], quiet=True)

    def process(self):
        f = np.load(self.raw_file_names[0])
        x = f['data']
        y = f['label'].astype("int")

        data_list = []
        for idx in range(len(x)):
            data_list.append(Data(x=x[idx,:,3],pos=x[idx,:,:3],y=y[idx]) )

        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]

        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])