import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, HeterophilousGraphDataset, WikiCS, AttributedGraphDataset, CitationFull
from ogb.nodeproppred import NodePropPredDataset

import numpy as np
import scipy.sparse as sp
from os import path
#from google_drive_downloader import GoogleDriveDownloader as gdd
import gdown
import scipy
from torch_geometric.datasets import Planetoid
from data_utils import dataset_drive_url, rand_train_test_idx


class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='auto', train_prop=0.5, valid_prop=0.25):
        """
        Automatically determine train/val/test split based on dataset name or given proportion.
        """
        labels = self.label
        name = self.name.lower()

        if split_type == 'auto':
            if name == 'dblp':
                train_percent = 0.01
                val_percent = 0.15
            elif name in ['blogcatalog', 'flickr']:
                train_percent = 0.05
                val_percent = 0.15
            else:
                train_percent = train_prop
                val_percent = valid_prop

            test_percent = 1.0 - train_percent - val_percent
            num_nodes = len(labels)

            train_size = int(train_percent * num_nodes)
            val_size = int(val_percent * num_nodes)
            test_size = num_nodes - train_size - val_size

            train_idx, valid_idx, test_idx = get_split(
                labels.cpu().numpy(),
                train_size=train_size,
                val_size=val_size,
                test_size=test_size
            )
        else:
            # default to fully random split if requested
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                labels, train_prop=train_prop, valid_prop=valid_prop
            )

        return {
            'train': torch.tensor(train_idx, dtype=torch.long),
            'valid': torch.tensor(valid_idx, dtype=torch.long),
            'test': torch.tensor(test_idx, dtype=torch.long)
        }

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_dataset(data_dir, dataname, sub_dataname=''):
    """ Loader for NCDataset
        Returns NCDataset
    """
    print(dataname)
    if dataname in  ('amazon-photo', 'amazon-computer'):
        dataset = load_amazon_dataset(data_dir, dataname)
    elif dataname in  ('coauthor-cs', 'coauthor-physics'):
        dataset = load_coauthor_dataset(data_dir, dataname)
    elif dataname in ('roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions'):
        dataset = load_hetero_dataset(data_dir, dataname)
    elif dataname == 'wikics':
        dataset = load_wikics_dataset(data_dir)
    elif dataname in ('ogbn-arxiv', 'ogbn-products'):
        dataset = load_ogb_dataset(data_dir, dataname)
    elif dataname == 'pokec':
        dataset = load_pokec_mat(data_dir)
    elif dataname in ('cora', 'citeseer', 'pubmed'):
        dataset = load_planetoid_dataset(data_dir, dataname)
    elif dataname in ('chameleon', 'squirrel'):
        dataset = load_wiki_new(data_dir, dataname)
    elif dataname in ('blogcatalog', 'flickr'):
        dataset = load_attributed_dataset(data_dir, dataname)
    elif dataname == 'dblp':
        dataset = load_citation_full_dataset(data_dir, dataname)
    else:
        raise ValueError('Invalid dataname')
    return dataset


def load_planetoid_dataset(data_dir, name, no_feat_norm=True):
    if not no_feat_norm:
        transform = T.NormalizeFeatures()
        torch_dataset = Planetoid(root=f'{data_dir}/Planetoid',
                                  name=name, transform=transform)
    else:
        torch_dataset = Planetoid(root=f'{data_dir}/Planetoid', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes
    print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset


def load_wiki_new(data_dir, name):
    path= f'{data_dir}/geom-gcn/{name}/{name}_filtered.npz'
    data=np.load(path)
    # lst=data.files
    # for item in lst:
    #     print(item)
    node_feat=data['node_features'] # unnormalized
    labels=data['node_labels']
    edges=data['edges'] #(E, 2)
    edge_index=edges.T

    dataset = NCDataset(name)

    edge_index=torch.as_tensor(edge_index)
    node_feat=torch.as_tensor(node_feat)
    labels=torch.as_tensor(labels)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': node_feat.shape[0]}
    dataset.label = labels

    return dataset

def load_wikics_dataset(data_dir):
    wikics_dataset = WikiCS(root=f'{data_dir}/wikics/')
    data = wikics_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset('wikics')
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset

def load_hetero_dataset(data_dir, name):
    #transform = T.NormalizeFeatures()
    torch_dataset = HeterophilousGraphDataset(name=name.capitalize(), root=data_dir)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    ## dataset splits are implemented in data_utils.py
    '''
    dataset.train_idx = torch.where(data.train_mask[:,0])[0]
    dataset.valid_idx = torch.where(data.val_mask[:,0])[0]
    dataset.test_idx = torch.where(data.test_mask[:,0])[0]
    '''

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset


def load_amazon_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == 'amazon-photo':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                                 name='Photo', transform=transform)
    elif name == 'amazon-computer':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                                 name='Computers', transform=transform)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset

def load_coauthor_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == 'coauthor-cs':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor',
                                 name='CS', transform=transform)
    elif name == 'coauthor-physics':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor',
                                 name='Physics', transform=transform)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset

def load_ogb_dataset(data_dir, name):
    dataset = NCDataset(name)
    ogb_dataset = NodePropPredDataset(name=name, root=f'{data_dir}/ogb')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    def ogb_idx_to_tensor():
        split_idx = ogb_dataset.get_idx_split()
        tensor_split_idx = {key: torch.as_tensor(
            split_idx[key]) for key in split_idx}
        return tensor_split_idx
    dataset.load_fixed_splits = ogb_idx_to_tensor  # ogb_dataset.get_idx_split
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    return dataset

def load_pokec_mat(data_dir):
    """ requires pokec.mat """
    if not path.exists(f'{data_dir}/pokec/pokec.mat'):
        drive_id = '1575QYJwJlj7AWuOKMlwVmMz8FcslUncu'
        gdown.download(id=drive_id, output="data/pokec/")
        #import sys; sys.exit()
        #gdd.download_file_from_google_drive(
        #    file_id= drive_id, \
        #    dest_path=f'{data_dir}/pokec/pokec.mat', showsize=True)

    fulldata = scipy.io.loadmat(f'{data_dir}/pokec/pokec.mat')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)
    return dataset


def load_attributed_dataset(data_dir, name):
    dataset = AttributedGraphDataset(root=data_dir, name=name)
    data = dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    if hasattr(node_feat, "to_dense"):
        node_feat = node_feat.to_dense()  # flickr 特征为稀疏
    label = data.y
    num_nodes = data.num_nodes

    nc_dataset = NCDataset(name)
    nc_dataset.graph = {'edge_index': edge_index,
                        'node_feat': node_feat,
                        'edge_feat': None,
                        'num_nodes': num_nodes}
    nc_dataset.label = label

    return nc_dataset


def load_citation_full_dataset(data_dir, name):
    dataset = CitationFull(root=data_dir, name='DBLP')
    data = dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    nc_dataset = NCDataset(name)
    nc_dataset.graph = {'edge_index': edge_index,
                        'node_feat': node_feat,
                        'edge_feat': None,
                        'num_nodes': num_nodes}
    nc_dataset.label = label

    return nc_dataset

def get_split(labels, train_examples_per_class=None, val_examples_per_class=None, test_examples_per_class=None,
              train_size=None, val_size=None, test_size=None):
    num_samples = len(labels)
    num_classes = labels.max() + 1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(labels, train_examples_per_class)
    else:
        train_indices = np.random.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = np.random.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))

    if test_examples_per_class is not None:
        test_indices = sample_per_class(labels, test_examples_per_class, forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = np.random.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    return train_indices, val_indices, test_indices


def sample_per_class(labels, num_examples_per_class, forbidden_indices=None):
    num_samples = len(labels)
    num_classes = labels.max() + 1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    return np.concatenate(
        [np.random.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))])
