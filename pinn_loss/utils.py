"""
No clear intent with this module yet. It is a placeholder for now.
"""

import numpy as np

import torch
from torch.utils.data import DataLoader

def create_graph_burger(nb_space, delta_x, nb_nodes=None, nb_edges=None):
    """
    In the case of the burger equation, we have a 1D space : easy to create the graph (linear graph)
    """
    # create the edges features, size : ((nb_space - 1) * 2, 1)
    # the first (nb_space - 1) edges are the edges that goes from top to bottom
    # the second (nb_space - 1) edges are the edges that goes from bottom to top
    edges = np.zeros(((nb_space - 1) * 2, 1), dtype=np.float32)

    edges[:nb_space - 1, 0] = delta_x # positive edges
    edges[nb_space - 1:, 0] = -delta_x # negative edges

    # now we can create the edges_index values
    edges_index = np.zeros(((nb_space - 1) * 2, 2), dtype=np.int32)

    # we create the edges_index for the positive edges
    # the first column is the index of the node that is the source of the edge
    # the second column is the index of the node that is the target of the edge
    
    # the first (nb_space - 1) edges are the edges that goes from top to bottom
    # the second (nb_space - 1) edges are the edges that goes from bottom to top

    edges_index[:(nb_space - 1), 0] = np.arange(nb_space - 1)
    edges_index[:(nb_space - 1), 1] = np.arange(1, nb_space)

    # now we can create the edges_index for the negative edges
    edges_index[(nb_space - 1):, 0] = np.arange(1, nb_space)
    edges_index[(nb_space - 1):, 1] = np.arange(nb_space - 1)

    # we check the number of nodes and edges
    if nb_nodes is not None:
        assert nb_nodes == nb_space, "The number of nodes is not correct"
    if nb_edges is not None:
        assert nb_edges == (nb_space - 1) * 2, "The number of edges is not correct"

    return edges, edges_index



def create_burger_dataset(nb_space, delta_x, batch_size=1, size_dataset=20000):
    """
    Creation of the dataset for the burger equation
    We have to create a continuous function and the nn will have to approximate the next temporal step of the PDE
    """
    # create the space mesh
    space_mesh = np.linspace(0, 1, nb_space)

    # gausien noise
    noise = np.random.normal(0., 10., size=(size_dataset, nb_space))

    # we create the initial condition by adding a gaussian noise to create a random initial condition
    dataset = np.cumsum(noise, axis=1) * delta_x

    # now we have to create the graph associated to the dataset
    # we create the graph
    edges, edges_index = create_graph_burger(nb_space, delta_x, nb_nodes=nb_space, nb_edges=(nb_space - 1) * 2)

    class BurgerDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, edges, edges_index):
            self.dataset = dataset
            self.edges = edges
            self.edges_index = edges_index

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            return {"nodes" : self.dataset[idx], "edges" : self.edges, "edges_index" : self.edges_index}

    # we create another dataset with the some initial condition that look like a sinus
    def get_sinus_init(frequency, scale):
        initial_condition = np.sin(frequency * 2 * np.pi * space_mesh)
        return initial_condition * scale 
    
    # we generate size_dataset sinus initial condition
    # frequency are int between 1 and 5
    # scale are float between 0.5 and 5
    frequency = np.random.randint(1, 3, size=size_dataset)
    scale = np.random.uniform(0.5, 5., size=size_dataset)

    sinus_dataset = np.array([get_sinus_init(frequency[i], scale[i]) for i in range(size_dataset)])

    dataset_normal = BurgerDataset(dataset, edges, edges_index)
    dataset_sinus = BurgerDataset(sinus_dataset, edges, edges_index)

    # now we concat the two dataset
    dataset = torch.utils.data.ConcatDataset([dataset_normal, dataset_sinus])

    # create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader