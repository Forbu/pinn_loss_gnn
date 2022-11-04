import pytest 

from pinn_loss.pdebenchmark import BurgerPDEDataset
from pinn_loss.utils import create_graph_burger

from torch.utils.data import DataLoader

def test_burgerpdedataset():

    # we choose the discretization of the space and the time
    nb_space = 1024
    nb_time = 100

    delta_x = 1.0 / nb_space
    delta_t = 1.0 / nb_time

    # we choose the batch size
    batch_size = 1

    ############### TRAINING ################
    # now we can create the dataloader
    edges, edges_index = create_graph_burger(nb_space, delta_x, nb_nodes=None, nb_edges=None)

    path_hdf5 = "/home/data_pde_benchmark/1D_Burgers_Sols_Nu0.001.hdf5"

    # if file not exist, we valid the test
    try:
        with open(path_hdf5, "r") as f:
            pass
    except FileNotFoundError:
        assert True
        return

    # we create the dataset
    dataset = BurgerPDEDataset(path_hdf5, edges, edges_index)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # we test the dataloader
    for i, data in enumerate(dataloader):
        assert data["nodes"].shape == (batch_size, nb_space,)
        assert data["edges"].shape == (batch_size, (nb_space - 1) * 2 , 1)
        assert data["edges_index"].shape == (batch_size, (nb_space - 1) * 2, 2)
        assert data["nodes_next"].shape == (batch_size, nb_space,)

        break    