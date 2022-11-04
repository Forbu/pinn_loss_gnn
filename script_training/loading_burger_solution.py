import h5py
import numpy as np

import matplotlib.pyplot as plt

# read the file in data_pde_benchmark/1D_Burgers_Sols_Nu0.001.hdf5
with h5py.File("data_pde_benchmark/1D_Burgers_Sols_Nu0.001.hdf5", "r") as f:
    # we print the dataset key
    print(f.keys())
    print(f['tensor'])
    print(f['t-coordinate'])
    print(f['x-coordinate'])


    # access to the data and the coordinates and covert the whole thing to numpy
    # data is a numpy array of shape (10000, 201, 1024)
    data = np.array(f['tensor'][2, :, :])
    print(data.shape)

    # t is a numpy array of shape (202,)
    t = np.array(f['t-coordinate'])
    print(t.shape)

    # x is a numpy array of shape (1024,)
    x = np.array(f['x-coordinate'])
    print(x.shape)

    print(x)

    # plot and save data
    plt.figure(figsize=(50, 10))
    plt.imshow(data)

    # adding x and t coordinates
    plt.xlabel("x")
    plt.ylabel("t")

    # adding x and t coordinates
    plt.xticks(np.arange(0, 1024, 100), x[::100])
    plt.yticks(np.arange(0, 201, 20), t[::20])

    # adding a colorbar
    plt.colorbar()

    plt.savefig("data_pde_benchmark/1D_Burgers_Sols_Nu0.001.png")

# read the data for the first time step to see the initial condition
limit_condition = data[0, :]
final_condition = data[-1, :]

# plot and save the initial condition
plt.figure(figsize=(50, 10))
plt.plot(limit_condition)
plt.plot(final_condition)

plt.show()
plt.savefig("data_pde_benchmark/1D_Burgers_Sols_Nu0.001_initial_condition.png")

# we can also plot the condition for the x limit
x_limit_up = data[:, 0]
x_limit_down = data[:, -1]

# plot and save the initial condition
plt.figure(figsize=(50, 10))
plt.plot(x_limit_up)
plt.plot(x_limit_down)

plt.show()
plt.savefig("data_pde_benchmark/1D_Burgers_Sols_Nu0.001_x_limit_condition.png")



