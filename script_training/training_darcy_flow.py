"""
This is the main file to train the GNN on the darcy flow equation

The darcy flow equation is a simple PDE that can be solved with a GNN (and a pinn loss).
The PDE is :
    -div(k*grad(u)) = f

With the following boundary conditions:
    u(0, t) = 0
    u(1, t) = 0
    u(x, 0) = sin(pi*x)

The goal is to train a GNN to solve this PDE.

"""