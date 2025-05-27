'''
Obtaining the time-dependent trajetory of a skyrmion created by a 2 pi rotation of a localized rotating edge field 'B_edge'.
After the 2 pi rotation, we turn the edge field off.

The magnetic interaction parameters as well as the racetrack parameters are initialized at the beginning of the script file, 
right below the import-section.
'''
import lib_physics as ps
import numpy as np
from numpy import pi, sqrt
import pandas as pd
from functools import partial

# Initializing the parameters of the system.
Nx = 4*30
Ny = 4*7
N = Nx*Ny
xlins = np.linspace(0, Nx, Nx, endpoint=False)
ylins = np.linspace(0, Ny, Ny, endpoint=False)
xcart, ycart = np.meshgrid(xlins, ylins)
grid = xcart + 1j*ycart

ex = ps.ex
ey = ps.ey
ez = ps.ez

meV = 10**(-3)
mu = 3*5.79*10**(-5)
J = 11.6*meV
Bz = 1.5*mu
K = 0.35*meV
Dx = 3.17*meV
Dy = 3.17*meV
gamma = 1
alpha = 0.1

# Initial ferromagnetic (FM) spin configuration.
n_init = np.tensordot(np.ones(N), ez, axes=0)

# Set the boundary-, neighbor- and empty-sites.
boundaries = ps.idx_boundary_sites(N, Nx)
neighbors = ps.get_neighbors(N, Nx)
idx_zero = ps.idx_zero_sides(N, Nx, boundaries)
size_zero = len(idx_zero)

# Reduction of the number of function arguments via the 'partial' class from the 'functools' library.
get_Beff = partial(ps.get_Beff, mu, J, Bz, K, Dx, Dy, N, idx_zero, size_zero, neighbors)
get_derivative = partial(ps.get_derivative, gamma, alpha)
get_skyr_pos = partial(ps.get_skyr_pos, Nx, xcart.flatten(), ycart.flatten())

# Specification of the edge field region.
dsize=7
idx_edges = np.arange(0+dsize*Nx, N-dsize*Nx, Nx, dtype=int)
num_edges = 5
for i in range(1, num_edges):
    idx_edges = np.concatenate((idx_edges, np.arange(i+dsize*Nx, i+N-dsize*Nx, Nx, dtype=int)))
edge_set = idx_edges

# Initial relaxation of the system.
n_t = np.copy(n_init)
dt = 0.00176
t = 0

for i in range(10000):
    Beff = get_Beff(n_t)
    K1 = get_derivative(n_t, Beff)

    Beff = get_Beff(n_t + dt*K1)
    K2 = get_derivative(n_t + dt*K1, Beff)

    n_t = n_t + (dt/2)*(K1+K2)

    n_t = np.einsum('ab, a -> ab', n_t, sqrt(1/np.einsum('ab, ab -> a', n_t, n_t)))
    t += dt

n_t_init = np.copy(n_t)

# Declare the skyrmion positions and measure times each as a list.
distances = []
times = []

idx_edge = edge_set
edge_length = len(idx_edge)

# Setting the edge field amplitude Bj and frequency v.
Bj = 10
v = 1*0.357

apply_edge_field = partial(ps.apply_edge_field, Bj, v)

t = 0
n_t = n_t_init

# Apply the rotated edge field.
for k in range(8000):
    Beff = get_Beff(n_t)
    Beff[idx_edge] += apply_edge_field(edge_length, t)
    K1 = get_derivative(n_t, Beff)

    Beff = get_Beff(n_t + dt*K1)
    Beff[idx_edge] += apply_edge_field(edge_length, t + dt)
    K2 = get_derivative(n_t + dt*K1, Beff)

    n_t = n_t + (dt/2)*(K1+K2)

    n_t = np.einsum('ab, a -> ab', n_t, sqrt(1/np.einsum('ab, ab -> a', n_t, n_t)))
    t += dt

# Apply the rotated edge field and measure the time-dependent skyrmion location.
for k in range(2000):
    Beff = get_Beff(n_t)
    Beff[idx_edge] += apply_edge_field(edge_length, t)
    K1 = get_derivative(n_t, Beff)

    Beff = get_Beff(n_t + dt*K1)
    Beff[idx_edge] += apply_edge_field(edge_length, t + dt)
    K2 = get_derivative(n_t + dt*K1, Beff)

    n_t = n_t + (dt/2)*(K1+K2)

    n_t = np.einsum('ab, a -> ab', n_t, sqrt(1/np.einsum('ab, ab -> a', n_t, n_t)))
    t += dt

    if k%1000 == 0:
        d = list(get_skyr_pos(n_t))
        distances.append(d)
        times.append(k*0.01)

# Deactivate the edge field and let the skyrmion propagate while measuring its' trajectory.
for k in range(2000, 202000):
    Beff = get_Beff(n_t)
    K1 = get_derivative(n_t, Beff)

    Beff = get_Beff(n_t + dt*K1)
    K2 = get_derivative(n_t + dt*K1, Beff)

    n_t = n_t + (dt/2)*(K1+K2)

    n_t = np.einsum('ab, a -> ab', n_t, sqrt(1/np.einsum('ab, ab -> a', n_t, n_t)))
    t += dt

    if k%1000 == 0:
        d = list(get_skyr_pos(n_t))
        distances.append(d)
        times.append(k*0.01)

# Save the trajectory data in a .csv-file.
df = pd.DataFrame(distances, columns = ['x','y'], index = times)
df.to_csv('trajectory.csv')