'''
This module provides a set of functions for solving the Landau-Lifshitz-Gilbert equation (LGG) for a system of normalized
magnetic moments (spins) on a rectangular two-dimensional racetrack.
Furthermore, we enable the application of a localized, rotating edge field to create and propagate skyrmions along 
the given racetrack geometry.
Nearest neighbor approximation is assumed.
The calculations are performed for T = 0 K.

'''
import numpy as np
from numpy import sin, cos, pi, sqrt

ex = np.array([1,0,0])
ey = np.array([0,1,0])
ez = np.array([0,0,1])

'''
Get the indices of nearest neighbor sites for a lattice site with index 'num' on a rectangular 2d lattice.

Args:
    N:      Number of lattice sites.
    Nx:     Number of lattice sites along the x-axis.
    num:    Index of the lattice site for which the neighbor indices will be returned.

Return:
    The indices of the nearest neighbors as a list.
'''
def idx_neighbor_sites(N, Nx, num):
    return [(num+1)%N, num-1, (num+Nx)%N, num-Nx]


'''
Get the indices of the boundary sites of a rectangular 2d lattice.

Args:
    N:      Number of lattice sites.
    Nx:     Number of lattice sites along the x-axis.

Return:
    The indices of all boundary sites as a list.
'''
def idx_boundary_sites(N, Nx):

    bound_array = np.arange(0, Nx, 1)
    bound_array = np.append(bound_array, np.arange(N-Nx, N, 1))
    bound_array = np.append(bound_array, np.arange(Nx, N-Nx+1, Nx))
    bound_array = np.append(bound_array, np.arange(Nx-1, N-Nx, Nx))

    return bound_array

'''
Get the indices of all lattice sites 'i' which have less than four nearest neighbors as well as the indices of the missing nearest neighbor sites.

Args:
    N:      Number of lattice sites.
    Nx:     Number of lattice sites along the x-axis.
    bd:     The indices of the boundary sites as a list.

Return:
    The indices of all boundary sites with their missing nearest neighbor sites as a list (of tuples).
'''
def idx_zero_sides(N, Nx, bd):
    idx_zero = []

    for i in bd:
        if i < Nx:
            idx_zero.append((i,3))
        elif N-Nx <= i < N:
            idx_zero.append((i,2))
        if i%Nx == 0:
            idx_zero.append((i,1))
        elif (i+1)%Nx == 0:
            idx_zero.append((i,0))

    return np.array(idx_zero)

'''
Get the nearest neighbors for all lattice sites. Here, boundary sites underly periodic boundary conditions.

Args:
    N:  Number of lattice sites.

Return:
    The indices of all boundary sites with their missing nearest neighbor sites as a list (of tuples).
'''
def get_neighbors(N, Nx):
    neighbors = np.zeros((N, 4), dtype=int)

    for i in range(N):
        neighbors[i] = idx_neighbor_sites(N, Nx, i)

    return neighbors


'''
Get the Hamiltonian/Energy of the physical system.

Args:
    J:      Exchange interaction.
    Bz:     Homogeneous magnetic field along the z-axis.
    K:      Magnetic Out-Of-Plane Anisotropy.
    Dx:     DMI-component along the x-axis.
    Dy:     DMI-component along the y-axis.
    nset:   Spins on the rectangular racetrack.

Return:
    The Hamiltonian/Energy for a given set of parameters and spins.
'''
def get_hamilton(J, Bz, K, Dx, Dy, nset, neigh):
    e = 0

    e += -J*np.sum(np.einsum('ac, abc -> a', nset, neigh[:,[0,2]], optimize=True))

    e += -Dx*np.sum(np.cross(nset, neigh[:,2])[:,0])
    e += +Dy*np.sum(np.cross(nset, neigh[:,0])[:,1])

    e += -K*np.dot(nset[:,2], nset[:,2])

    e += -Bz*np.sum(nset[:,2])

    return e

'''
Get the derivative of the Exchange-term with respect to the spin.

Args:
    J:      Exchange interaction.
    neigh:  List of nearest neighbors.

Return:
    The derivative of the exchange term with respect to the spin.
'''
def get_derivative_exchange(J, neigh):
    return -J*np.einsum('abc -> ac', neigh, optimize=True)

'''
Get the derivative of the DMI-term with respect to the x-component of the spin.

Args:
    Dx:      DMI-component along the x-axis.
    neigh:  List of nearest neighbors.

Return:
    The derivative of the Dx-term with respect to the x-component of the spin.
'''
def get_derivative_dm_x(Dx, neigh):
    dx_array = -Dx*np.array([1,-1])
    return np.einsum('ab, b -> a', neigh, dx_array, optimize=True)

'''
Get the derivative of the DMI-term with respect to the y-component of the spin.

Args:
    Dy:      DMI-component along the y-axis.
    neigh:  List of nearest neighbors.

Return:
    The derivative of the Dy-term with respect to the y-component of the spin.
'''
def get_derivative_dm_y(Dy, neigh):
    dy_array = -Dy*np.array([1,-1])
    return np.einsum('ab, b -> a', neigh, dy_array, optimize=True)

'''
Get the derivative of the DMI-term with respect to the z-component of the spin.

Args:
    Dx:      DMI-component along the x-axis.
    Dy:      DMI-component along the y-axis.
    neigh:  List of nearest neighbors.

Return:
    The derivative of the Dy-term with respect to the z-component of the spin.
'''
def get_derivative_dm_z(Dx, Dy, neigh):
    dx_array = -Dx*np.array([1,-1])
    dy_array = -Dy*np.array([1,-1])
    return -np.einsum('ab, b -> a', neigh[:,[0,1]][:,:,0], dy_array, optimize=True)-np.einsum('ab, b -> a', neigh[:,[2,3]][:,:,1], dx_array, optimize=True)

'''
Get the effective magnetic field for each spin on the racetrack.

Args:
    mu:         The magnetic moment of each spin.
    J:          Exchange interaction.
    Bz:         Homogeneous magnetic field along the z-axis.
    K:          Magnetic Out-Of-Plane Anisotropy.
    Dx:         DMI-component along the x-axis.
    Dy:         DMI-component along the y-axis.
    N:          Number of spins on the racetrack.
    idx_zero:   List of indices where no nearest neighbor exists (at the boundary).
    size_zero:  Number of empty lattice sites.
    neighbors:  List of nearest neighbors for each spin on the racetrack.
    nset:       List of spins on the racetrack.

Return:
    The effective magnetic field for each spin on the racetrack as a list.
'''
def get_Beff(mu, J, Bz, K, Dx, Dy, N, idx_zero, size_zero, neighbors, nset):
    Beff = np.zeros((N, 3))

    nset_neigh = nset[neighbors]
    nset_neigh[idx_zero[:, 0], idx_zero[:, 1]] = np.zeros((size_zero, 3))


    Beff += get_derivative_exchange(J, nset_neigh)

    Beff[:,0] += get_derivative_dm_y(Dy, nset_neigh[:,[0,1]][:,:,2])
    Beff[:,1] += get_derivative_dm_x(Dx, nset_neigh[:,[2,3]][:,:,2])
    Beff[:,2] += get_derivative_dm_z(Dx, Dy, nset_neigh) -Bz*np.ones(N) - 2*K*nset[:,2]

    Beff *= -(1/mu)

    return Beff

'''
Get the localized rotating edge field.

Args:
    B0:             The strength of the edge field.
    v:              The rotation frequency of the edge field.
    edge_length:    Number of spins which are affected by the edge field.
    t:              The current time value.

Return:
    The localized rotating edge field for each affected spin.
'''
def apply_edge_field(B0, v, edge_length, t):
    B = -B0*np.array([-sin(v*t), 0, -cos(v*t)])
    return np.tensordot(np.ones(edge_length), B, axes=0)


'''
Get the position of a skyrmion on the racetrack.

Args:
    Nx:     Number of lattice sites along the x-axis.
    x_grid: Flattened meshgrid along the x-axis.
    y_grid: Flattened meshgrid along the y-axis.
    nset:   List of spins on the racetrack.

Return:
    The estimated position of the skyrmions center.
'''
def get_skyr_pos(Nx, x_grid, y_grid, nset):
    min_pos = np.argmin(nset[:,2], axis=0)
    args = np.array([-Nx,-1,1,Nx])
    min_square = min_pos + args

    n_sqr = nset[min_square]
    n_min = nset[min_pos]

    xmin = 0
    ymin = 0

    if np.sign(n_sqr[2][0])-np.sign(n_min[0]) != 0:
        xmin = -n_min[0]/(n_sqr[2][0] - n_min[0])
    else:
        xmin = n_min[0]/(n_sqr[1][0] - n_min[0])

    if np.sign(n_sqr[3][1])-np.sign(n_min[1]) != 0:
        ymin = -n_min[1]/(n_sqr[3][1] - n_min[1])
    else:
        ymin = n_min[1]/(n_sqr[0][1] - n_min[1])

    return np.array([x_grid[min_pos] + xmin, y_grid[min_pos] + ymin])


'''
Get the topological charge number of the spin system.

Args:
    nset:   List of spins on the racetrack.
    neigh:  List of nearest neighbor spins on the racetrack.

Return:
    The topological charge number of the spin system.
'''
def get_topological_charge(nset, neigh):
    s23 = np.einsum('ab, ab -> a', nset, neigh[:,0])
    s12 = np.einsum('ab, ab -> a', nset, neigh[:,2])
    s31 = np.einsum('ab, ab -> a', neigh[:,0], neigh[:,2])
    scross = np.einsum('ab, ab -> a', neigh[:,2], np.cross(nset, neigh[:,0]))

    s24 = np.einsum('ab, ab -> a', nset, neigh[:,1])
    s45 = np.einsum('ab, ab -> a', neigh[:,1], neigh[:,3])
    s52 = np.einsum('ab, ab -> a', nset, neigh[:,3])
    scross2 = np.einsum('ab, ab -> a', nset, np.cross(neigh[:,1], neigh[:,3]))

    p1 = sqrt(2*(1+s12)*(1+s23)*(1+s31))
    exp_omega1 = (1+s12+s23+s31+1j*scross)/p1
    omega1 = 2*np.imag(np.log(exp_omega1))

    p2 = sqrt(2*(1+s24)*(1+s45)*(1+s52))
    exp_omega2 = (1+s24+s45+s52+1j*scross2)/p2
    omega2 = 2*np.imag(np.log(exp_omega2))

    return np.sum(omega1+omega2)/(4*pi)

'''
Get the right hand side of the LLG-equation to determine the time-evolution of the spins on the racetrack.

Args:
    gamma:  Gyromagnetic ratio.
    alpha:  Damping constant.
    n_t:    List of spins on the racetrack at time value t.
    Beff:   List of local effective magnetic fields at time value t.

Return:
    The right hand side of the LLG-equation.
'''
def get_derivative(gamma, alpha, n_t, Beff):
    n_x_Beff = np.cross(n_t, Beff)
    return -gamma*(n_x_Beff + alpha*np.cross(n_t, n_x_Beff))/(1+alpha**2)