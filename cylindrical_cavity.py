"""
Created on Wed Mar 13 12:54:16 2025

@author: elisabm99
"""

# Import necessary libraries
from skfem import *
import numpy as np
from scipy.special import yv, iv, yn, jn, kv

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker

# Configure NumPy print options for better readability
np.set_printoptions(precision=10, suppress=False, floatmode='fixed')

# --- Plotting and Visualization Configuration ---
# Set up plotting style using matplotlib
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'  # Enable siunitx LaTeX package
plt.rcParams['axes.linewidth'] = 2
plt.rc('text', usetex=True)  # Use LaTeX for all text
plt.rc('font', family='ComputerModern')  # Set font to Computer Modern
plt.rcParams['axes.linewidth'] = 1.5
lines = ["-", "--", "-.", ":"]

# --- Physical Parameters and Geometry Definition ---
# Note: alpha10 is a physical constant scaled by 10^-10 GeV^-2.
# 'kappa_ref' is a reference length scale, 1/sqrt(alpha10 * rho), for a specific alpha10 and aluminum density.
kappa_ref = 0.005784597488223868  # meters. 1 / np.sqrt(alpha10 rho) for alpha10 = 10^-10 GeV^-2 and aluminum density 2700 kg m^-3

# Inner dimensions of the cylindrical cavity
R = 0.1  # meters. Cylinder inner radius
dR = 0.01  # meters. Thickness of the cavity wall
L = 0.1  # meters. Half cylinder inner height
# L = 2.5 # meters. Half cylinder inner height (alternate value)
dL = 0.01  # meters. Thickness of the cavity wall

# Outer dimensions of the computational box
Zmax = 10 * (L + dL)  # meters
Rmax = 10 * (R + dR)  # meters
Npoint_R = 21  # Number of points in R direction for initial mesh
Npoint_Z = 41  # Number of points in Z direction for initial mesh

# Parameters for mesh refinement
Z_refine = 5 * (L + dL)  # meters. Z-extent of the initial refinement region
R_refine = 5 * (R + dR)  # meters. R-extent of the initial refinement region
epsilon = 1.2 * dL  # Tolerance for identifying corner regions for extra refinement

# --- Mesh Refinement Levels ---
N_refine = 2  # Initial uniform refinement level of the mesh
N_refine_additional = 3  # Number of additional refinements in the region of interest (inner region)
N_refine_additional_additional = 3  # Additional refinements around the cavity walls
N_refine_additional_additional_additional = 0  # Additional refinements in the cavity corners


###############################################################
###############################################################
###############################################################

# --- Core Functions for Simulation Setup and Solution ---
def get_dimless_sizes(kappa, myL):
    """
    Calculates dimensionless lengths by scaling physical lengths by 'kappa'.

    Args:
        kappa (float): The length scale of the problem (meters).
        myL (float): The half height of the cylinder (meters).

    Returns:
        tuple: A tuple containing the dimensionless R, dR, myL, dL, Rmax, and Zmax.
    """
    return R / kappa, dR / kappa, myL / kappa, dL / kappa, Rmax / kappa, Zmax / kappa


def rho(r, z, R_dimless, dR_dimless, L_dimless, dL_dimless):
    """
    Defines the matter density for a cylindrical cavity.
    The density is 1 within the cavity walls and 0 outside.
    The actual value of the density is absorbed in kappa

    Args:
        r (ndarray): Radial coordinates.
        z (ndarray): Axial coordinates.
        R_dimless (float): Dimensionless inner radius.
        dR_dimless (float): Dimensionless wall thickness.
        L_dimless (float): Dimensionless half height.
        dL_dimless (float): Dimensionless wall thickness.

    Returns:
        ndarray: An array representing the matter density at each point.
    """
    # Uses the Heaviside step function to define the cylindrical shell
    return (np.heaviside(R_dimless - r, 0) * np.heaviside(np.abs(z) - L_dimless, 0
                                                          ) + np.heaviside(r - R_dimless, 0) * np.heaviside(
        R_dimless + dR_dimless - r, 0)) * np.heaviside(L_dimless + dL_dimless - np.abs(z), 0)


def rho_sph(r, z, R_dimless, dR_dimless):
    """
    Defines the matter density for a spherical cavity.

    Args:
        r (ndarray): Radial coordinates.
        z (ndarray): Axial coordinates.
        R_dimless (float): Dimensionless inner radius.
        dR_dimless (float): Dimensionless wall thickness.

    Returns:
        ndarray: An array representing the matter density at each point.
    """
    r_sph = np.sqrt(r ** 2 + z ** 2)
    # Uses Heaviside functions to define a spherical shell
    return np.heaviside(r_sph - R_dimless, 0) * np.heaviside(R_dimless + dR_dimless - r_sph, 0)


def init_mesh(kappa):
    """
    Initializes a triangular mesh for the computational domain.

    Args:
        kappa (float): The length scale for creating dimensionless coordinates.

    Returns:
        MeshTri: An initial, uniformly refined triangular mesh.
    """
    R_dimless, dR_dimless, L_dimless, dL_dimless, Rmax_dimless, Zmax_dimless = get_dimless_sizes(kappa, L)
    rs = np.linspace(0, Rmax_dimless, Npoint_R)
    zs = np.linspace(-Zmax_dimless, Zmax_dimless, Npoint_Z)
    
    mesh = MeshTri().init_tensor(rs, zs)
    mesh = mesh.refined(N_refine)
    
    return mesh


def get_centroids(m):
    """
    Computes the centroids of each triangular element in the mesh.

    Args:
        m (MeshTri): The mesh object.

    Returns:
        tuple: x and y coordinates of the centroids.
    """
    centroid_x = m.p[0, m.t].mean(axis=0)  # x-coordinate of triangle centroids
    centroid_y = m.p[1, m.t].mean(axis=0)  # y-coordinate of triangle centroids
    
    return centroid_x, centroid_y


def refinement_region_cyl(m, kappa):
    """
    Returns a boolean mask for mesh elements within the main refinement region for a cylinder.

    Args:
        m (MeshTri): The mesh object.
        kappa (float): The length scale.

    Returns:
        ndarray: A boolean array indicating which elements to refine.
    """
    Z_refine_dimless = Z_refine / kappa
    R_refine_dimless = R_refine / kappa
    
    centroid_x, centroid_y = get_centroids(m)
    return ((centroid_x <= R_refine_dimless) & (centroid_y <= Z_refine_dimless)) & (centroid_y >= -Z_refine_dimless)


def refinement_region2_cyl(m, kappa):
    """
    Returns a boolean mask for mesh elements around the cylindrical cavity walls.

    Args:
        m (MeshTri): The mesh object.
        kappa (float): The length scale.

    Returns:
        ndarray: A boolean array indicating which elements to refine.
    """
    centroid_x, centroid_y = get_centroids(m)
    condition = (
        ((centroid_x <= 0.8 * R / kappa) & (np.abs(centroid_y) >= 0.8 * L / kappa) & (
                    np.abs(centroid_y) <= 1.2 * (L + dL) / kappa) |
         ((centroid_x >= 0.8 * R / kappa) & (centroid_x <= 1.2 * (R + dR) / kappa) & (
                     np.abs(centroid_y) <= 1.2 * (L + dL) / kappa)))
    )
    return condition


def refinement_region3_cyl(m, kappa):
    """
    Returns a boolean mask for mesh elements near the corners of the cylindrical cavity.

    Args:
        m (MeshTri): The mesh object.
        kappa (float): The length scale.

    Returns:
        ndarray: A boolean array indicating which elements to refine.
    """
    centroid_x, centroid_y = get_centroids(m)
    epsilon_ = epsilon / kappa
    
    # Define the corner regions
    condition = (
            (
                    (np.abs(centroid_x - 0) <= epsilon_) &
                    (np.abs(centroid_y - (-(L + dL / 2) / kappa)) <= epsilon_)  # bottom right "corner"
            ) | (
                    (np.abs(centroid_x - 0) <= epsilon_) &
                    (np.abs(centroid_y - (L + dL / 2) / kappa) <= epsilon_)  # top right "corner"
            ) | (
                    (np.abs(centroid_x - (R + dR / 2) / kappa) <= epsilon_) &
                    (np.abs(centroid_y - (-(L + dL / 2) / kappa)) <= epsilon_)  # bottom left corner
            ) | (
                    (np.abs(centroid_x - (R + dR / 2)) <= epsilon_) &
                    (np.abs(centroid_y - (L + dL / 2) / kappa) <= epsilon_)  # top left corner
            ))
    return condition


def refinement_region_sph(m, kappa):
    """
    Returns a boolean mask for mesh elements within the main refinement region for a sphere.

    Args:
        m (MeshTri): The mesh object.
        kappa (float): The length scale.

    Returns:
        ndarray: A boolean array indicating which elements to refine.
    """
    R_refine_dimless = R_refine / kappa
    
    centroid_x, centroid_y = get_centroids(m)
    centroid_r_sph = np.sqrt(centroid_x ** 2 + centroid_y ** 2)
    return centroid_r_sph <= R_refine_dimless


def refinement_region2_sph(m, kappa):
    """
    Returns a boolean mask for mesh elements around the spherical cavity walls.

    Args:
        m (MeshTri): The mesh object.
        kappa (float): The length scale.

    Returns:
        ndarray: A boolean array indicating which elements to refine.
    """
    centroid_x, centroid_y = get_centroids(m)
    centroid_r_sph = np.sqrt(centroid_x ** 2 + centroid_y ** 2)
    condition = (centroid_r_sph >= 0.8 * R / kappa) & (centroid_r_sph <= 1.2 * R / kappa)
    return condition


def refine_mesh_in_region(m, kappa, N, mask_function):
    """
    Performs a specified number of mesh refinements within a region defined by a mask.

    Args:
        m (MeshTri): The initial mesh object.
        kappa (float): The length scale.
        N (int): The number of refinements to perform.
        mask_function (function): A function that returns a boolean mask for refinement.

    Returns:
        MeshTri: The refined mesh object.
    """
    for i in range(N):
        refine_mask = mask_function(m, kappa)
        m = m.refined(refine_mask)
    
    return m


def declare_named_boundaries(m, kappa):
    """
    Defines named boundaries for the mesh, which are used to apply boundary conditions.

    Args:
        m (MeshTri): The mesh object.
        kappa (float): The length scale.

    Returns:
        MeshTri: The mesh object with named boundaries.
    """
    R_dimless, dR_dimless, L_dimless, dL_dimless, Rmax_dimless, Zmax_dimless = get_dimless_sizes(kappa, L)
    
    return m.with_boundaries({'top': lambda x: x[1] == Zmax_dimless, 'bottom': lambda x: x[1] == -Zmax_dimless,
                              'right': lambda x: x[0] == Rmax_dimless})


def get_solution(kappa, isAlphaPositive=True, isCylinder=True):
    """
    Solves the PDE using the Finite Element Method and returns the solution.

    Args:
        kappa (float): The length scale parameter.
        isAlphaPositive (bool): True for positive alpha, False for negative.
        isCylinder (bool): True for a cylindrical cavity, False for a spherical one.

    Returns:
        tuple: A tuple containing the solution vector 'x' and the final mesh 'mesh'.
    """
    R_dimless, dR_dimless, L_dimless, dL_dimless, Rmax_dimless, Zmax_dimless = get_dimless_sizes(kappa, L)
    
    mesh_ini = init_mesh(kappa)
    if isCylinder:
        mesh = refine_mesh_in_region(mesh_ini, kappa, N_refine_additional, refinement_region_cyl)
        mesh = refine_mesh_in_region(mesh, kappa, N_refine_additional_additional, refinement_region2_cyl)
        mesh = refine_mesh_in_region(mesh, kappa, N_refine_additional_additional_additional, refinement_region3_cyl)
    else:
        mesh = refine_mesh_in_region(mesh_ini, kappa, N_refine_additional, refinement_region_sph)
        mesh = refine_mesh_in_region(mesh, kappa, N_refine_additional_additional, refinement_region2_sph)
    
    mesh = declare_named_boundaries(mesh, kappa)
    
    basis = Basis(mesh, ElementTriP1())
    
    pm = 1 if isAlphaPositive else -1
    
    @BilinearForm
    def laplace(u, v, w):
        """ Defines the bilinear form for the PDE in cylindrical coordinates. """
        r, z = w.x  # cylindrical coordinates
        ur, uz = u.grad
        vr, vz = v.grad
        if isCylinder:
            # The PDE is (1/r) * d/dr(r * d phi/dr) + d^2 phi / dz^2 + pm * rho * phi = 0
            # which is transformed into the weak form for the FEM solver.
            lapl = - (ur * vr + uz * vz) * r - pm * r * rho(r, z, R_dimless, dR_dimless, L_dimless, dL_dimless) * u * v
        else:
            # Similar weak form for spherical coordinates
            r2_sph = r ** 2 + z ** 2  # spherical radial coordinate
            r_sph = np.sqrt(r2_sph)
            du_dr_sph = (r * ur + z * uz) / r_sph
            lapl = - 1 / r_sph * du_dr_sph * (r * vr + z * vz - v) - pm * rho_sph(r, z, R_dimless, dR_dimless) * u * v
        
        return lapl
    
    @LinearForm
    def rhs(v, _):
        """ Defines the right-hand side of the PDE (which is zero). """
        return 0. * v
    
    # Assemble the system matrix (A) and right-hand side vector (b)
    A = laplace.assemble(basis)
    b = rhs.assemble(basis)
    
    # Apply Dirichlet boundary conditions
    u = basis.zeros()
    u[basis.get_dofs({'top', 'right', 'bottom'})] = 1.  # Set value to 1 at boundaries
    A, b = enforce(A, b, D=basis.get_dofs({'top', 'right', 'bottom'}), x=u)
    
    # Solve the linear system A * x = b
    x = solve(A, b)
    
    return x, mesh


def find_nearest(array, value):
    """ Finds the index of the element in an array closest to a given value. """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_value_at_origin(x, mymesh, mykappa, myL):
    """
    Returns the solution value at the origin and the average value inside the cavity.

    Args:
        x (ndarray): The solution vector.
        mymesh (MeshTri): The mesh object.
        mykappa (float): The length scale.
        myL (float): The half height of the cavity.

    Returns:
        tuple: The solution value at the origin and the average solution inside the cavity.
    """
    idalpha0 = np.where((mymesh.p[0] <= R / mykappa) & (np.abs(mymesh.p[1]) <= myL / mykappa))
    idalpha = (np.isclose(mymesh.p[1], 0)) & (np.isclose(mymesh.p[0], 0))
    ii = np.where(idalpha == True)[0][0]
    return x[ii], np.mean(x[idalpha0])


def get_center_values(isAlphaPositive, isCylinder):
    """
    Iterates through a range of alpha10 values, solves the PDE, and records
    the field value at the center of the cavity. It also saves the data and checks for peaks.

    Args:
        isAlphaPositive (bool): True for positive alpha, False for negative.
        isCylinder (bool): True for a cylindrical cavity, False for a spherical one.
    """
    if L == 2.5:
        alpha10s = np.concatenate((np.logspace(-4, -2, num=50), np.logspace(-2, np.log10(1e-1), num=2000)))
    else:
        alpha10s = np.concatenate((np.logspace(-4, -2, num=50), np.logspace(-2, 0, num=2000)))
    
    kappas = kappa_ref / np.sqrt(alpha10s)
    center_values = np.empty([len(kappas), 2])
    
    peak_indices = []
    peak_values = []
    peak_data = []
    
    # Buffers to detect local maxima (peaks)
    prev_prev_val = None
    prev_idx = None
    prev_sign = 1
    prev_val = None
    curr_idx = None
    curr_sign = None
    
    for i, kappa in enumerate(kappas):
        x, m = get_solution(kappa, isAlphaPositive, isCylinder)
        center_values[i] = get_value_at_origin(x, m, kappa, L)
        print(i, alpha10s[i], center_values[i])
        abs_val = abs(center_values[i, 0])
        curr_sign = np.sign(center_values[i, 0])
        
        # Check for local maxima (peaks)
        if prev_val is not None and prev_prev_val is not None:
            if prev_val > prev_prev_val and prev_val > abs_val and prev_val / prev_prev_val - 1 > 5e-2:
                print(f"Local maximum at index {curr_idx}, value = {prev_val}")
                print('curr_idx,  alpha10s[curr_idx] =', curr_idx, alpha10s[curr_idx])
                peak_indices.append(curr_idx)
                peak_values.append(prev_val * prev_sign)
                # Save the mesh and solution at the peak
                prev_mesh.save(
                    'solutions/solution_isAlphaPositive=%s_isCylinder=%s_L=%.1f_alpha = %.4e.msh' % (False, True, 2 * L,
                                                                                                     alpha10s[
                                                                                                         curr_idx] * 1e-10),
                    point_data={'solution': prev_sol}, encode_point_data=True)
        
        # Update buffers
        prev_prev_val = prev_val
        prev_idx = curr_idx
        prev_sign = curr_sign
        prev_sol = x
        prev_mesh = m
        prev_val = abs_val
        curr_idx = i
    
    peak_indices = np.array(peak_indices)
    peak_values = np.array(peak_values)
    
    if not isAlphaPositive:
        print(peak_indices)
        print(peak_values)
        print(alpha10s[peak_indices])
        np.savez('peaks_L=%.1f.npz' % (2 * L), peak_indices=peak_indices, peak_values=peak_values, alpha10s=alpha10s)
    
    # Save the results to a text file
    np.savetxt('center_values_isAlphaPositive=%s_isCylinder=%s_L=%.1f.txt' % (isAlphaPositive, isCylinder, 2 * L),
               np.c_[alpha10s * 1e-10, center_values[:, 0], center_values[:, 1]], fmt=['%.8e', '%.8e', '%.8e'],
               delimiter='\t\t', header='alpha [GeV^-2] \t phi_center / phi_infty \t phi_inside / phi_infty')


def get_center_value_infinite_cyl(alpha10, RMax, isAlphaPositive):
    """
    Calculates the analytical solution for the field at the center of an infinite cylinder.

    Args:
        alpha10 (ndarray): Array of alpha10 values.
        RMax (float): The outer radius of the computational domain.
        isAlphaPositive (bool): True for positive alpha, False for negative.

    Returns:
        ndarray: The analytical field values.
    """
    if isAlphaPositive:
        # Solution using Modified Bessel functions for alpha > 0
        kappa = kappa_ref / np.sqrt(alpha10)  # meter
        R1 = R
        R2 = R + dR
        z1 = R1 / kappa
        z2 = R2 / kappa
        log_term = np.log(RMax / R2)
        
        term1 = kappa * iv(0, z2)
        term2 = R2 * iv(1, z2) * log_term
        term3 = kappa * kv(0, z2)
        term4 = R2 * kv(1, z2) * log_term
        
        denominator = (
                R1 * kv(1, z1) * (term1 + term2) +
                R1 * iv(1, z1) * (term3 - term4)
        )
        
        return kappa ** 2 / denominator
    
    else:
        # Solution using Bessel functions for alpha < 0
        kappa = kappa_ref / np.sqrt(alpha10)  # meter
        R1 = R
        R2 = R + dR
        term1 = yn(1, R1 / kappa) * (-kappa * jn(0, R2 / kappa) + R2 * jn(1, R2 / kappa) * np.log(RMax / R2))
        term2 = jn(1, R1 / kappa) * (kappa * yn(0, R2 / kappa) - R2 * yn(1, R2 / kappa) * np.log(RMax / R2))
        denominator = np.pi * R1 * (term1 + term2)
        return (2 * kappa ** 2) / denominator


def get_center_value_sphere_analytical(alpha10, isAlphaPositive):
    """
    Calculates the analytical solution for the field at the center of a sphere.

    Args:
        alpha10 (ndarray): Array of alpha10 values.
        isAlphaPositive (bool): True for positive alpha, False for negative.

    Returns:
        ndarray: The analytical field values.
    """
    kappa = kappa_ref / np.sqrt(alpha10)  # meter
    
    if isAlphaPositive:
        # Solution for alpha > 0
        denominator = (R / kappa + 1) * np.exp(dR / kappa) - (R / kappa - 1) * np.exp(-dR / kappa)
        return 2 / denominator
    
    else:
        # Solution for alpha < 0
        denominator = np.cos(dR / kappa) - R / kappa * np.sin(dR / kappa)
        return 1 / denominator


##################################################################
#                        Plotting Functions                      #
##################################################################
def do_cosmetics(ax, labelsize):
    """
    Applies cosmetic settings to a matplotlib axis object.

    Args:
        ax (matplotlib.axes.Axes): The axis to format.
        labelsize (int): Font size for the labels.
    """
    ax.tick_params(which='major', direction='in', width=1, length=5, top=True, right=True, pad=10)
    ax.tick_params(which='minor', direction='in', width=1, length=3, top=True, right=True, pad=10)
    ax.tick_params(axis='both', labelsize=labelsize)
    ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def plot_center_values():
    """
    Generates a 2x2 panel plot comparing the numerical solutions to analytical solutions
    for both long and short cylinders and a sphere, for positive and negative alpha.
    """
    isCylinder = True
    fontsize = 18
    labelsize = 18
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axs = axs.flatten()
    
    # Plot analytical solutions for infinite cylinder and sphere
    alpha10s = np.logspace(-4, 0, num=2000)
    alphas = alpha10s * 1e-10
    
    axs[0].plot(alphas, np.abs(get_center_value_infinite_cyl(alpha10s, RMax=Rmax, isAlphaPositive=True)), marker='',
                color='k', linestyle='--', zorder=-200, label=r'$L\to\infty$')
    # ... (other analytical plots) ...
    axs[1].plot(alphas, np.abs(get_center_value_sphere_analytical(alpha10s, isAlphaPositive=True)), marker='',
                color='k', linestyle=':', zorder=-200, label=r'$\mathrm{Sphere}$')
    
    # Load and plot numerical data for different cavity lengths and alpha signs
    # Panel 1: alpha > 0, long cavity
    axs[0].set_title(r'$L=5~\mathrm{m}$', fontsize=fontsize)
    data = np.loadtxt('center_values_isAlphaPositive=%s_isCylinder=%s_L=%.1f.txt' % (True, isCylinder, 5), skiprows=1)
    # ... (data processing and plotting) ...
    
    # Panel 2: alpha > 0, short cavity
    axs[1].set_title(r'$L=0.2~\mathrm{m}$', fontsize=fontsize)
    data = np.loadtxt('center_values_isAlphaPositive=%s_isCylinder=%s_L=%.1f.txt' % (True, isCylinder, 0.2), skiprows=1)
    # ... (data processing and plotting) ...
    
    # Panel 3: alpha < 0, long cavity
    data = np.loadtxt('center_values_isAlphaPositive=%s_isCylinder=%s_L=%.1f.txt' % (False, isCylinder, 5), skiprows=1)
    # ... (data processing and plotting) ...
    
    # Panel 4: alpha < 0, short cavity
    data = np.loadtxt('center_values_isAlphaPositive=%s_isCylinder=%s_L=%.1f.txt' % (False, isCylinder, 0.2),
                      skiprows=1)
    # ... (data processing and plotting) ...
    
    # Apply plot cosmetics and save the figure
    for i, ax in enumerate(axs):
        ax.set_xscale('log')
        ax.set_yscale('log')
        if i > 1: ax.set_xlabel(r'$|\alpha|~[\mathrm{GeV}^{-2}]$', fontsize=fontsize)
        if i % 2 == 0: ax.set_ylabel(r'$|\varphi|$', fontsize=fontsize)
        ax.set_ylim([1e-5, 4e4])
        do_cosmetics(ax, labelsize=labelsize)
    
    axs[0].legend(fontsize=fontsize, frameon=False)
    axs[1].legend(fontsize=fontsize, frameon=False)
    plt.tight_layout()
    plt.savefig('center_values_cyl.pdf', bbox_inches='tight', dpi=500)
    # plt.show()


def get_peak_solutions():
    """
    Loads peak information from a saved file, re-runs the simulation for a few peaks,
    and saves the mesh and solution at those specific points for later plotting.
    """
    Npeaks = 3
    if L == 0.5:
        alpha10s = np.concatenate((np.logspace(-4, -2, num=50), np.logspace(-2, np.log10(1e-1), num=2000)))
    else:
        alpha10s = np.concatenate((np.logspace(-4, -2, num=50), np.logspace(-2, 0, num=2000)))
    
    for i, LL in enumerate([0.2]):
        print('L =', LL)
        data = np.load('peaks_L=%.1f_wrong.npz' % (2 * LL))
        alpha10s = data['alpha10s']
        peak_indices = np.array([588, 1350, 1589, 1758, 1934, 2024])  # Pre-defined peak indices
        alpha10s_ = data['alpha10s'][peak_indices]
        peak_values = np.empty_like(peak_indices)
        
        for j, myalpha10 in enumerate(alpha10s_):
            mykappa = kappa_ref / np.sqrt(myalpha10)
            x, m = get_solution(mykappa, False, True)
            peak_values[j] = get_value_at_origin(x, m, mykappa, LL)[0]
            m.save('solutions/solution_isAlphaPositive=%s_isCylinder=%s_L=%.1f_alpha = %.4e_get_peak_solutions.msh' % (
                False, True, 2 * LL, myalpha10 * 1e-10), point_data={'solution': x}, encode_point_data=True)
            np.savez('peaks_L=%.1f.npz' % (2 * L), peak_indices=peak_indices, peak_values=peak_values,
                     alpha10s=alpha10s)


def plot_solution_panels():
    """
    Generates a panel plot showing the field solution (phi) inside and around
    the cavity for different resonant peaks.
    """
    Npeaks = 3
    fontsize = 18
    labelsize = 18
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=False, sharey=False)
    
    for i, L in enumerate([2.5, 0.1]):
        
        data = np.load('peaks_L=%.1f.npz' % (2 * L))
        peak_indices = data['peak_indices'][:Npeaks]
        peak_values = data['peak_values'][:Npeaks]
        alpha10s = data['alpha10s'][peak_indices][:Npeaks]
        
        print(peak_indices)
        print(peak_values)
        print(alpha10s)
        
        for j, myalpha10 in enumerate(alpha10s):
            print(L, myalpha10 * 1e-10)
            kappa = kappa_ref / np.sqrt(myalpha10)
            out = ['point_data']
            
            filename = 'solutions/solution_isAlphaPositive=%s_isCylinder=%s_L=%.1f_alpha = %.4e.msh' % (False, True,
                                                                                                        2 * L,
                                                                                                        myalpha10 * 1e-10)
            print('Loading ', filename)
            # continue
            m = MeshTri.load(filename, out=out)
            x = out[0]['solution']
            
            rs = np.linspace(np.amin(m.p[0]), np.amax(m.p[0]), 300)
            zs = np.linspace(np.amin(m.p[1]), np.amax(m.p[1]), 300)
            
            R_, Z_ = np.meshgrid(rs, zs)
            R_dimless, dR_dimless, L_dimless, dL_dimless, Rmax_dimless, Zmax_dimless = get_dimless_sizes(kappa, L)
            
            cond1 = (R_ <= R_dimless) & (np.abs(Z_) >= L_dimless) & (np.abs(Z_) <= L_dimless + dL_dimless)
            cond2 = (R_ >= R_dimless) & (R_ <= R_dimless + dR_dimless) & (np.abs(Z_) <= L_dimless + dL_dimless)
            condition = (cond1 | cond2)
            masked_condition = np.where(condition, 1, np.nan)
            
            tpc = axs[i, j].tripcolor(m.p[0], m.p[1], m.t.T, x, cmap='rainbow')
            
            # fig.colorbar(tpc)
            if i == 1:
                axs[i, j].set_xlabel(r'$r / \ell$', fontsize=fontsize)
            if j == 0:
                axs[i, j].set_ylabel(r'$z / \ell$', fontsize=fontsize)
            
            divider = make_axes_locatable(axs[i, j])
            cax = divider.append_axes('right', size='15%', pad=0.1)
            cbar = fig.colorbar(tpc, cax=cax, orientation='vertical')
            if j == 2:
                cbar.set_label(r'$\varphi(r,z)$', fontsize=fontsize)
            cbar.ax.tick_params(labelsize=15)
            cbar.formatter = ticker.FormatStrFormatter('%.2f')
            
            # im1 = plt.imshow(condition, extent=(R_.min(), R_.max(), Z_.min(), Z_.max()),origin="lower", cmap="Greys")
            cmap = mpl.colors.ListedColormap(["k", "k"])
            axs[i, j].pcolormesh(R_, Z_, masked_condition, cmap=cmap, alpha=0.5)
            
            ff_r = 6 if i == 0 else 3
            ff_z = 2
            axs[i, j].set_xlim([0, ff_r * (R_dimless + dR_dimless)])
            axs[i, j].set_ylim([-ff_z * (L_dimless + dL_dimless), ff_z * (L_dimless + dL_dimless)])
            do_cosmetics(axs[i, j], labelsize)
    
    plt.tight_layout()
    plt.show()
    # plt.savefig('solutions_cyl.pdf' , bbox_inches='tight', dpi=200)



##################################################################
##################################################################

# --- Main Execution Block ---
# These function calls run the entire simulation, data processing, and plotting pipeline.
# First, solve for center values for both positive and negative alpha.
get_center_values(isAlphaPositive=True, isCylinder=True)
get_center_values(isAlphaPositive=False, isCylinder=True)

# Plot the center values comparison
plot_center_values()

# Find and save solutions at resonant peaks
get_peak_solutions()

# Plot the field solutions at the peaks
plot_solution_panels()
