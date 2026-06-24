# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Welcome to Raf's setup_1D_neutrals_mgop module! (An extension for the jax-sn package by T. Bogaarts)
#
# The purpose of this extension is to solve the Boltzmann equation for arbitrary neutrals instead of just neutrons.
# This is done by rewriting the neutrals equation into the same form as the one used by jax-sn.
# With this rewritten equation, we get a set of equations that map the two Boltzmann equations to one another
# The goal of this module is to implement these mappings into the code.
#
# For the sake of simplicity, this model only considers a 1D domain (hence the name).
# Higher dimensional extensions of this model should be possible, but have not been implemented as of yet.
#
# NOTE: Due to time constrains, this model does not yet include the scattering source term of the Boltzmann equation.
# As such, the results produced by this model are not fully physical, and should thus not be regarded as such.
# The other terms of the equation (the total scattering area and the external source term) have been tested
# and seem to be operating correctly.
#
# Further extension of this module should firstly aim towards implementing the scattering source term,
# after which extension into higher-dimensional domains may be considered.
#
# Note that, as this module is an extension for the jax-sn package, it and all its dependencies must be installed
# in order to run the code below.
#
# Lastly, this code has been made with substantial direct and indirect help from Timo Bogaarts.
#
# - R.K.K. van der Veeken
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Import the necessary libraries
from matplotlib import pyplot as plt
from jax_sn.tools import setup_test_mgop
import jax_sn.solution_domain
from jax_sn.operators.multi_group_operator import MultiGroupOperator
from jax_sn.domain import CrossSectionData, CrossSectionsIndexed
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Optional, Tuple
from jax_sn.domain import CrossSectionData
import lineax as lx

# Make sure jax 64bit is enabled in order to be able to handle very small or large numbers
jax.config.update('jax_enable_x64', True)


# ////////////////////////////////////////////////////////////////////
# Define all of the new functions we will need to perform a simulation
# ////////////////////////////////////////////////////////////////////

# %% Define the needed functions that comprise the composite functions
def SecondOrder(x, a, b, c):
    y = a * x ** 2 + b * x + c
    return y


def ThirdOrder(x, a, b, c, d):
    y = a * x ** 3 + b * x ** 2 + c * x + d
    return y


def FourthOrder(x, a, b, c, d, e):
    y = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e
    return y


def Gaussian(x, a, b, c, d):
    y = a + b * np.exp(-c * (x - d) ** 2)
    return y


# %% Define the background profile
def profile_1(x):
    """
    Defines the temperature, density and velocity profiles from the 1D model from Wen et al. as functions of x
    Fits are obtained by trial and error and are composed of multiple functions
    Fits do not represent any analytical solutions

    :param x: 1D position array

    :return Te(x), Td(x), Nd(x), Ud(x): The different needed functions of x as a tuple
    """

    def Te(x):
        """
        Defines the electron temperature, T_e, from the 1D example from Wen et al.
        Fit consists of a third and second order polynomial, joined at 0.95

        :param x: 1D position array.    Unit: [normalised to domain length]

        :return y: Value of Te at x.    Unit: [eV]
        """
        y = jnp.where(x <= 0.95, ThirdOrder(x, -15.6204455, 1.19356502, -68.32567613, 84.17674725),
                      SecondOrder(x, 2424.85090127, -4863.52608013, 2438.67517886))

        # Uncomment this line to make Te constant in space
        #y = jnp.ones_like(y) * jnp.mean(y)

        return y

    def Td(x):
        """
        Defines the ion temperature, T_d, from the 1D example from Wen et al.
        Fit consists of a fourth and two second order polynomials, joined at 0.85 and 0.95

        :param x: 1D position array.    Unit: [normalised to domain length]

        :return y: Value of Td at x.    Unit: [eV]
        """
        y = jnp.where(x <= 0.85, FourthOrder(x, -103.88068322, 129.14867609, 48.44540225, -168.10331715, 97.83390725),
                      jnp.where(x <= 0.95, SecondOrder(x, -403.19495502, 644.25264105, -241.37620952),
                                SecondOrder(x, 2424.85090127, -4863.52608013, 2438.67517886)))

        # Uncomment this line to make Ts constant in space
        #y = jnp.ones_like(y) * jnp.mean(y)

        return y

    def Nd(x):
        """
        Defines the ion particle density, n_d, from the 1D example from Wen et al.
        Fit consists of four fourth order polynomials and a Gaussian
        Functions are joined at 0.6, 0.9, 0.9325, 0.95, and 0.97

        :param x: 1D position array.    Unit: [normalised to domain length]

        :return y: Value of Nd at x.    Unit: [m-3]
        """

        y = jnp.where(x <= 0.6,
                      1e20 * FourthOrder(x, -6.39115562e+00, 9.97641928e+00, -2.95485623e+00, 1.25167201e+00, -8.43517610e-03),
                      jnp.where(x <= 0.9, 1e20 * FourthOrder(x, 108.24823641, -265.3389528, 251.59309483, -105.66970629,17.11927553),
                                jnp.where(x <= 0.9325, 1e20 * SecondOrder(x, 401.1766785, -701.09278489, 309.43270446),
                                          jnp.where(x <= 0.95, 1e20 * SecondOrder(x, 2220.72739794, -4039.9505757 , 1840.79479044),
                                                    jnp.where(x <= 0.97, 1e20 * FourthOrder(x, -273083.11073179, 552298.28096658, -78828.80098799, -408863.44466554,208471.98793824),
                                                                1e20 * Gaussian(x, 9.21546911e+00, 4.56231100e+00,5.17944444e+03, 9.73223718e-01))))))

        # Uncomment this line to make Nd constant in space
        #y = jnp.ones_like(y) * jnp.mean(y)

        return y

    def Ud(x):
        """
        Defines the ion velocity, u_d, from the 1D example from Wen et al.
        Fit consists of a second order polynomial, and three fourth order ones
        Functions are joined at 0.075, 0.825, and 0.9375

        :param x: 1D position array.    Unit: [normalised to domain length]

        :return y: Value of Ud at x.    Unit: [m/s]
        """

        y = jnp.where(x <= 0.075, 1e3 * SecondOrder(x, 40.85982227, -17.41636791, -8.91684715),
                      jnp.where(x <= 0.825, 1e3 * FourthOrder(x, -114.41587819, 194.49030858, -84.71189273, 26.33154421, -11.64868075),
                                jnp.where(x <= 0.9375, 1e3 * FourthOrder(x, -61807.39830649, 214067.9046647, -278011.74463991,160470.80146485, -34728.22629126),
                                          1e3 * FourthOrder(x, -705601.16129143, 2718279.13623602, -3924328.63366324, 2516285.2449803, -604624.71190673))))

        # Uncomment this line to make Ud constant in space
        #y = jnp.ones_like(y) * jnp.mean(y)

        return y

    return Te(x), Td(x), Nd(x), Ud(x)


# %% Define the rate coefficients
def get_Kr(Te):
    """
    Defines the radiative recombination rate coefficient, Kr, as given by equation 3 from Wen et al.

    :param Te: The electron temperature.                    Unit: [eV]

    :return Kr: The radiative recombination coefficient.    Unit: [m3 s-1]
    """
    Kr = 0.7e-19 * jnp.sqrt(13.6 / Te)
    return Kr


def get_Kd(Te):
    """
    Defines the electron impact ionization rate coefficient, Kd, as given by equation 3 from Wen et al.

    :param Te: The electron temperature.                    Unit: [eV]

    :return Kd: The electron impact ionization coefficient. Unit: [m3 s-1]
    """
    Kd = ((2e-13) / (6 + Te / 13.6)) * jnp.sqrt(Te / 13.6) * jnp.exp(-13.6 / Te)
    return Kd


def get_Kcx(Td):
    """
    Defines the charge exchange rate coefficient, Kcx, as given by equation 3 from Wen et al.

    :param Td: The ion temperature.                         Unit: [eV]

    :return Kcx: The charge exchange rate coefficient.      Unit: [m3 s-1]
    """
    Kcx = 3.2e-15 * jnp.sqrt(Td / 0.026)
    return Kcx


# %% Define n_e
def get_Ne(n_d: 'array', Z_d: int | float):
    """
    Defines the electron particle density, n_d, as a function of n_d, according to n_e = Z_d * n_d,
    where Z_d is the ion atomic number

    :param n_d: Ion density distribution.       Unit: [m-3]
    :param Z_d: Ion atomic number.              Unit: [-]

    :return n_e: Electron density distribution  Unit: [m-3]
    """
    n_e = Z_d * n_d
    return n_e


# %% Define the energy levels
def linear_energy_levels(n_energy_levels: int):
    """
    Creates a linear set of n_energy_levels energy levels between 0.001 (arbitrary > 0, << 1) and E_max
    Unit of E is [J]

    :param n_energy_levels: The number of energy levels.    Unit: [-]

    :return E: An array of all energy levels.               Unit: [J]
    """
    E_max = 7.84457e-16  # [J] Maximum energy
    E_bounds = jnp.linspace(0, E_max, n_energy_levels + 1)

    E_centers = np.array([])
    for i in range(n_energy_levels):
        E_avg = 0.5*(E_bounds[i] + E_bounds[i+1])
        E_centers = np.append(E_centers, E_avg)

    return E_centers


# %% Define the ion distribution function
def get_fd(profile, energies, angles, x):
    """
    Defines the ion distribution function according to equation 4 from Wen et al.
    fd uses a given background profile for parameters n_d, u_d, and T_d
    Returned value is a 3D array of size n_energy_levels x n_angles x n_elements, containing the values of fd with
    respect to the corresponding variables on each axis (0: energy, 1: angle, 2: position)

    :param profile:     The background profile to be used to calculate fd
    :param energies:    The energies for which to determine fd                          Unit: [J]
    :param angles:      The angles at which to determine fd                             Unit: [Ω]
    :param x:           The normalised positions at which to determine fd               Unit: [-]

    :return fd:         3D array containing the value of fd for each energy (axis 0),
                        direction (axis 1) and position (axis 2).                       Unit: [m-6 s3]
    """

    # === Define all constants ===
    e = 1.60217663e-19              # [-]   Ratio between eV and J (elementary charge)
    m_ions = 3.343586054e-27        # [kg]  PLACEHOLDER, ASSUMING DEUTERIUM ION MASS
    m_neutrals = 3.344496993e-27    # [kg]  PLACEHOLDER, ASSUMING DEUTERIUM ATOM MASS

    # === Define the background parameters as a function of position ===
    Te, Td, Nd, Ud = profile(x)   # Background electron/ion temperature, ion density, and ion velocity

    # === Determine the velocity as a function of energy and direction ===
    v_abs = jnp.sqrt(2 * energies / m_neutrals)             # [m/s] Length of the velocity vector as a function of energy

    # === Determine f_d for all energy levels, angles, and positions
    v_inner_product = v_abs[:, None, None]**2 + Ud[None, None, :]**2 - 2 * v_abs[:, None, None] * Ud[None, None, :] * angles[:, 0][None, :, None]

    fd = Nd[None, None, :] * (m_ions/(2*jnp.pi * e * Td[None, None, :]))**(3/2) * jnp.exp(-(m_ions/(2*e*Td[None, None, :])) * v_inner_product)

    # === Return fd as function output ===
    return fd


# %% Define the external source term
def get_qe(profile, energies, angles, x, basis):
    """
    Defines the external source term q_e of the Boltzmann transport equation (equation 1 in Bogaarts & Warmer)
    qe uses a given background profile for parameters n_d, u_d, T_d, and T_e

    :param profile:     The background profile to be used
    :param energies:    The energies for which to determine qe                      Unit: [J]
    :param angles:      The angles at which to determine qe                         Unit: [Ω]
    :param x:           The normalised positions at which to determine qe           Unit: [-]
    :param basis:       The basis functions for which to determine qe

    :return qe:         Array containing the value of qe for each energy (axis 0),
                        angle (axis 1), and position (axis 2).                      Unit: [kg-1 m-5 s]
    """
    # === Define all constants ===
    m_neutrals = 3.344496993e-27    # [kg]  PLACEHOLDER, ASSUMING DEUTERIUM ATOM MASS
    Zd = 1                          # [-]   PLACEHOLDER, ASSUMING DEUTERIUM ATOMIC NUMBER

    # === Define the background parameters as a function of position ===
    Te, Td, Nd, Ud = profile(x)     # [eV], [eV], [m-3], [m/s] Background electron/ion temperature, ion density, and ion velocity
    Ne = get_Ne(Nd, Zd)             # [m-3] background electron density
    Kr = get_Kr(Te)                 # [m3 s-1] radiative recombination rate coefficient

    # === Define the ion distribution function ===
    fd = get_fd(profile, energies, angles, x)

    # === Determine the velocity as a function of energy and direction ===
    v_abs = jnp.sqrt(2 * energies / m_neutrals)         # [m/s] Length of the velocity vector as a function of energy

    # === Determine q_e as a function of energy, angle, and position ===
    qe = Ne[None, None, :] * Kr[None, None, :] * fd * v_abs[:, None, None] / m_neutrals

    # === Return qe as function output (after some fuckery with the basis functions) ===
    return np.stack([qe] * basis, axis=-1) #qe #[q_angles, n_elements, n_basis]


# %% Define the total scattering cross-section
def get_sigma_t(profile, x, energies):
    """
    Defines the total scattering cross-section of the Boltzmann transport equation (equation 1 in Bogaarts & Warmer)
    sigma_t uses a given background profile for parameters n_d, u_d, T_d, and T_e

    :param profile:     The background profile to be used.
    :param x:           The normalised positions at which to determine sigma_t.             Unit: [-]
    :param energies:    The energies at which to determine sigma_t.                         Unit: [J]

    :return sigma_t:    Array containing the value of sigma_t for each position (axis 0),
                        and energy (axis 1).                                                Unit: [m]
    """

    # === Define all constants ===
    m_neutrals = 3.344496993e-27    # [kg]  PLACEHOLDER, ASSUMING DEUTERIUM ATOM MASS
    Zd = 1                          # [-]   PLACEHOLDER, ASSUMING DEUTERIUM ATOMIC NUMBER

    # === Define the background parameters as a function of position ===
    Te, Td, Nd, Ud = profile(x)     # [eV], [eV], [m-3], [m/s] Background electron/ion temperature, ion density, and ion velocity

    Ne = get_Ne(Nd, Zd)             # [m-3] background electron density
    Kd = get_Kd(Te)                 # [m3 s-1] electron impact ionization rate coefficient
    Kcx = get_Kcx(Td)               # [m3 s-1] charge exchange rate coefficient

    # === Determine the velocity as a function of energy ===
    v_abs = jnp.sqrt(2 * energies / m_neutrals)         # [m/s] Length of the velocity vector as a function of energy

    # === Determine sigma_t as a function of position and energy ===
    sigma_t = (Ne[:, None] * Kd[:, None] + Nd[:, None] * Kcx[:, None]) / v_abs[None, :]

    # === Return sigma_t as function output ===
    return sigma_t


# %% Define the new mgop creation function
def Create1DNeutralDomain(
    profile,
    degree: int,
    tn_quadrature_set: int,
    domain_size : float,
    domain_res  : int,
    energy_centers : jnp.ndarray,
    use_numpy: bool = False,
    **kwargs,
):
    """
    Creates a 1D domain based on a profile with varying background parameters, resulting in cross-sections
    and source terms that vary throughout the domain as functions of position.

    :param profile:                 The background profile to be used
    :param degree:                  The polynomial degree of the FE basis
    :param tn_quadrature_set:       The T_N quadrature order
    :param domain_size:             The physical length of the domain                               Unit: [m]
    :param domain_res:              The number of elements in the domain along x
    :param energy_centers:          The center values of the energy levels to be used               Unit: [J]
    :param use_numpy:               Use numpy-based domain construction (faster for large meshes)
    :param kwargs:                  Passed through to ``create_multi_group_operator``

    :return multigroupoperator:     A multi-group operator representing the domain
    :return qe:                     The source term in the Boltzmann equation                       Unit: [kg-1 m-5 s]
    :return element:                The spatial elements that make up the domain
    """

    # === Import necessary extensions ===
    from jax_sn.domain import Domain
    from jax_sn.solution_domain.basix_fem import BasixLagrangianSimplex
    from jax_sn.solution_domain import SolutionDomain
    from jax_sn.operators.multi_group_operator.multi_group_operator_creation import create_multi_group_operator
    from jax_sn.quadrature_set import create_tn_quadrature_set, QuadratureSetReduced

    # === Define the positions ===
    vertices = jnp.linspace(0, domain_size, domain_res + 1)                 #[0.0, 0.01, 0.02, ... 1.0]
    connectivity = jnp.stack([jnp.array([i, i+1]) for i in range(domain_res)])#, axis=-1)    #[[0,1], [1,2], [2,3]

    vertices_in_element = vertices[connectivity]    # [n_elements, n_vertices_per_element, 1]
    x = jnp.mean(vertices_in_element, axis=1)       # [n_elements] average position in element

    # === Define the angles ===
    tn_qs = QuadratureSetReduced.from_quadrature_set(create_tn_quadrature_set(tn_quadrature_set), n_dim=1)
    angles = tn_qs.angles

    # === Define the basis functions ===
    element = BasixLagrangianSimplex(degree, 1)
    basis = element.n_basis

    # === Define the external source term ===
    qe = get_qe(profile, energy_centers, angles, x / domain_size, basis)

    # === Define the cross section data ===
    sigma_t = get_sigma_t(profile, x / domain_size, energy_centers)

    cross_section_data = CrossSectionData(
        total=sigma_t,  # [n_elements, n_energy_groups]
        scattering=jnp.zeros((domain_res, 1, len(energy_centers), len(energy_centers)))  # ..#[n_elements, lorder, n_energy_groups[outgoing], n_energy_groups[incoming]
    )

    # === Define the layered cross-section ===
    layered_xs = CrossSectionsIndexed(
        indices = jnp.arange(0, domain_res),
        cross_section_data = cross_section_data
    )

    # === Create the domain ===
    if use_numpy:
        domain = Domain.from_mesh_and_cross_sections_numpy((vertices[:, None], connectivity), element.face_template, layered_xs)
    else:
        domain = Domain.from_mesh_and_cross_sections((vertices[:, None], connectivity), element.face_template, layered_xs)

    # === Determine the solution domain
    solution_domain = SolutionDomain.from_element_and_domain(element, domain)

    # === Create and return the multi-group operator, as well as the external source term, and the element ===
    return create_multi_group_operator(solution_domain, tn_qs, **kwargs), qe, element


# %% Define a function that determines the analytical result to the reduced implementation of the Boltzmann equation
def AnalyticalConstantSolution(E: float, Angle: float, x: 'array', x_max: float):
    """
    Determines the analytical solution for ψ, when every background parameter is constant
    In this case, we use the average values of the background parameters from Wen, Zhang and Wu

    :param E: The energy at which to determine the analytical solution.                     Unit: [J]
    :param Angle: The angle at which to determine the analytical solution.                  Unit: [Ω_x]
    :param x: An array of the positions at which to determine the analytical solution.      Unit: [m]
    :param x_max: The length of the domain.                                                 Unit: [m]

    :return y: The analytical solution to the constant background parameter problem (ψ).    Unit: [kg-1 m-4 s]
    """
    # === Define natural constants ===
    e = 1.60217663e-19              # Ratio between eV and J (elementary charge)    [-]
    Zd = 1                          # The Atomic number, assuming deuterium         [-]
    m_ions = 3.343586054e-27        # The ion mass (assumption: deuteron)           [kg]
    m_neutrals = 3.344496993e-27    # The atom mass (assumption: deuterium)         [kg]

    # === Define the constant values for the background parameters ===
    Te = 46.41462921364735          # The electron temperature  [eV]
    Td = 41.25532761794925          # The ion temperature       [eV]
    Nd = 1.57164332335595e+20       # The ion density           [m-3]
    Ud = -1075.2566711599022        # The ion velocity          [m s-1]

    # === Determine the electron density ===
    Ne = get_Ne(Nd, Zd)     # The electron density      [m-3]

    # === Determine the rate coefficients ===
    Kr = get_Kr(Te)         # The radiative recombination rate coefficient      [m3 s-1]
    Kd = get_Kd(Te)         # The electron impact ionization rate coefficient   [m3 s-1]
    Kcx = get_Kcx(Td)       # The charge exchange rate coefficient              [m3 s-1]

    # === Determine the speed ===
    v = jnp.sqrt(2 * E / m_neutrals)                # The particle speed        [m s-1]

    # === Determine the ion distribution function ===
    inpr = v**2 + Ud**2 - 2 * v * Ud * Angle        # The square in the Gaussian in fd      [m2 s-2]
    fd = Nd * (m_ions/(2*jnp.pi * e * Td))**(3/2) * jnp.exp(-(m_ions/(2*e*Td)) * inpr)      # [m-6 s3]

    # === Determine the external source term ===
    qe = (1/m_neutrals) * Ne * Kr * fd * v      # The external source term      [kg-1 m-5 s]

    # === Determine the total scattering term ===
    sigma_t = (Ne * Kd + Nd * Kcx) / v          # The total scattering term     [m-1]

    # === Determine the analytical solution ===
    if Angle > 0:
        y = (qe / sigma_t) * (1 - jnp.exp(-(1/Angle) * sigma_t * x))
    elif Angle < 0:
        y = (qe / sigma_t) * (1 - jnp.exp(-(1/Angle) * sigma_t * (x - x_max)))
    else:
        y = (qe / sigma_t) * jnp.ones_like(x)

    # === Return the analytical solution ===
    return y




# %%///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Perform a simulation using all of the functions defined above
# Note: the code below is an input terminal, some lines may need to be commented out or changed, depending on the simulation we wish to run
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# === define the parameters of the domain ===
n_energy_groups = 200       # The number of energy groups we want to simulate
domain_size = 1             # The size of our simulation domain                     Unit: [m]
domain_res = 1000           # The number of points in the domain
profile = profile_1         # The background profile we want to use
degree = 5                  # The polynomial degree of the FN we want to use
tn_quadrature_set = 3       # The TN quadrature order we want to use

energy_centers = linear_energy_levels(n_energy_groups)      # The array of energies we want to use      Unit: [J]

# === Create the domain, source term, and element ===
mgop, source, element = Create1DNeutralDomain(
    profile, degree=degree, tn_quadrature_set=tn_quadrature_set,
    domain_size=domain_size, domain_res=domain_res, energy_centers=energy_centers
)


# === Create the right-hand side of the equation ===
rhs = jnp.zeros_like(mgop.out_structure())      # Initialise the rhs
rhs_per_angle = []                              # Initialise the angle-dependent rhs
for i in range(n_energy_groups):                # Determine rhs and rhs_per_angle for each energy group in the system
    rhs_per_angle.append(mgop.transport_operator.to_energy_group(i).inv(mgop.mass_operator.mv(source[i,...])))
    rhs = rhs.at[i].set(mgop.moment_operator.mv(rhs_per_angle[-1]))
print("RHS Done")                               # Show that the rhs has been determined
print()                                         # ^ (Only here for convenience)

# === Define the numerical solution ===
solution = rhs          # Because we have left out scattering, we do not need a solver, and rhs is already our solution
solution_domain = jax_sn.solution_domain.SolutionDomain.from_element_and_domain(element, mgop.domain)

# === Define the set of points which we want to interpolate our solution onto ===
xinterpsize = 5000
x_interp = jnp.linspace(0, domain_size, xinterpsize)


# === Determine the simplified numerical and analytical solution for the angular flux ===
# === Use for verification.                                                           ===
# === Note that the averages in profile_1() need to be uncommented to make sure       ===
#       the numerical solution uses the constant values
nth_energy = 0     # The index of the specific energy group for which we want the results
nth_angle = 0      # The index of the specific angle for which we want the results

#"""
phi_result = solution_domain.interpolate(x_interp[:, None], rhs_per_angle[nth_energy][nth_angle, ...]) # The numerical result
    
phi_result_analytical = AnalyticalConstantSolution(                 # The analytical result
    linear_energy_levels(n_energy_groups)[nth_energy],
    mgop.quadrature_set.angles[nth_angle, 0],
    x_interp,
    domain_size
)
    
    
# === Determine the average relative error between the two graphs ===
# Note that we have left out the first and last value of both arrays, as here the analytical result can be 0 and the relative error would be infinite
difference = phi_result - phi_result_analytical             # The difference between each of the points
error = abs(difference)                                     # The error for each point
relative_error = error / phi_result_analytical              # The relative error for each point
average_relative_error = jnp.sum(relative_error[1:-1]) / (xinterpsize - 1)          # The average relative error
print(average_relative_error)
#"""


# === Determine the scalar flux (integral of the angular flux) ===
#phi_result = solution_domain.interpolate(x_interp[:,None], rhs[nth_energy, nth_angle, ...]) #[n_energy_groups, n_interp]


# === Produce a series of plots that illustrate the results ===
#""" # Create a linear plot of the data
plt.plot(x_interp, phi_result, label='Simulation result')
#plt.plot(x_interp, phi_result_analytical, label='Analytical result')
plt.title(f'linear plot:\n E = {linear_energy_levels(n_energy_groups)[nth_energy]} J')#, Ω_x = {mgop.quadrature_set.angles[nth_angle, 0]} ')
plt.xlabel('x [m]')
plt.ylabel('ψ [kg-1 m-4 s]')
plt.legend()
plt.show()
"""

""" # Create a logarithmic plot of the data
plt.semilogy(x_interp, phi_result, label='Simulation result')
#plt.semilogy(x_interp, phi_result_analytical, label='Analytical result')
plt.title(f'logarithmic plot:\n E = {linear_energy_levels(n_energy_groups)[nth_energy]} J')#, Ω_x = {mgop.quadrature_set.angles[nth_angle, 0]}')
plt.xlabel('x [m]')
plt.ylabel('ψ [kg-1 m-4 s]')
plt.legend()
plt.show()
"""

""" # Create a zoomed-in plot of the far right side of the data
plt.plot(x_interp, phi_result, label='Simulation result')
#plt.plot(x_interp, phi_result_analytical, 'o', label='Analytical result')
plt.xlim(domain_size * 0.99, domain_size)
plt.title(f'Zoomed linear plot back:\n E = {linear_energy_levels(n_energy_groups)[nth_energy]} J')#, Ω_x = {mgop.quadrature_set.angles[nth_angle, 0]}')
plt.xlabel('x [m]')
plt.ylabel('ψ [kg-1 m-4 s]')
plt.legend()
plt.show()
"""

""" # Create a zoomed-in plot of the far left side of the data
plt.plot(x_interp, phi_result, label='Simulation result')
#plt.plot(x_interp, phi_result_analytical, 'o', label='Analytical result')
plt.xlim(0, 0.01 * domain_size)
plt.title(f'Zoomed linear plot front:\n E = {linear_energy_levels(n_energy_groups)[nth_energy]} J')#, Ω_x = {mgop.quadrature_set.angles[nth_angle, 0]}')
plt.xlabel('x [m]')
plt.ylabel('ψ [kg-1 m-4 s]')
plt.legend()
plt.show()
#"""

