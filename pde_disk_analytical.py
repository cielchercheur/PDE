"""

This script:
  1) Imports pde_disk.py and programmatically sets its parameters to the part (c) case.
  2) Runs its generator solve_annulus_diffusion() and stores radial profiles at selected times.
  3) Computes the analytic axisymmetric solution (m=0) using Bessel functions:
       phi_n(r) = J0(k_n r)Y0(k_n a) - Y0(k_n r)J0(k_n a)
     where k_n solve: J0(k a)Y0(k b) - Y0(k a)J0(k b) = 0
  4) Plots numerical vs analytic profiles.

Dependencies
------------
numpy, matplotlib, scipy

"""

# -----------------------------
# Imports
# -----------------------------
import numpy as np                       # arrays, vectorized math
import matplotlib.pyplot as plt          # plotting

from scipy.special import j0, y0         # Bessel J0 and Y0
from scipy.optimize import brentq        # robust 1D root finder (bracketing method)
from scipy.integrate import simpson      # numerical integration (Simpson rule)

import pde_disk as sim                   # numerical code (must be in same folder)


# ============================================================
# USER SETTINGS
# ============================================================

# Geometry (annulus radii).
#   sim.R_INNER == a, sim.R_OUTER == b
A_INNER = 1.0
B_OUTER = 5.0

# Thermal diffusivity alpha
ALPHA = 1.0

# Spatial resolution
NR = 50           # number of radial intervals
NTHETA = 120      # number of angular nodes (not critical for axisymmetric test)

# Verification initial condition (ring)
T_BORDER = 0.0    # fixed temperature at both borders (Dirichlet)
T_RING = 50.0     # temperature on the initial ring band
RING_RADIUS = 0.5 * (A_INNER + B_OUTER)  # mid-radius ring, matching statement

# Time settings
T_FINAL = 5    # final physical time to run to
TIMES_TO_PLOT = [0.00, 0.05, 0.10, 0.25, 0.50, 0.75, 1., 2., 5.]  # times at which to compare profiles

# Analytic series settings
N_TERMS = 120        # number of modes to include in the analytic series
R_INT_POINTS = 8000  # integration points for coefficient integrals (higher = more accurate)

# Plotting
MAKE_PLOTS = True


# ============================================================
# Helper functions: for Dirichlet annulus (m=0)
# ============================================================

def dirichlet_characteristic(k: float, a: float, b: float) -> float:
    """
    Characteristic equation for Dirichlet-Dirichlet annulus values (axisymmetric, m=0):

        F(k) = J0(k a) Y0(k b) - Y0(k a) J0(k b)

    Roots k_n > 0 of F(k)=0 give the radial values.
    """
    return j0(k * a) * y0(k * b) - y0(k * a) * j0(k * b)


def find_dirichlet_values(a: float, b: float, n_roots: int) -> np.ndarray:
    """
    Find the first n_roots positive roots of the Dirichlet annulus characteristic equation
    using:
      1) a sign-change scan on a k-grid
      2) brentq root solve on each bracket [k1,k2]

    Returns
    -------
    k : (n_roots,) ndarray
        values in ascending order.
    """
    # Root spacing is roughly ~ pi/(b-a), so pick a conservative maximum search k.
    k_max = (n_roots + 8) * np.pi / (b - a)

    # Scan step: enough to see sign changes reliably.
    dk = np.pi / (30 * (b - a))

    # Build scan grid (avoid k=0 because Y0(0) is singular).
    ks = np.arange(1e-6, k_max + dk, dk)

    # Evaluate characteristic function on the scan grid.
    F = dirichlet_characteristic(ks, a, b)

    roots = []

    # Loop over adjacent pairs to detect sign changes.
    for k1, k2, f1, f2 in zip(ks[:-1], ks[1:], F[:-1], F[1:]):
        # Skip invalid values (can happen if something goes non-finite).
        if (not np.isfinite(f1)) or (not np.isfinite(f2)):
            continue

        # If there is a strict sign change, there is at least one root in (k1,k2).
        if f1 * f2 < 0.0:
            root = brentq(dirichlet_characteristic, k1, k2, args=(a, b), maxiter=200)
            roots.append(root)

        # Stop once we have enough roots.
        if len(roots) >= n_roots:
            break

    # Safety: if we didn't find enough roots, tell the user what to change.
    if len(roots) < n_roots:
        raise RuntimeError(
            f"Found only {len(roots)} roots. Increase N_TERMS or adjust scan parameters."
        )

    return np.array(roots, dtype=float)


def phi_dirichlet(k: float, r: np.ndarray, a: float) -> np.ndarray:
    """
    Dirichlet annulus function (axisymmetric, m=0), constructed to satisfy phi(a)=0:

        phi(r) = J0(k r) Y0(k a) - Y0(k r) J0(k a)

    If k is a root of the characteristic equation, then also phi(b)=0.
    """
    return j0(k * r) * y0(k * a) - y0(k * r) * j0(k * a)


# ============================================================
# Analytic series builder (matches numerical "ring band" initial condition)
# ============================================================

def build_analytic_solution_for_ring_band(
    a: float,
    b: float,
    alpha: float,
    ring_radius: float,
    ring_temperature: float,
    border_temperature: float,
    nr: int,
    n_terms: int,
    r_int_points: int,
):
    """
    Construct analytic series data for the same type of initial condition code uses:

    pde_disk.py with INIT_TYPE="ring" sets:
      T(r,theta,0) = ring_temperature for nodes with |r - ring_radius| <= 0.5*dr
      and 0 elsewhere (plus borders later enforced).

    This function:
      1) computes values k_n
      2) computes coefficients c_n by numerical integration with weight r
      3) returns a function T_analytic(r_nodes, t) evaluating the truncated series

    Returns
    -------
    k : (n_terms,) values
    c : (n_terms,) coefficients
    T_analytic : callable
        T_analytic(r_nodes, t) -> array of shape (len(r_nodes),)
    """
    # Radial grid spacing used by numerical method
    dr = (b - a) / nr

    # Integration grid for coefficient integrals.
    r_int = np.linspace(a, b, r_int_points)

    # Initial condition used for analytic coefficient computation:
    # a "ring band" of thickness dr around ring_radius.
    u0 = np.where(np.abs(r_int - ring_radius) <= 0.5 * dr,
                  ring_temperature - border_temperature,
                  0.0)

    # 1) Values
    k = find_dirichlet_values(a, b, n_terms)

    # 2) Coefficients via weighted inner products
    c = np.zeros_like(k)

    for i, ki in enumerate(k):
        # Function samples on integration grid
        phi_i = phi_dirichlet(ki, r_int, a)

        # Numerator: ∫ u0(r) * phi_i(r) * r dr
        num = simpson(u0 * phi_i * r_int, x=r_int)

        # Denominator: ∫ phi_i(r)^2 * r dr
        den = simpson(phi_i * phi_i * r_int, x=r_int)

        c[i] = num / den

    # 3) Return evaluator for the truncated series
    def T_analytic(r_nodes: np.ndarray, t: float) -> np.ndarray:
        # Precompute functions at the requested nodes
        phi_mat = np.array([phi_dirichlet(ki, r_nodes, a) for ki in k])  # (n_terms, Nr_nodes)

        # Time decay factors e^{-alpha k^2 t}
        decay = np.exp(-alpha * (k ** 2) * t)[:, None]                   # (n_terms, 1)

        # Series sum for u(r,t) = Σ c_n phi_n(r) e^{-alpha k_n^2 t}
        u = (c[:, None] * phi_mat * decay).sum(axis=0)

        # Add border temperature back: T = T_border + u
        return border_temperature + u

    return k, c, T_analytic


# ============================================================
# Numerical run wrapper (calls existing generator)
# ============================================================

def run_numerical_and_store_profiles(times_to_store):
    """
    Runs sim.solve_annulus_diffusion() and stores the *radial mean* profile at specified times.

    Because the part (c) test is axisymmetric, the correct 1D quantity to compare is:
        T_mean(r_i,t) = mean over theta of T(r_i,theta,t)
    """
    # Convert desired times into integer time step indices.
    steps_to_store = {int(round(t / sim.DT)) for t in times_to_store}

    # Initialize the generator
    gen = sim.solve_annulus_diffusion()

    # First yield is t=0.0
    rho, T, t = next(gen)

    # Storage dict: step -> profile array
    stored = {0: T.mean(axis=1).copy()}

    # March until we have all requested times.
    for rho, T, t in gen:
        step = int(round(t / sim.DT))

        if (step in steps_to_store) and (step not in stored):
            stored[step] = T.mean(axis=1).copy()

        if len(stored) == len(steps_to_store):
            break

    # Convert to time-keyed dictionary for convenience
    out = {}
    for step, prof in stored.items():
        out[step * sim.DT] = prof

    return rho, out


# ============================================================
# Main script: set solver parameters, run numeric, run analytic, plot
# ============================================================

def main():
    # -----------------------------
    # 1) Programmatically set solver's parameters to the part (c) test
    # -----------------------------

    sim.ALPHA = ALPHA
    sim.R_INNER = A_INNER
    sim.R_OUTER = B_OUTER

    sim.NR = NR
    sim.NTHETA = NTHETA

    # Part (c): fixed equal temperature at both borders (Dirichlet/Dirichlet)
    sim.BC_TYPE = "dirichlet"
    sim.T_IN = T_BORDER
    sim.T_OUT = T_BORDER

    # Part (c): initial ring at different temperature
    sim.INIT_TYPE = "ring"
    sim.T0 = T_RING
    sim.RING_RADIUS = RING_RADIUS

    # Part (c): no source term (so analytic solution is axisymmetric)
    sim.SOURCE_TYPE = "none"

    # Ensure we get every step if needed
    sim.PLOT_EVERY = 1

    # -----------------------------
    # 2) Choose DT from solver's explicit stability limit
    #    (same formula as in code)
    # -----------------------------
    dr = (sim.R_OUTER - sim.R_INNER) / sim.NR
    dth = 2.0 * np.pi / sim.NTHETA

    dt_max = 0.5 / (sim.ALPHA * (1.0 / dr**2 + 1.0 / (sim.R_INNER * dth)**2))

    # Pick a safe fraction of dt_max
    sim.DT = 0.8 * dt_max
    #sim.DT = 0.0004


    # Pick NT from the desired final time
    sim.NT = int(np.ceil(T_FINAL / sim.DT))

    print(f"[setup] dr={dr:.6g}, dθ={dth:.6g}, dt_max={dt_max:.6g}, DT={sim.DT:.6g}, NT={sim.NT}")

    # -----------------------------
    # 3) Run numerical solver and store profiles at selected times
    # -----------------------------
    rho_nodes, num_profiles = run_numerical_and_store_profiles(TIMES_TO_PLOT)

    # -----------------------------
    # 4) Build analytic series (matching numerical ring-band initialization)
    # -----------------------------
    _, _, T_analytic = build_analytic_solution_for_ring_band(
        a=sim.R_INNER,
        b=sim.R_OUTER,
        alpha=sim.ALPHA,
        ring_radius=sim.RING_RADIUS,
        ring_temperature=sim.T0,
        border_temperature=sim.T_IN,
        nr=sim.NR,
        n_terms=N_TERMS,
        r_int_points=R_INT_POINTS,
    )

    # Evaluate analytic profiles at the same node radii and times
    ana_profiles = {t: T_analytic(rho_nodes, t) for t in sorted(num_profiles.keys())}

    # -----------------------------
    # 5) Print a simple error summary
    # -----------------------------
    for t in sorted(num_profiles.keys()):
        num = num_profiles[t]
        ana = ana_profiles[t]
        rel_l2 = np.linalg.norm(num - ana) / (np.linalg.norm(ana) + 1e-14)
        print(f"[compare] t={t:.6g}  rel_L2_error={rel_l2:.3e}  max(num)={num.max():.6g}  max(ana)={ana.max():.6g}")

    # -----------------------------
    # 6) Minimal plotting (optional)
    # -----------------------------
    if MAKE_PLOTS:
        for t in sorted(num_profiles.keys()):
            plt.figure()
            plt.plot(rho_nodes, num_profiles[t], label="numerical (theta-mean)")
            plt.plot(rho_nodes, ana_profiles[t], "--", label="analytic series")
            plt.xlabel("r")
            plt.ylabel("T")
            plt.title(f"Annulus heat diffusion: numeric vs analytic at t={t:.4g}")
            plt.legend()
            plt.grid(True)

        plt.show()


if __name__ == "__main__":
    main()
