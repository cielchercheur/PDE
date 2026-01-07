import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
#                               PARAMETERS
# =============================================================================
# Physics / Domain
ALPHA = 1.0          # Thermal diffusivity
R_INNER = 1.0        # Inner radius
R_OUTER = 5.0        # Outer radius

# Grid
NR = 50              # Number of radial intervals
NTHETA = 120         # Number of angular points

# Time Integration
DT = 0.0004          # Time step (must be small for stability)
#DT = 0.0012         # Time step (for blow up)
NT = 12500           # Number of time steps

# Moving Source
MOVING_SOURCE = True
#MOVING_SOURCE = False
OMEGA = 5.0         # Angular speed (rad/s)
TS = 50.0            # Source temperature scale
RING_RADIUS = 3.0    # Radius where the source moves (mid-radius)

Q = 50.0  # constant heating rate (temperature/time)

# Boundary Conditions
BC_TYPE = "dirichlet" # Options: "dirichlet" or "neumann"
#BC_TYPE = "neumann"   # Options: "dirichlet" or "neumann"
T_IN = 0.0            # Inner wall temperature (for Dirichlet)
T_OUT = 0.0           # Outer wall temperature (for Dirichlet)

# Initial Condition
INIT_TYPE = "ring"    # Options: "ring", "gaussian_ring"
T0 = 0.0              # Initial temperature scale

# Visualization
V_MIN = 0.0           # Colorbar minimum
V_MAX = 80.0          # Colorbar maximum
# =============================================================================

def solve_annulus_diffusion():
    # Grid setup
    dr = (R_OUTER - R_INNER) / NR
    dth = 2.0 * np.pi / NTHETA
    rho = np.linspace(R_INNER, R_OUTER, NR + 1)

    # Stability Check
    dt_max = 0.5 / (ALPHA * (1.0 / dr**2 + 1.0 / (R_INNER * dth)**2))
    print(f"Simulation Info: dt={DT:.5f}, Max Stable dt={dt_max:.5f}")
    if DT > dt_max:
        print("WARNING: Time step exceeds stability limit!")

    # State array
    T = np.zeros((NR + 1, NTHETA))

    # Initial Condition
    if INIT_TYPE == "ring":
        mask = np.abs(rho - RING_RADIUS) <= 0.5 * dr
        T[mask, :] = T0
    elif INIT_TYPE == "gaussian_ring":
        T[:] = T0 * np.exp(-((rho[:, np.newaxis] - RING_RADIUS)**2) / (1.0**2))

    # Pre-calculate geometric factors
    ri = rho[1:-1, np.newaxis]
    rip = (ri + 0.5 * dr)
    rim = (ri - 0.5 * dr)
    r_coeff = DT * ALPHA / (ri * dr**2)
    a_coeff = DT * ALPHA / (ri**2 * dth**2)
    isrc = int(round((RING_RADIUS - R_INNER) / dr))
    isrc = max(1, min(NR - 1, isrc))
    cell_area = rho[isrc] * dr * dth

    print(f"Starting time evolution for {NT} steps...")


    for n in range(NT):
        T_inner = T[1:-1, :]
        
        # Vectorized Laplacian
        T_jp = np.roll(T_inner, -1, axis=1)
        T_jm = np.roll(T_inner, 1, axis=1)
        
        radial = r_coeff * (rip * (T[2:, :] - T_inner) - rim * (T_inner - T[:-2, :]))
        angular = a_coeff * (T_jp - 2.0 * T_inner + T_jm)
        
        T_new_inner = T_inner + radial + angular

        # Moving source logic
        if MOVING_SOURCE:
            t = n * DT
            theta_s = (OMEGA * t) % (2.0 * np.pi)
            js = int(round(theta_s / dth)) % NTHETA
            T_new_inner[isrc-1, js] += DT * (OMEGA * TS) / cell_area
        else:
            theta_s = 0.0
            js = int(round(theta_s / dth)) % NTHETA

            add = DT * Q # Q is a heating rate (temp/time), so DT*Q is a temperature change
            T_new_inner[isrc - 1, js] += add

        T[1:-1, :] = T_new_inner

        # Boundary Conditions
        if BC_TYPE == "dirichlet":
            T[0, :] = T_IN
            T[NR, :] = T_OUT
        elif BC_TYPE == "neumann":
            T[0, :] = (4 * T[1, :] - T[2, :]) / 3
            T[NR, :] = (4 * T[NR - 1, :] - T[NR - 2, :]) / 3

    return rho, T

def main():
    # 1. Run Solver
    print("Running simulation...")
    radii, Temp_grid = solve_annulus_diffusion()

    # 2. Visualization
    print("Plotting results...")
    theta_vals = np.linspace(0, 2 * np.pi, NTHETA + 1)
    r_vals = np.array(radii)

    R_mesh, THETA_mesh = np.meshgrid(r_vals, theta_vals)
    X = R_mesh * np.cos(THETA_mesh)
    Y = R_mesh * np.sin(THETA_mesh)

    # Wrap data: Append first angular column to the end to close the circle (NTHETA + 1)
    Z_wrapped = np.concatenate([Temp_grid, Temp_grid[:, [0]]], axis=1)
    Z = Z_wrapped.T # Transpose to (Ntheta+1, Nr+1)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, Z, cmap='inferno', shading='gouraud')
    plt.clim(V_MIN, V_MAX)
    plt.colorbar(label='Temperature')
    plt.title(f"2D Heat Diffusion in Annulus (t = {DT * NT:.3f}s)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.savefig("annulus_diffusion_dirichlet.png")

    plt.show()


if __name__ == "__main__":
    main()