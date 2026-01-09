import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

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
#SOURCE_TYPE = "ring"
SOURCE_TYPE = "moving_source"
#SOURCE_TYPE = "stable_source"

OMEGA = 5.0          # Angular speed (rad/s)
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
PLOT_EVERY = 50       # Plot every Nth time step
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

    yield rho, T.copy(), 0.0

    for n in range(NT):
        T_inner = T[1:-1, :]
    
        # Vectorized Laplacian
        T_jp = np.roll(T_inner, -1, axis=1)
        T_jm = np.roll(T_inner, 1, axis=1)
        
        radial = r_coeff * (rip * (T[2:, :] - T_inner) - rim * (T_inner - T[:-2, :]))
        angular = a_coeff * (T_jp - 2.0 * T_inner + T_jm)
        
        T_new_inner = T_inner + radial + angular

        # Moving source logic
        if SOURCE_TYPE == "moving_source":
            t = n * DT
            theta_s = (OMEGA * t) % (2.0 * np.pi)
            js = int(round(theta_s / dth)) % NTHETA
            T_new_inner[isrc-1, js] += DT * (OMEGA * TS) / cell_area
        elif SOURCE_TYPE == "stable_source":
            theta_s = 0.0
            js = int(round(theta_s / dth)) % NTHETA
            add = DT * Q / cell_area
            T_new_inner[isrc - 1, js] += add
        elif SOURCE_TYPE == "ring":
            add = DT * Q / cell_area
            T_new_inner[isrc - 1, :] += add

        T[1:-1, :] = T_new_inner

        # Boundary Conditions
        if BC_TYPE == "dirichlet":
            T[0, :] = T_IN
            T[NR, :] = T_OUT
        elif BC_TYPE == "neumann":
            T[0, :] = (4 * T[1, :] - T[2, :]) / 3
            T[NR, :] = (4 * T[NR - 1, :] - T[NR - 2, :]) / 3

        if (n + 1) % PLOT_EVERY == 0:
            yield rho, T, (n + 1) * DT

def main():
    # 1. Initialize Generator
    print("Initializing simulation...")
    sim_gen = solve_annulus_diffusion()
    
    # Get initial state to setup the plot
    rho, T, t0 = next(sim_gen)

    # 2. Visualization Setup
    print("Setting up animation...")
    theta_vals = np.linspace(0, 2 * np.pi, NTHETA + 1)
    r_vals = np.array(rho)

    R_mesh, THETA_mesh = np.meshgrid(r_vals, theta_vals)
    X = R_mesh * np.cos(THETA_mesh)
    Y = R_mesh * np.sin(THETA_mesh)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Initial Plot
    Z_wrapped = np.concatenate([T, T[:, [0]]], axis=1)
    Z = Z_wrapped.T 
    
    quad = ax.pcolormesh(X, Y, Z, cmap='inferno', shading='gouraud', vmin=V_MIN, vmax=V_MAX)
    plt.colorbar(quad, label='Temperature')
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    title_text = ax.set_title(f"2D Heat Diffusion in Annulus (t = {t0:.3f}s)")

    def update(frame_data):
        rho_frame, T_frame, t_frame = frame_data
        
        # Wrap data
        Z_wrapped_f = np.concatenate([T_frame, T_frame[:, [0]]], axis=1)
        Z_f = Z_wrapped_f.T
        
        quad.set_array(Z_f.ravel())
        title_text.set_text(f"2D Heat Diffusion in Annulus (t = {t_frame:.3f}s)")
        return quad, title_text

    print("Generating video... (this may take a while)")
    anim = FuncAnimation(fig, update, frames=sim_gen, blit=False, cache_frame_data=False)
    
    # Save video
    try:
        anim.save("annulus_diffusion.gif", writer='pillow', fps=30)
        print("Video saved as annulus_diffusion.gif")
    except Exception as e2:
            print(f"Error saving gif: {e2}")

if __name__ == "__main__":
    main()