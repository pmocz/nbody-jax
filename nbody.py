import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jaxopt import ScipyMinimize

"""
Create Your Own Differentiable N-body Simulation (With Python/JAX)
Philip Mocz (2025) @pmocz

Simulate orbits of stars interacting due to gravity.
The code evolves particles according to pairwise interactions following Newton's Law of Gravity.

Auto-differentiability allows for solving inverse problems!
E.g. find the initial velocities of particles that result in a target distribution at t=1.0
"""

# Global parameters
N = 360  # number of particles
t_end = 1.0  # time at which simulation ends
dt = 0.01  # timestep
softening = 0.1  # softening length
G = 1.0  # Newton's gravitational constant
Nt = int(jnp.ceil(t_end / dt))  # number of timesteps


@jax.jit
def get_acc(pos, mass):
    """
    Calculate the acceleration on each particle according to Newton's Law
    pos        is an N x 3 matrix of positions
    mass       is an N x 1 vector of masses
    acc        is an N x 3 matrix of accelerations
    """
    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2) ** (-1.5)

    # acceleration components
    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

    # pack together the acceleration components
    acc = jnp.hstack((ax, ay, az))

    return acc


@jax.jit
def get_energy(pos, vel, mass):
    """
    Get kinetic energy (KE) and potential energy (PE) of simulation
    pos   is N x 3 matrix of positions
    vel   is N x 3 matrix of velocities
    mass  is an N x 1 vector of masses
    KE    is the kinetic energy of the system
    PE    is the potential energy of the system
    """
    # Kinetic Energy:
    KE = 0.5 * jnp.sum(jnp.sum(mass * vel**2))

    # Potential Energy:

    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r for all particle pairwise particle separations
    inv_r = 1.0 / jnp.sqrt(dx**2 + dy**2 + dz**2)

    # sum over upper triangle, to count each interaction only once
    PE = G * jnp.sum(jnp.sum(jnp.triu(-(mass * mass.T) * inv_r, 1)))

    return KE, PE


@jax.jit
def leapfrog(i, state):
    """
    Take one timestep of the leapfrog integration scheme
    i          is the iteration number
    state      is a tuple of (pos, vel, acc, mass)
    """
    pos, vel, acc, mass = state

    # (1/2) kick
    vel += acc * dt / 2.0

    # drift
    pos += vel * dt

    # update accelerations
    acc = get_acc(pos, mass)

    # (1/2) kick
    vel += acc * dt / 2.0

    return pos, vel, acc, mass


@jax.jit
def do_simulation(pos, vel, mass, t):
    """
    Run the simulation for Nt timesteps
    pos        is an N x 3 matrix of positions
    vel        is an N x 3 matrix of velocities
    mass       is an N x 1 vector of masses
    t          is the time
    """

    # calculate initial gravitational accelerations
    acc = get_acc(pos, mass)

    # advance the simulation by Nt timesteps
    state = jax.lax.fori_loop(0, Nt, leapfrog, init_val=(pos, vel, acc, mass))
    pos, vel, acc, mass = state

    # update time
    t += Nt * dt

    return pos, vel, acc, mass, t


@jax.jit
def loss_function(vel, pos, mass, target):
    """
    Loss function for optimization:
    Find initial conditions for velocity that results the target distribution
    Note: the 1st argument is the one that will be optimized over
    vel        is an N x 3 matrix of velocities
    pos        is an N x 3 matrix of positions
    mass       is an N x 1 vector of masses
    target     is an m x m matrix representing the target distribution
    """

    # Set initial condition
    t = 0
    np.random.seed(17)  # set the random number generator seed
    mass = 20.0 * jnp.ones((N, 1)) / N  # total mass of particles is 20
    pos = jnp.array(np.random.randn(N, 3))  # randomly selected positions and velocities

    # Convert to center-of-mass frame
    vel -= jnp.mean(mass * vel, 0) / jnp.mean(mass)

    # Carry out simulation
    pos, _, _, _, _ = do_simulation(pos, vel, mass, t)

    # Bin the resulting 2D positions and estimate the density
    pos2d = pos[:, 0:2].T

    m = target.shape[0]  # number of bins

    kde = jax.scipy.stats.gaussian_kde(pos2d)
    xlin = jnp.linspace(-2.0, 2.0, m)
    mx, my = jnp.meshgrid(xlin, xlin, indexing="ij")
    counts = kde.evaluate(jnp.array([mx.reshape(-1), my.reshape(-1)])).reshape((m, m))
    counts /= jnp.sum(counts)

    return jnp.sum((counts - target) ** 2)


def main():
    """N-body simulation with Inverse Problem
    Find initial velocities that result in a target distribution (a heart) at t = 1.0
    """

    # Set initial conditions
    np.random.seed(17)  # set the random number generator seed
    mass = 20.0 * jnp.ones((N, 1)) / N  # total mass of particles is 20
    pos = jnp.array(np.random.randn(N, 3))  # randomly selected positions and velocities
    vel = jnp.array(np.random.randn(N, 3))
    vel -= jnp.mean(mass * vel, 0) / jnp.mean(mass)

    # target distribution (heart-shape)
    n_bins = 80
    xlin = jnp.linspace(-2, 2, n_bins)
    xx, yy = jnp.meshgrid(xlin, xlin, indexing="ij")
    target = (xx**2 + yy**2 - 1.0) ** 3 - xx**2 * yy**3 < 0.0
    target = target.astype(float) / jnp.sum(target)

    # run the optimization:

    def print_step_info(intermediate_vel):
        print("loss:", loss_function(intermediate_vel, pos, mass, target))

    optimizer = ScipyMinimize(
        method="l-bfgs-b", fun=loss_function, tol=1e-8, callback=print_step_info
    )

    print("Optimizing initial conditions...")
    sol = optimizer.run(vel, pos, mass, target)
    print("Optimization complete.")

    # Setup the simulation with the optimized initial conditions
    np.random.seed(17)
    pos = jnp.array(np.random.randn(N, 3))
    mass = 20.0 * jnp.ones((N, 1)) / N
    vel = sol.params
    vel -= jnp.mean(mass * vel, 0) / jnp.mean(mass)
    acc = get_acc(pos, mass)

    # plot initial velocities as arrows
    fig = plt.figure(figsize=(4, 5), dpi=80)
    ax = fig.add_subplot(111)
    ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1])
    plt.scatter(pos[:, 0], pos[:, 1], s=10, color="blue")
    ax.set(xlim=(-2, 2), ylim=(-2, 2))
    ax.set_aspect("equal", "box")
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Initial velocities")
    plt.savefig("heart-ics.png", dpi=240)
    plt.pause(1.0)

    # calculate initial energy of system
    KE, PE = get_energy(pos, vel, mass)

    # save energies, particle orbits for plotting trails
    pos_save = np.zeros((N, 3, Nt + 1))
    KE_save = np.zeros(Nt + 1)
    PE_save = np.zeros(Nt + 1)
    t_all = np.arange(Nt + 1) * dt
    pos_save[:, :, 0] = pos
    KE_save[0] = KE
    PE_save[0] = PE

    # switch on to plot in real time
    plot_realtime = True

    # prep figure
    fig = plt.figure(figsize=(4, 5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])
    ax2 = plt.subplot(grid[2, 0])

    # Simulation Main Loop
    for i in range(Nt):
        pos, vel, acc, mass = leapfrog(i, (pos, vel, acc, mass))

        # get energy of system
        KE, PE = get_energy(pos, vel, mass)

        # save energies, positions for plotting trail
        pos_save[:, :, i + 1] = pos
        KE_save[i + 1] = KE
        PE_save[i + 1] = PE

        # plot in real time
        if plot_realtime or (i == Nt - 1):
            plt.sca(ax1)
            plt.cla()
            xx = pos_save[:, 0, max(i - 50, 0) : i + 1]
            yy = pos_save[:, 1, max(i - 50, 0) : i + 1]
            plt.scatter(xx, yy, s=1, color=[0.7, 0.7, 1])
            plt.scatter(pos[:, 0], pos[:, 1], s=10, color="blue")
            ax1.set(xlim=(-2, 2), ylim=(-2, 2))
            ax1.set_aspect("equal", "box")
            ax1.set_xticks([-2, -1, 0, 1, 2])
            ax1.set_yticks([-2, -1, 0, 1, 2])

            plt.sca(ax2)
            plt.cla()
            plt.scatter(
                t_all, KE_save, color="red", s=1, label="KE" if i == Nt - 1 else ""
            )
            plt.scatter(
                t_all, PE_save, color="blue", s=1, label="PE" if i == Nt - 1 else ""
            )
            plt.scatter(
                t_all,
                KE_save + PE_save,
                color="black",
                s=1,
                label="Etot" if i == Nt - 1 else "",
            )
            ax2.set(xlim=(0, t_end), ylim=(-300, 300))
            ax2.set_aspect(0.0007)

            plt.pause(0.001)

    # add labels/legend
    plt.sca(ax2)
    plt.xlabel("time")
    plt.ylabel("energy")
    ax2.legend(loc="upper right")

    # Save figure
    plt.savefig("heart-nbody.png", dpi=240)
    plt.show()

    return


if __name__ == "__main__":
    main()
