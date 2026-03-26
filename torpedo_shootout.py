import math
import numpy as np
from random import randrange
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from python_vehicle_simulator.vehicles import torpedo
from python_vehicle_simulator.lib.gnc import (
    attitudeEuler,
    crossFlowDrag,
    forceLiftDrag,
    gvect,
    m2c,
)

from fossen_solver import J_eta, R_b_to_n


# Command bank with some randomization

def wrap_to_pi(angle: float) -> float:
    """
    Wrap angle to [-pi, pi).
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def build_control_bank(sample_time: float, N: int) -> np.ndarray:
    """
    Build reference command bank:
        u[k] = [z_cmd, psi_cmd, n_rpm]

    z_cmd is positive-down depth in meters, psi_cmd is heading in radians,
    and n_rpm is the commanded propeller speed.
    """
    t = np.arange(N + 1) * sample_time
    u_bank = np.zeros((N + 1, 3), dtype=float)

    for k, tk in enumerate(t):
        if tk < 10.0:
            z_cmd = 0.0
            psi_cmd = 0.0
            rpm = 900.0 * (tk / 10.0)
        elif tk < 50.0:
            z_cmd = 5.0
            psi_cmd = np.deg2rad(5.0) * np.sin(0.05 * tk)
            rpm = 900.0 + 100.0 * np.sin(0.03 * tk)
        elif tk < 90.0:
            z_cmd = 10.0 + 2.0 * np.sin(0.04 * tk)
            psi_cmd = np.deg2rad(5.0) * 0.01 * randrange(100, 900, 25) * np.sin(0.10 * tk) + (
                0.05 * randrange(100, 900, 5) if tk % 25 == 0 else 0.0
            )
            rpm = 900.0 + 200.0 * np.sin(0.03 * tk)
        else:
            alpha = max(1.0 - (tk - 90.0) / 20.0, 0.0)
            z_cmd = alpha * 6.0
            psi_cmd = alpha * np.deg2rad(3.0) * np.sin(0.10 * tk)
            rpm = alpha * 900.0

        u_bank[k, :] = np.array([z_cmd, psi_cmd, rpm])

    return u_bank

# MSS to FossSim #
def depth_heading_autopilot(
    eta: np.ndarray,
    nu: np.ndarray,
    ref_cmd: np.ndarray,
    prev_act_cmd: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Convert depth-heading-RPM reference commands to actuator commands:
        u[k] = [delta_r_top, delta_r_bottom, delta_s_star, delta_s_port, n_rpm]

    A cascaded outer-loop depth controller generates a pitch reference, and an
    inner-loop pitch controller generates stern plane commands.
    """
    z = eta[2]
    theta = eta[4]
    psi = eta[5]

    w = nu[2]
    q = nu[4]
    r = nu[5]

    z_cmd = ref_cmd[0]
    psi_cmd = ref_cmd[1]
    rpm_cmd = ref_cmd[2]

    # depth outer loop
    e_z = z_cmd - z
    Kp_z = 0.08
    Kd_z = 0.12
    theta_max = np.deg2rad(20.0)

    # positive depth error => command nose-down pitch in NED
    theta_cmd = -(Kp_z * e_z + Kd_z * w)
    theta_cmd = np.clip(theta_cmd, -theta_max, theta_max)

    # pitch inner loop
    e_theta = theta_cmd - theta
    Kp_theta = -1.8
    Kd_theta = -0.9
    delta_s_max = np.deg2rad(25.0)

    delta_s = Kp_theta * e_theta + Kd_theta * q
    delta_s = np.clip(delta_s, -delta_s_max, delta_s_max)

    # heading loop
    e_psi = wrap_to_pi(psi_cmd - psi)
    Kp_psi = 1.2
    Kd_psi = -0.8
    delta_r_max = np.deg2rad(20.0)

    delta_r = Kp_psi * e_psi + Kd_psi * r
    delta_r = np.clip(delta_r, -delta_r_max, delta_r_max)

    # rate limiting
    delta_r_rate_max = np.deg2rad(30.0)
    delta_s_rate_max = np.deg2rad(30.0)
    rpm_rate_max = 400.0

    delta_r_prev = prev_act_cmd[0]
    delta_s_prev = prev_act_cmd[2]
    rpm_prev = prev_act_cmd[4]

    delta_r = np.clip(
        delta_r,
        delta_r_prev - delta_r_rate_max * dt,
        delta_r_prev + delta_r_rate_max * dt,
    )
    delta_s = np.clip(
        delta_s,
        delta_s_prev - delta_s_rate_max * dt,
        delta_s_prev + delta_s_rate_max * dt,
    )
    rpm_cmd = np.clip(
        rpm_cmd,
        rpm_prev - rpm_rate_max * dt,
        rpm_prev + rpm_rate_max * dt,
    )

    return np.array([delta_r, -delta_r, delta_s, -delta_s, rpm_cmd], dtype=float)


def fin_tau_from_delta(fin_obj, delta_actual: float, nu_r: np.ndarray) -> np.ndarray:
    """
    Convert angle delta command to torque command.
    """
    vx, vy, vz = nu_r[:3]
    angle_rad = fin_obj.angle_rad

    vy_rot = np.sqrt((vy * np.sin(angle_rad)) ** 2 + (vz * np.cos(angle_rad)) ** 2)
    ur = np.sqrt(vx**2 + vy_rot**2)

    f = 0.5 * fin_obj.rho * fin_obj.area * fin_obj.CL * delta_actual * ur**2
    fy = np.sin(angle_rad) * f
    fz = -np.cos(angle_rad) * f
    F = np.array([0.0, fy, fz])
    torque = np.cross(fin_obj.R, F)
    return np.concatenate((F, torque))


def thruster_tau_from_rpm(thruster_obj, n_actual: float, nu: np.ndarray) -> np.ndarray:
    """
    Convert RPM command to thruster force command.
    """
    U = np.linalg.norm(nu[:3])
    n = float(np.clip(n_actual, -thruster_obj.nMax, thruster_obj.nMax))

    # propeller params
    D_prop = 0.14
    t_prop = 0.1
    n_rps = n / 60.0
    Va = 0.944 * U
    Ja_max = 0.6632

    KT_0 = 0.4566
    KQ_0 = 0.0700
    KT_max = 0.1798
    KQ_max = 0.0312

    if n_rps > 0.0:
        X_prop = thruster_obj.rho * D_prop**4 * (
            KT_0 * abs(n_rps) * n_rps
            + (KT_max - KT_0) / Ja_max * (Va / D_prop) * abs(n_rps)
        )
        K_prop = thruster_obj.rho * D_prop**5 * (
            KQ_0 * abs(n_rps) * n_rps
            + (KQ_max - KQ_0) / Ja_max * (Va / D_prop) * abs(n_rps)
        )
    else:
        X_prop = thruster_obj.rho * D_prop**4 * KT_0 * abs(n_rps) * n_rps
        K_prop = thruster_obj.rho * D_prop**5 * KQ_0 * abs(n_rps) * n_rps

    return np.array([(1.0 - t_prop) * X_prop, 0.0, 0.0, K_prop / 10.0, 0.0, 0.0])


def actuator_derivatives(vehicle, u_actual: np.ndarray, u_cmd: np.ndarray) -> np.ndarray:
    """
    Computes fin and thruster angle rates assuming first-order response.
    """
    du = np.zeros(5, dtype=float)
    for i, act in enumerate(vehicle.actuators[:4]):
        du[i] = (u_cmd[i] - u_actual[i]) / act.T_delta
    thr = vehicle.actuators[4]
    du[4] = (u_cmd[4] - u_actual[4]) / thr.T_n
    return du


def clip_u_actual(vehicle, u_actual: np.ndarray) -> np.ndarray:
    """
    Implements forward thruster saturation.
    """
    out = np.array(u_actual, dtype=float).copy()
    for i, act in enumerate(vehicle.actuators[:4]):
        out[i] = np.clip(out[i], -act.deltaMax, act.deltaMax)
    thr = vehicle.actuators[4]
    out[4] = np.clip(out[4], -thr.nMax, thr.nMax)
    return out


def torpedo_rhs_rk4(vehicle, x: np.ndarray, u_cmd: np.ndarray) -> np.ndarray:
    """
    Single iteration of Runge-Kutta for submarine state:
    Inputs:  State vector X: [eta | nu | command]'
    Outputs: State vector derivative X_dot: [eta_dot | nu_dot | command_dot]'
    """
    eta = x[0:6]
    nu = x[6:12]
    u_actual = x[12:17]

    # Current model from MSS
    u_c = vehicle.V_c * math.cos(vehicle.beta_c - eta[5])
    v_c = vehicle.V_c * math.sin(vehicle.beta_c - eta[5])
    nu_c = np.array([u_c, v_c, 0.0, 0.0, 0.0, 0.0])
    Dnu_c = np.array([nu[5] * v_c, -nu[5] * u_c, 0.0, 0.0, 0.0, 0.0])
    nu_r = nu - nu_c

    alpha = math.atan2(nu_r[2], nu_r[0])
    U_r = np.linalg.norm(nu_r[:3])

    # Coriolis formula from MSS
    CRB = m2c(vehicle.MRB, nu_r)
    CA = m2c(vehicle.MA, nu_r)
    CA[4, 0] = 0.0
    CA[0, 4] = 0.0
    CA[4, 2] = 0.0
    CA[2, 4] = 0.0
    CA[5, 0] = 0.0
    CA[0, 5] = 0.0
    CA[5, 1] = 0.0
    CA[1, 5] = 0.0
    C = CRB + CA

    # Damping formula from MSS
    D = np.diag([
        vehicle.M[0, 0] / vehicle.T_surge,
        vehicle.M[1, 1] / vehicle.T_sway,
        vehicle.M[2, 2] / vehicle.T_heave,
        vehicle.M[3, 3] * 2.0 * vehicle.zeta_roll * vehicle.w_roll,
        vehicle.M[4, 4] * 2.0 * vehicle.zeta_pitch * vehicle.w_pitch,
        vehicle.M[5, 5] / vehicle.T_yaw,
    ])
    D[0, 0] *= math.exp(-3.0 * U_r)
    D[1, 1] *= math.exp(-3.0 * U_r)

    tau_liftdrag = forceLiftDrag(vehicle.diam, vehicle.S, vehicle.CD_0, alpha, U_r)
    tau_crossflow = crossFlowDrag(vehicle.L, vehicle.diam, vehicle.diam, nu_r)
    g = gvect(vehicle.W, vehicle.B, eta[4], eta[3], vehicle.r_bg, vehicle.r_bb)

    # Actuator-generated tau from current actuator states
    tau_act = np.zeros(6, dtype=float)
    for i, fin_obj in enumerate(vehicle.actuators[:4]):
        tau_act += fin_tau_from_delta(fin_obj, u_actual[i], nu_r)
    tau_act += thruster_tau_from_rpm(vehicle.actuators[4], u_actual[4], nu)

    tau_sum = tau_act + tau_liftdrag + tau_crossflow - (C + D) @ nu_r - g
    nu_dot = Dnu_c + vehicle.Minv @ tau_sum
    eta_dot = J_eta(eta) @ nu
    u_dot = actuator_derivatives(vehicle, u_actual, u_cmd)

    return np.concatenate((eta_dot, nu_dot, u_dot))


## Integrators
def rk4_step(vehicle, x: np.ndarray, u_cmd: np.ndarray, dt: float) -> np.ndarray:
    """
    Fourth-order Runge-Kutta single step for FossSim dynamics solver: 
      RK4 for ord(1) linear diffEQ implemented as: 
        y(k + del) = (k1/6 + k2/6 + k3/6 + k4/6) * del 
        WHERE
          k1 = f( y(t0,t0) )                          
          k2 = f( y(t0 + del * k1/2, t0 + (del * 0.5)) )
          k3 = f( y(t0 + del * k2/2, t0 + (del * 0.5)) )
          k4 = f( y(t0 + del * k3,   t0 +  del)        )
    """
    k1 = torpedo_rhs_rk4(vehicle, x, u_cmd)
    k2 = torpedo_rhs_rk4(vehicle, x + 0.5 * dt * k1, u_cmd)
    k3 = torpedo_rhs_rk4(vehicle, x + 0.5 * dt * k2, u_cmd)
    k4 = torpedo_rhs_rk4(vehicle, x + dt * k3, u_cmd)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next[12:17] = clip_u_actual(vehicle, x_next[12:17])
    return x_next


def mss_step(vehicle, eta: np.ndarray, nu: np.ndarray, u_cmd: np.ndarray, dt: float):
    """
    Single step driver for MSS vehicle.dynamics() solver.
    """
    nu_next, u_actual_next = vehicle.dynamics(eta.copy(), nu.copy(), vehicle.u_actual.copy(), u_cmd, dt)
    eta_next = attitudeEuler(eta.copy(), nu_next.copy(), dt)
    return eta_next, nu_next.copy(), u_actual_next.copy()


# Shootout
def run_shootout(sample_time: float = 0.02, N: int = 6000):
    """
    Runs the MSS vs FossSim torpedo shootout.
    Inputs:
            sample_time: Timestep in S
            N: Number of timesteps
    """

    MSS_Torpedo = torpedo(controlSystem="stepInput")
    FossSim_Torpedo = torpedo(controlSystem="stepInput")

    u_bank = build_control_bank(sample_time, N)
    sim_time = np.arange(N + 1) * sample_time

    eta_MSS = np.zeros(6, dtype=float)
    nu_MSS = np.zeros(6, dtype=float)

    x_FossSim = np.zeros(17, dtype=float)

    eta_MSS_log = np.zeros((N + 1, 6), dtype=float)
    nu_MSS_log = np.zeros((N + 1, 6), dtype=float)
    ua_MSS_log = np.zeros((N + 1, 5), dtype=float)

    eta_FossSim_log = np.zeros((N + 1, 6), dtype=float)
    nu_FossSim_log = np.zeros((N + 1, 6), dtype=float)
    ua_FossSim_log = np.zeros((N + 1, 5), dtype=float)

    act_cmd_log = np.zeros((N + 1, 5), dtype=float)
    act_cmd_prev = np.zeros(5, dtype=float)

    eta_MSS_log[0] = eta_MSS
    nu_MSS_log[0] = nu_MSS
    ua_MSS_log[0] = MSS_Torpedo.u_actual.copy()

    eta_FossSim_log[0] = x_FossSim[0:6]
    nu_FossSim_log[0] = x_FossSim[6:12]
    ua_FossSim_log[0] = x_FossSim[12:17]
    act_cmd_log[0] = act_cmd_prev

    for k in range(N):
        ref_cmd = u_bank[k]
        u_cmd = depth_heading_autopilot(eta_MSS, nu_MSS, ref_cmd, act_cmd_prev, sample_time)
        act_cmd_prev = u_cmd.copy()

        eta_MSS, nu_MSS, ua_MSS = mss_step(MSS_Torpedo, eta_MSS, nu_MSS, u_cmd, sample_time)
        x_FossSim = rk4_step(FossSim_Torpedo, x_FossSim, u_cmd, sample_time)

        eta_MSS_log[k + 1] = eta_MSS
        nu_MSS_log[k + 1] = nu_MSS
        ua_MSS_log[k + 1] = ua_MSS

        eta_FossSim_log[k + 1] = x_FossSim[0:6]
        nu_FossSim_log[k + 1] = x_FossSim[6:12]
        ua_FossSim_log[k + 1] = x_FossSim[12:17]
        act_cmd_log[k + 1] = act_cmd_prev

        if np.linalg.norm(nu_MSS) > 1e3 or np.linalg.norm(x_FossSim[6:12]) > 1e3:
            print(f"Blow-up detected at step {k}, t={sim_time[k]:.2f} s")
            sim_time = sim_time[:k + 2]
            eta_MSS_log = eta_MSS_log[:k + 2]
            nu_MSS_log = nu_MSS_log[:k + 2]
            ua_MSS_log = ua_MSS_log[:k + 2]
            eta_FossSim_log = eta_FossSim_log[:k + 2]
            nu_FossSim_log = nu_FossSim_log[:k + 2]
            ua_FossSim_log = ua_FossSim_log[:k + 2]
            act_cmd_log = act_cmd_log[:k + 2]
            break

    return sim_time, eta_MSS_log, nu_MSS_log, ua_MSS_log, eta_FossSim_log, nu_FossSim_log, ua_FossSim_log, u_bank[:len(sim_time)], act_cmd_log

# Plot Normalization
def initialize_quiver(ax, sim_time: np.ndarray, sv: np.ndarray, offset=0, q_color='red') -> np.ndarray:
    """
    Sets up quivers for velocity vectors in 3D plot.
    Inputs:
        ax: Matplotlib 3D axes
        sim_time: base time scale
        sv: state vector array, columns [nu(0:6), eta(6:12)]
            position in sv[:, 6:9]
            Euler angles in sv[:, 9:12]
        q_color: quiver color

    Outputs:
        arrow_data: array of points used for axis scaling
    """
    sv = np.asarray(sv, dtype=float)
    sim_time = np.asarray(sim_time)

    step = 50
    idx = np.arange(0, min(len(sim_time), len(sv)), step)

    pos = sv[idx, 6:9]
    vecs = np.zeros((len(idx), 3), dtype=float)

    for j, k in enumerate(idx):
        phi, theta, psi = sv[k, 9:12]
        R_bn = R_b_to_n(phi, theta, psi)
        vecs[j] = R_bn @ sv[k, 0:3]

    norms = np.linalg.norm(vecs, axis=1)
    dirs = np.zeros_like(vecs)
    mask = norms > 1e-12
    dirs[mask] = vecs[mask] / norms[mask, None]

    ax.quiver(
        pos[:, 0], pos[:, 1], pos[:, 2],
        dirs[:, 0], dirs[:, 1], dirs[:, 2],
        normalize=False,
        color=q_color
    )

    arrow_tips = pos + 4*dirs

    # return points relevant for autoscaling
    arrow_data = np.vstack([sv[:, 6:9], pos, arrow_tips])
    return arrow_data

def set_3d_axes_equal_meters(ax, xyz: np.ndarray, pad_m: float = 0.5) -> None:
    xyz = np.asarray(xyz, dtype=float)
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    center = 0.5 * (mins + maxs)
    span = max(np.max(maxs - mins), 1e-6)
    radius = 0.5 * span + pad_m
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))

def main():
    sim_time, eta_MSS, nu_MSS, ua_MSS, eta_FossSim, nu_FossSim, ua_FossSim, u_bank, act_cmd_log = run_shootout()

    e_eta = eta_FossSim - eta_MSS
    e_nu = nu_FossSim - nu_MSS
    e_ua = ua_FossSim - ua_MSS

    print("RMS eta error [x,y,z,phi,theta,psi]:")
    print(np.sqrt(np.mean(e_eta**2, axis=0)))
    print("RMS nu error [u,v,w,p,q,r]:")
    print(np.sqrt(np.mean(e_nu**2, axis=0)))
    print("RMS actuator-state error [5 channels]:")
    print(np.sqrt(np.mean(e_ua**2, axis=0)))

    print("\nFinal eta error:")
    print(e_eta[-1])
    print("Final nu error:")
    print(e_nu[-1])
    print("Final actuator-state error:")
    print(e_ua[-1])

    ## 3D trajectory plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(eta_MSS[:, 0], eta_MSS[:, 1], eta_MSS[:, 2], label="MSS_Torpedo", color='green')
    ax.plot(eta_FossSim[:, 0], eta_FossSim[:, 1], eta_FossSim[:, 2], "--", label="FossSim_Torpedo", color='blue')
    
    # Quiver setup
    sv_MSS = np.concatenate((nu_MSS, eta_MSS), axis=1)
    sv_FossSim = np.concatenate((nu_FossSim, eta_FossSim), axis=1)
    arrow_data_MSS = initialize_quiver(ax, sim_time, sv_MSS, offset=1, q_color='purple')
    arrow_data_FossSim = initialize_quiver(ax, sim_time, sv_FossSim)
    set_3d_axes_equal_meters(ax, np.concatenate((arrow_data_MSS, arrow_data_FossSim), axis=0), pad_m=0.5)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("MSS_Torpedo vs FossSim_Torpedo trajectory")
    ax.grid(True)
    ax.legend()
    ax.invert_zaxis()

    ## Speed and depth
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(sim_time, nu_MSS[:, 0], label="u_MSS")
    plt.plot(sim_time, nu_FossSim[:, 0], "--", label="u_FossSim")
    plt.ylabel("Surge u [m/s]")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(sim_time, u_bank[:, 0], "k:", label="z_cmd")
    plt.plot(sim_time, eta_MSS[:, 2], label="z_MSS")
    plt.plot(sim_time, eta_FossSim[:, 2], "--", label="z_FossSim")
    plt.xlabel("Time [s]")
    plt.ylabel("Depth z [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    ## Horizontal track
    plt.figure()
    plt.plot(eta_MSS[:, 0], eta_MSS[:, 1], label="MSS_Torpedo")
    plt.plot(eta_FossSim[:, 0], eta_FossSim[:, 1], "--", label="FossSim_Torpedo")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Horizontal track")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    ## Velocity errors
    plt.figure()
    labels = ["u", "v", "w", "p", "q", "r"]
    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.plot(sim_time, e_nu[:, i])
        plt.ylabel(f"{labels[i]} err")
        plt.grid(True)
    plt.xlabel("Time [s]")
    plt.tight_layout()

    ## Command tracking visualization
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(sim_time, u_bank[:, 0], label="cmd depth z [m]")
    plt.plot(sim_time, eta_MSS[:, 2], "--", label="MSS_Torpedo depth z [m]")
    plt.plot(sim_time, eta_FossSim[:, 2], ":", label="FossSim_Torpedo depth z [m]")
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(sim_time, np.rad2deg(act_cmd_log[:, 0]), label="cmd rudder top [deg]")
    plt.plot(sim_time, np.rad2deg(ua_MSS[:, 0]), label="MSS_Torpedo actual top [deg]")
    plt.plot(sim_time, np.rad2deg(ua_FossSim[:, 0]), ":", label="FossSim_Torpedo actual top [deg]")
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(sim_time, np.rad2deg(act_cmd_log[:, 2]), label="cmd stern star [deg]")
    plt.plot(sim_time, np.rad2deg(ua_MSS[:, 2]), "--", label="MSS_Torpedo actual star [deg]")
    plt.plot(sim_time, np.rad2deg(ua_FossSim[:, 2]), ":", label="FossSim_Torpedo actual star [deg]")
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(sim_time, u_bank[:, 2], label="cmd rpm")
    plt.plot(sim_time, ua_MSS[:, 4], "--", label="MSS_Torpedo actual rpm")
    plt.plot(sim_time, ua_FossSim[:, 4], ":", label="FossSim_Torpedo actual rpm")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()