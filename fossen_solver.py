# Author: Zaid Yasin @ DSIM Lab
import numpy as np

"""
Important Reference Information:

Fossen Dynamics: https://fossen.biz/html/marineCraftModel.html
SNAME Manuevering Notation: https://www.usna.edu/NAOE/_files/documents/Courses/EN455/AY20_Notes/EN455CourseNotesAY20_ManeuveringNotation.pdf
3-2-1 Euler Angle Parameterization: https://personalpages.surrey.ac.uk/t.bridges/SLOSH/3-2-1-Eulerangles.pdf
"""


def skew(v: np.ndarray) -> np.ndarray:
    """
    Returns the skew-symmetric matrix associated with a 3-vector.
    """
    v = np.asarray(v, dtype=float).reshape(3)
    x, y, z = v
    return np.array([
        [0.0, -z,   y],
        [z,    0.0, -x],
        [-y,   x,   0.0]
    ])


def _as_vec3(name: str, v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    if v.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {v.shape}.")
    return v


def _as_3x3(name: str, A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {A.shape}.")
    return A


def _as_6x6(name: str, A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.shape != (6, 6):
        raise ValueError(f"{name} must have shape (6, 6), got {A.shape}.")
    return A


def _check_symmetric(name: str, A: np.ndarray, atol: float = 1e-9) -> None:
    if not np.allclose(A, A.T, atol=atol):
        raise ValueError(f"{name} must be symmetric.")


def _check_invertible(name: str, A: np.ndarray, cond_warn: float = 1e12) -> None:
    rank = np.linalg.matrix_rank(A)
    if rank < A.shape[0]:
        raise ValueError(f"{name} is singular.")
    cond = np.linalg.cond(A)
    if not np.isfinite(cond):
        raise ValueError(f"{name} has non-finite condition number.")
    if cond > cond_warn:
        print(f"Warning: {name} is ill-conditioned (cond={cond:.3e}).")


def R_b_to_n(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Returns the BODY-to-NED rotation matrix for 3-2-1 Euler angles.
    """
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    return np.array([
        [cpsi * cth,
         -spsi * cphi + cpsi * sth * sphi,
         spsi * sphi + cpsi * cphi * sth],

        [spsi * cth,
         cpsi * cphi + sphi * sth * spsi,
         -cpsi * sphi + sth * spsi * cphi],

        [-sth,
         cth * sphi,
         cth * cphi]
    ])


def R_n_to_b(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Returns the NED-to-BODY rotation matrix.
    """
    return R_b_to_n(phi, theta, psi).T

def wrap_to_pi(angle: float) -> float:
    """
    Wrap an angle to the interval [-pi, pi).
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def wrap_euler_error(e: np.ndarray) -> np.ndarray:
    """
    Wraps Euler-angle components of a 6-vector error.
    Expected input:
        [x, y, z, phi, theta, psi]
    """
    e = np.asarray(e, dtype=float).copy().reshape(6)
    e[3] = wrap_to_pi(e[3])
    e[4] = wrap_to_pi(e[4])
    e[5] = wrap_to_pi(e[5])
    return e

def T_euler(phi: float, theta: float, eps: float = 1e-6) -> np.ndarray:
    """
    Returns the mapping from body angular rates [p, q, r] to
    Euler angle rates [phi_dot, theta_dot, psi_dot] for 3-2-1 angles.
    """
    sphi = np.sin(phi)
    cphi = np.cos(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)

    if abs(cth) < eps:
        cth = eps if cth >= 0.0 else -eps

    return np.array([
        [1.0, sphi * sth / cth, cphi * sth / cth],
        [0.0, cphi,            -sphi],
        [0.0, sphi / cth,       cphi / cth]
    ])


def J_eta(eta: np.ndarray) -> np.ndarray:
    """
    Returns the configuration-dependent kinematic mapping such that

        eta_dot = J(eta) @ nu

    where:
        eta = [x, y, z, phi, theta, psi]
        nu  = [u, v, w, p, q, r]
    """
    eta = np.asarray(eta, dtype=float).reshape(6)
    _, _, _, phi, theta, psi = eta

    return np.block([
        [R_b_to_n(phi, theta, psi), np.zeros((3, 3))],
        [np.zeros((3, 3)),          T_euler(phi, theta)]
    ])


def m2c(M: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """
    Returns the partition-based Coriolis / centripetal matrix associated
    with a constant 6x6 inertia-like matrix.

    Mainly intended for constant added mass.
    """
    M = np.asarray(M, dtype=float).reshape(6, 6)
    nu = np.asarray(nu, dtype=float).reshape(6)

    M = 0.5 * (M + M.T)

    nu1 = nu[0:3]
    nu2 = nu[3:6]

    M11 = M[0:3, 0:3]
    M12 = M[0:3, 3:6]
    M21 = M[3:6, 0:3]
    M22 = M[3:6, 3:6]

    t1 = M11 @ nu1 + M12 @ nu2
    t2 = M21 @ nu1 + M22 @ nu2

    C = np.zeros((6, 6))
    C[0:3, 3:6] = -skew(t1)
    C[3:6, 0:3] = -C[0:3, 3:6]
    C[3:6, 3:6] = -skew(t2)
    return C


def diagonal_added_mass_from_scalars(ma_u=0.0, ma_v=0.0, ma_w=0.0, 
                                     ma_p=0.0, ma_q=0.0, ma_r=0.0) -> np.ndarray:
    """
    Returns a diagonal added-mass matrix assembled directly as MA.

    Values are treated as positive magnitudes.
    """
    vals = np.array([ma_u, ma_v, ma_w, ma_p, ma_q, ma_r], dtype=float)
    if np.any(vals < 0.0):
        raise ValueError("Added-mass magnitudes must be nonnegative.")
    return np.diag(vals)


def diagonal_added_mass_heuristic(mass: float, Ix: float, Iy: float, Iz: float, 
                                  surge_ratio: float = 0.10, sway_ratio: float = 0.60,
                                  heave_ratio: float = 0.60, roll_ratio: float = 0.05,
                                  pitch_ratio: float = 0.25, yaw_ratio: float = 0.25,) -> np.ndarray:
    """
    Returns a diagonal added-mass approximation for a compact
    underwater vehicle.
    """
    if min(mass, Ix, Iy, Iz) <= 0.0:
        raise ValueError("mass and inertias must be positive.")

    return np.diag([
        surge_ratio * mass,
        sway_ratio * mass,
        heave_ratio * mass,
        roll_ratio * Ix,
        pitch_ratio * Iy,
        yaw_ratio * Iz
    ])


class Fossen6DOFParams:
    """
    Represents a 6-DOF marine craft model using explicit physical parameters.

    State:
        x = [eta; nu]
        eta = [x, y, z, phi, theta, psi]
        nu  = [u, v, w, p, q, r]

    Dynamics:
        eta_dot = J(eta) @ nu
        M @ nu_dot + C_RB(nu) @ nu + C_A(nu) @ nu + d(nu) + g(eta) = tau

    Conventions
    -----------
    - BODY-frame velocities, NED pose.
    - z is positive downward.
    - W and B are positive magnitudes in newtons.
    - MA is the assembled added-mass matrix used directly in M = MRB + MA.
    """

    def __init__(
        self,
        m: float,
        r_G: np.ndarray,
        I_G: np.ndarray,
        W: float,
        B: float,
        r_B: np.ndarray,
        MA: np.ndarray = None,
        D_lin: np.ndarray = None,
        D_quad: np.ndarray = None,
    ):
        if m <= 0.0:
            raise ValueError("m must be positive.")
        if W < 0.0 or B < 0.0:
            raise ValueError("W and B must be nonnegative.")

        self.m = float(m)
        self.r_G = _as_vec3("r_G", r_G)
        self.I_G = _as_3x3("I_G", I_G)
        _check_symmetric("I_G", self.I_G)

        self.W = float(W)
        self.B = float(B)
        self.r_B = _as_vec3("r_B", r_B)

        if MA is None:
            MA = np.zeros((6, 6))
        if D_lin is None:
            D_lin = np.zeros((6, 6))
        if D_quad is None:
            D_quad = np.zeros((6, 6))

        self.MA = _as_6x6("MA", MA)
        self.D_lin = _as_6x6("D_lin", D_lin)
        self.D_quad = _as_6x6("D_quad", D_quad)

        _check_symmetric("MA", self.MA)

        self.MRB = self._build_MRB()
        self.M = self.MRB + self.MA

        _check_symmetric("MRB", self.MRB)
        _check_symmetric("M", self.M)
        _check_invertible("M", self.M)

        self._no_damping = (
            np.allclose(self.D_lin, 0.0) and np.allclose(self.D_quad, 0.0)
        )

    def _build_MRB(self) -> np.ndarray:
        """
        Returns the rigid-body mass matrix about the BODY origin,
        given the inertia tensor about the center of gravity.
        """
        Srg = skew(self.r_G)
        I_O = self.I_G - self.m * (Srg @ Srg)

        return np.block([
            [self.m * np.eye(3),   -self.m * Srg],
            [self.m * Srg,          I_O]
        ])

    @staticmethod
    def _coriolis_rigid_body(
        m: float,
        r_G: np.ndarray,
        I_G: np.ndarray,
        nu: np.ndarray
    ) -> np.ndarray:
        """
        Returns the rigid-body Coriolis / centripetal matrix using
        the standard Fossen expression with inertia defined about CG.
        """
        nu = np.asarray(nu, dtype=float).reshape(6)
        nu1 = nu[0:3]
        nu2 = nu[3:6]

        h1 = m * (nu1 + np.cross(nu2, r_G))
        h2 = I_G @ nu2 + m * np.cross(r_G, nu1)

        C = np.zeros((6, 6))
        C[0:3, 3:6] = -skew(h1)
        C[3:6, 0:3] =  skew(h1)
        C[3:6, 3:6] = -skew(h2)
        return C

    def C_RB(self, nu: np.ndarray) -> np.ndarray:
        """
        Returns the rigid-body Coriolis / centripetal matrix.
        """
        return self._coriolis_rigid_body(self.m, self.r_G, self.I_G, nu)

    def C_A(self, nu: np.ndarray) -> np.ndarray:
        """
        Returns the added-mass Coriolis / centripetal matrix.
        """
        return m2c(self.MA, nu)

    def C_total(self, nu: np.ndarray) -> np.ndarray:
        """
        Returns the total Coriolis / centripetal matrix.
        """
        return self.C_RB(nu) + self.C_A(nu)

    def d_nu(self, nu: np.ndarray) -> np.ndarray:
        """
        Returns the damping force / moment vector

            d(nu) = D_lin @ nu + D_quad @ (|nu| ⊙ nu)
        """
        nu = np.asarray(nu, dtype=float).reshape(6)
        if self._no_damping:
            return np.zeros(6)
        return self.D_lin @ nu + self.D_quad @ (np.abs(nu) * nu)

    def g_eta(self, eta: np.ndarray) -> np.ndarray:
        """
        Returns the hydrostatic restoring vector for a submerged vehicle in NED.

        The closed-form expression assumes:
        - weight acts at CG
        - buoyancy acts at CB
        - both are resolved in BODY frame
        """
        eta = np.asarray(eta, dtype=float).reshape(6)
        phi = eta[3]
        theta = eta[4]

        xG, yG, zG = self.r_G
        xB, yB, zB = self.r_B

        sphi = np.sin(phi)
        cphi = np.cos(phi)
        sth = np.sin(theta)
        cth = np.cos(theta)

        g = np.array([
            (self.W - self.B) * sth,
            -(self.W - self.B) * cth * sphi,
            -(self.W - self.B) * cth * cphi,
            -(yG * self.W - yB * self.B) * cth * cphi + (zG * self.W - zB * self.B) * cth * sphi,
            (zG * self.W - zB * self.B) * sth + (xG * self.W - xB * self.B) * cth * cphi,
            -(xG * self.W - xB * self.B) * cth * sphi - (yG * self.W - yB * self.B) * sth
        ])

        return g

    def rhs(self, t: float, x: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        Returns the continuous-time state derivative

            x_dot = [eta_dot; nu_dot]
        """
        x = np.asarray(x, dtype=float).reshape(12)
        tau = np.asarray(tau, dtype=float).reshape(6)

        eta = x[0:6]
        nu = x[6:12]

        eta_dot = J_eta(eta) @ nu

        C = self.C_total(nu)
        d = self.d_nu(nu)
        g = self.g_eta(eta)

        nu_dot = np.linalg.solve(self.M, tau - C @ nu - d - g)

        return np.concatenate((eta_dot, nu_dot))

    def rk4_step(self, t: float, x: np.ndarray, tau: np.ndarray, dt: float) -> np.ndarray:
        """
        Returns one RK4 integration step with piecewise-constant input tau.
        """
        k1 = self.rhs(t, x, tau)
        k2 = self.rhs(t + 0.5 * dt, x + 0.5 * dt * k1, tau)
        k3 = self.rhs(t + 0.5 * dt, x + 0.5 * dt * k2, tau)
        k4 = self.rhs(t + dt, x + dt * k3, tau)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


if __name__ == "__main__":
    """
    Proof of life demo
    """
    m = 11.0
    g0 = 9.81
    W = m * g0
    B = W

    r_G = np.array([0.0, 0.0, 0.0])
    r_B = np.array([0.0, 0.0, 0.0])

    I_G = np.diag([0.16, 0.16, 0.25])

    MA = diagonal_added_mass_heuristic(
        mass=m,
        Ix=I_G[0, 0],
        Iy=I_G[1, 1],
        Iz=I_G[2, 2],
        surge_ratio=0.10,
        sway_ratio=0.60,
        heave_ratio=0.60,
        roll_ratio=0.05,
        pitch_ratio=0.25,
        yaw_ratio=0.25,
    )

    D_lin = np.diag([4.0, 6.0, 6.0, 0.2, 0.3, 0.3])
    D_quad = np.diag([18.0, 25.0, 25.0, 0.5, 0.8, 0.8])

    model = Fossen6DOFParams(
        m=m,
        r_G=r_G,
        I_G=I_G,
        W=W,
        B=B,
        r_B=r_B,
        MA=MA,
        D_lin=D_lin,
        D_quad=D_quad,
    )

    x0 = np.ones(12)
    tau = np.ones(6)

    xdot = model.rhs(0.0, x0, tau)

    print("MRB =")
    print(model.MRB)
    print("\nMA =")
    print(model.MA)
    print("\nM =")
    print(model.M)
    print("\ng(eta=0) =")
    print(model.g_eta(np.zeros(6)))
    print("\nxdot at trim =")
    print(xdot)
    print(tau)
