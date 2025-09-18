"""

Unicycle model class

author Atsushi Sakai

"""

import math
import numpy as np
# from utils.angle import angle_mod

def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)

    Parameters
    ----------
    x : float or array_like
        A angle or an array of angles. This array is flattened for
        the calculation. When an angle is provided, a float angle is returned.
    zero_2_2pi : bool, optional
        Change angle modulo range to [0, 2pi)
        Default is False.
    degree : bool, optional
        If True, then the given angles are assumed to be in degrees.
        Default is False.

    Returns
    -------
    ret : float or ndarray
        an angle or an array of modulated angle.

    Examples
    --------
    >>> angle_mod(-4.0)
    2.28318531

    >>> angle_mod([-4.0])
    np.array(2.28318531)

    >>> angle_mod([-150.0, 190.0, 350], degree=True)
    array([-150., -170.,  -10.])

    >>> angle_mod(-60.0, zero_2_2pi=True, degree=True)
    array([300.])

    """
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle


dt = 0.05  # [s]
L = 0.9  # [m]
steer_max = np.deg2rad(40.0)
curvature_max = math.tan(steer_max) / L
curvature_max = 1.0 / curvature_max + 1.0

accel_max = 5.0


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


def update(state, a, delta):

    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    state.yaw = pi_2_pi(state.yaw)
    state.v = state.v + a * dt

    return state


def pi_2_pi(angle):
    return angle_mod(angle)


if __name__ == '__main__':  # pragma: no cover
    print("start unicycle simulation")
    import matplotlib.pyplot as plt

    T = 100
    a = [1.0] * T
    delta = [np.deg2rad(1.0)] * T
    #  print(delta)
    #  print(a, delta)

    state = State()

    x = []
    y = []
    yaw = []
    v = []

    for (ai, di) in zip(a, delta):
        state = update(state, ai, di)

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)

    plt.subplots(1)
    plt.plot(x, y)
    plt.axis("equal")
    plt.grid(True)

    plt.subplots(1)
    plt.plot(v)
    plt.grid(True)

    plt.show()
