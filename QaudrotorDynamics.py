import numpy as np

class QuadrotorDynamics:

    def __init__(self, dt):

        """
        Flat-space jerk model for quadrotor MPC.

        State x (11x1):
            0: px       (position x)
            1: py       (position y)
            2: pz       (position z)
            3: yaw      (heading)
            4: vx       (velocity x)
            5: vy       (velocity y)
            6: vz       (velocity z)
            7: yaw_rate (yaw dot)
            8: ax       (acceleration x)
            9: ay       (acceleration y)
            10: az      (acceleration z)

        Input u (4x1):
            0: jx          (jerk x)
            1: jy          (jerk y)
            2: jz          (jerk z)
            3: yaw_ddot    (yaw acceleration)
        """
        self.A_c = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # px_dot = vx
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # py_dot = vy
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # pz_dot = vz
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # yaw_dot = yaw_rate
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # vx_dot = ax
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # vy_dot = ay
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # vz_dot = az
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # yaw_rate_dot = yaw_ddot (input only)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ax_dot = jx (input only)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ay_dot = jy (input only)
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # az_dot = jz (input only)
        ])

        self.B_c = np.array([
            [0, 0, 0, 0],  # px
            [0, 0, 0, 0],  # py
            [0, 0, 0, 0],  # pz
            [0, 0, 0, 0],  # yaw
            [0, 0, 0, 0],  # vx
            [0, 0, 0, 0],  # vy
            [0, 0, 0, 0],  # vz
            [0, 0, 0, 1],  # yaw_rate
            [1, 0, 0, 0],  # ax
            [0, 1, 0, 0],  # ay
            [0, 0, 1, 0]   # az
        ])

        self.C_c = np.eye(11) # output all states
        self.D_c = np.zeros((11, 4)) # no direct feedthrough

        # Euler discretization
        self.A = np.eye(11) + dt * self.A_c
        self.B = dt * self.B_c
        self.C = self.C_c
        self.D = self.D_c

        # Limits on state and input
        big = 1e6

        # positions, yaw, velocities, yaw_rate, accelerations
        self.x_min = np.array([
            -big, -big, 0.0, -np.pi,
            -10., -10., -10., -4.,
            -6.,  -6.,  -6.
        ])
        self.x_max = np.array([
             big,  big,  big,  np.pi,
             10.,  10.,  10.,  4.,
             6.,   6.,   6.
        ])

        # jerk and yaw_ddot limits
        self.u_min = np.array([-10., -10., -10., -20.])
        self.u_max = np.array([ 10.,  10.,  10.,  20.])

    def next_x(self, x, u):
        return self.A.dot(x) + self.B.dot(u)

def dummy_control(quadrotor, x_init, x_target):
    """
    Dummy control that returns constant input.
    """

    u = np.array([1, 1, 1, 1])
    return dummy_control, quadrotor.next_x(x_init, dummy_control), x_init, None

