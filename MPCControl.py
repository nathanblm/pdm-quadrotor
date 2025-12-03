import numpy as np
import pybullet as p
import cvxpy as cp


from QaudrotorDynamics import QuadrotorDynamics

class MPCControl:
    """MPC planner class for quadrotors.

    Currently implements a dummy controller with constant inputs [1, 1, 1, 1] for testing.

    """

    def __init__(self, dt, lookahead_steps=10):
        self.dt = dt
        self.quadrotor = QuadrotorDynamics(dt=dt)
        
        self.quad_state = np.zeros(11)
        
        # MPC parameters
        self.N = 30  # Prediction horizon (steps)
        self.reference_segment = None  # (N+1, 3) array of xyz positions
        
        self.control_counter = 0
        self.initialized = False
        self.last_vel = np.zeros(3)
        self.last_control = np.zeros(4)
        self.lookahead_steps = lookahead_steps  # number of MPC steps to look ahead for the tracking target

    def reset(self):
        """Resets the control class."""
        self.quad_state = np.zeros(11)
        self.control_counter = 0
        self.initialized = False
        self.last_vel = np.zeros(3)
        self.last_control = np.zeros(4)
    
    def set_reference_segment(self, reference_xyz):
        """Provide a window of the global trajectory for the MPC horizon.

        Parameters
        ----------
        reference_xyz : ndarray
            Array of shape (N+1, 3) containing xyz samples along the global path.
        """
        if reference_xyz.shape[0] != self.N + 1:
            raise ValueError(
                f"Reference segment must have {self.N + 1} points, got {reference_xyz.shape[0]}"
            )
        self.reference_segment = reference_xyz.copy()

    def _build_reference_states(self):
        if self.reference_segment is None:
            raise ValueError("Reference segment not set. Call set_reference_segment before running MPC.")

        ref_states = np.zeros((11, self.N + 1))

        positions = self.reference_segment
        ref_states[0:3, :] = positions.T

        # Compute velocities using forward differences, keep last valid sample
        velocities = np.zeros_like(positions)
        velocities[:-1] = (positions[1:] - positions[:-1]) / self.dt
        velocities[-1] = velocities[-2]
        ref_states[4:7, :] = velocities.T

        # Estimate accelerations similarly
        accelerations = np.zeros_like(positions)
        accelerations[:-1] = (velocities[1:] - velocities[:-1]) / self.dt
        accelerations[-1] = accelerations[-2]
        ref_states[8:11, :] = accelerations.T

        # Yaw is aligned with planar velocity direction
        yaw = np.arctan2(ref_states[5, :], ref_states[4, :])
        yaw[np.isnan(yaw)] = 0.0
        ref_states[3, :] = yaw

        yaw_rate = np.zeros(self.N + 1)
        yaw_rate[:-1] = (yaw[1:] - yaw[:-1]) / self.dt
        yaw_rate[-1] = yaw_rate[-2]
        ref_states[7, :] = yaw_rate

        return ref_states

    def mpc_control(self, quadrotor, N, x_init, x_reference):

        # Reduce input penalty to allow more aggressive control
        weight_input = 0.01*np.eye(4)
        # Focus weights on position and velocity tracking, less on accelerations
        weight_state = np.diag([100., 100., 100., 10., 10., 10., 10., 5., 0.1, 0.1, 0.1])
        # Higher terminal cost to prioritize reaching target
        weight_terminal = np.diag([200., 200., 200., 20., 20., 20., 20., 10., 0.1, 0.1, 0.1])

        x = cp.Variable((11, N+1))
        u = cp.Variable((4, N))
        
        cost = 0
        constraints = []

        for k in range(N):
            cost += cp.quad_form(x[:,k] - x_reference[:,k], weight_state) + cp.quad_form(u[:,k], weight_input)
            constraints += [x[:,k+1] == quadrotor.A @ (x[:,k]) + quadrotor.B @ (u[:,k])]
            constraints += [quadrotor.x_min <= x[:,k], x[:,k] <= quadrotor.x_max]
            constraints += [quadrotor.u_min <= u[:,k], u[:,k] <= quadrotor.u_max]
        
        # Add terminal cost
        cost += cp.quad_form(x[:,N] - x_reference[:,N], weight_terminal)
        
        constraints += [x[:,0] == x_init]

        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            problem.solve(solver=cp.OSQP, verbose=False)
        except cp.error.SolverError as exc:
            print(f"[WARN] MPC solver error: {exc}")
            return None, None

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"[WARN] MPC problem status {problem.status}, skipping control step")
            return None, None

        if u[:, 0].value is None or x.value is None:
            print("[WARN] MPC solver returned empty solution")
            return None, None

        return u[:,0].value, x.value

    def compute_target(self, cur_pos, cur_quat, cur_vel, cur_ang_vel):
        """Computes target position and yaw using MPC.

        Parameters
        ----------
        cur_pos : ndarray
            (3,)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,)-shaped array of floats containing the current angular velocity.

        Returns
        -------
        ndarray
            (3,)-shaped array containing target position [x, y, z].
        float
            Target yaw angle.

        """
        self.control_counter += 1
        
        # Convert quaternion to Euler angles
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        
        # Build current state vector
        x_current = np.zeros(11)
        x_current[0:3] = cur_pos
        x_current[3] = cur_rpy[2]  # yaw
        x_current[4:7] = cur_vel
        x_current[7] = cur_ang_vel[2]  # yaw_rate
        
        # Estimate accelerations if initialized
        if self.initialized:
            x_current[8:11] = (cur_vel - self.last_vel) / self.dt
        else:
            x_current[8:11] = np.zeros(3)
        
        # Build reference trajectory for the horizon
        x_reference = self._build_reference_states()

        # Solve MPC problem
        mpc_input, predicted_states = self.mpc_control(
            self.quadrotor, 
            self.N, 
            x_current, 
            x_reference
        )

        if predicted_states is None:
            print(f"[WARN] MPC infeasible at step {self.control_counter}, falling back to reference")
            fallback_state = x_reference[:, min(self.lookahead_steps, self.N)]
            target_pos = fallback_state[0:3]
            target_yaw = fallback_state[3]
            self.last_vel = cur_vel.copy()
            if mpc_input is not None:
                self.last_control = mpc_input
            self.initialized = False
            return target_pos, target_yaw

        lookahead_idx = min(self.lookahead_steps, self.N)
        next_state = predicted_states[:, lookahead_idx]

        print(f"MPC Input at step {self.control_counter}: {mpc_input}")
        print(f"Predicted next state: {next_state}")
        
        # Extract target position and yaw from MPC solution
        target_pos = next_state[0:3]  # [px, py, pz]
        target_yaw = next_state[3]    # yaw
        
        # Update state for next iteration
        self.last_vel = cur_vel.copy()
        self.initialized = True
        self.last_control = mpc_input
        
        return target_pos, target_yaw
