import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from PlanningAviary import PlanningAviary
from QaudrotorDynamics import QuadrotorDynamics
from MPCControl import MPCControl
import logging

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1                  # I've simplified to a single drone for our purposes, changing this does not work anymore.
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_USE_MPC = True               # Whether to use MPC planning + PID tracking or just PID tracking
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 100
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_TRAJECTORY_PERIOD = 10
DEFAULT_WAYPOINT_STRIDE = 4
DEFAULT_MPC_LOOKAHEAD_STEPS = 25


def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        use_mpc=DEFAULT_USE_MPC,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
    colab=DEFAULT_COLAB,
    trajectory_period=DEFAULT_TRAJECTORY_PERIOD,
    waypoint_stride=DEFAULT_WAYPOINT_STRIDE,
    mpc_lookahead_steps=DEFAULT_MPC_LOOKAHEAD_STEPS
        ):
    #### Initialize the simulation #############################
    INIT_XYZ = np.array([0, 0, 0.25])
    INIT_RPY = np.array([0, 0, 0])

    #### Initialize a straight line trajectory ####################
    PERIOD = trajectory_period
    NUM_WP = int(control_freq_hz*PERIOD)
    TARGET_POS = np.zeros((NUM_WP,3))

    END_POS = np.array([6, 6, 0.25])
    for i in range(NUM_WP):
        TARGET_POS[i, :] = (END_POS - INIT_XYZ) * (i / NUM_WP) + INIT_XYZ
    wp_counter = 0

    #### Create the environment ################################
    env = PlanningAviary(drone_model=drone,
                        num_drones=1,
                        initial_xyzs=INIT_XYZ.reshape(1, 3),
                        initial_rpys=INIT_RPY.reshape(1, 3),
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=1,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the MPC planner and/or PID controllers #########
    if use_mpc:
        mpc_planner = MPCControl(dt=1/control_freq_hz, lookahead_steps=mpc_lookahead_steps)
    else:
        mpc_planner = None
    
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = DSLPIDControl(drone_model=drone)

    #### Run the simulation ####################################
    action = np.zeros((1, 4))
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control #######################################
        if use_mpc:
            segment_indices = (wp_counter + waypoint_stride * np.arange(mpc_planner.N + 1)) % NUM_WP
            reference_segment = TARGET_POS[segment_indices]
            mpc_planner.set_reference_segment(reference_segment)
            target_pos_mpc, target_yaw_mpc = mpc_planner.compute_target(
                cur_pos=obs[0][0:3],
                cur_quat=obs[0][3:7],
                cur_vel=obs[0][10:13],
                cur_ang_vel=obs[0][13:16]
            )
            
            # Feed MPC targets to PID controller
            action[0, :], _, _ = ctrl.computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[0],
                target_pos=target_pos_mpc,
                target_rpy=np.array([0, 0, target_yaw_mpc])
            )
        else:
            # Use waypoint targets directly
            action[0, :], _, _ = ctrl.computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[0],
                target_pos=np.hstack([TARGET_POS[wp_counter, 0:2], INIT_XYZ[2]]),
                target_rpy=INIT_RPY
            )

        #### Go to the next way point and loop #####################
        wp_counter = (wp_counter + waypoint_stride) % NUM_WP

        #### Log the simulation ####################################
        if use_mpc:
            # Log MPC target position and yaw
            logger.log(drone=0,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[0],
                       control=np.hstack([target_pos_mpc, np.array([0, 0, target_yaw_mpc]), np.zeros(6)])
                       )
        else:
            logger.log(drone=0,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[0],
                       control=np.hstack([TARGET_POS[wp_counter, 0:2], INIT_XYZ[2], INIT_RPY, np.zeros(6)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Quadrotor MPC simulation with PID tracking control')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--use_mpc',            default=DEFAULT_USE_MPC,       type=str2bool,      help='Whether to use MPC control (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--trajectory_period',  default=DEFAULT_TRAJECTORY_PERIOD, type=float,        help='Time in seconds to traverse the global trajectory once (default: 10)', metavar='')
    parser.add_argument('--waypoint_stride',    default=DEFAULT_WAYPOINT_STRIDE, type=int,           help='Number of waypoints to advance per control step (default: 4)', metavar='')
    parser.add_argument('--mpc_lookahead_steps', default=DEFAULT_MPC_LOOKAHEAD_STEPS, type=int,       help='How many MPC prediction steps ahead to track (default: 25)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
