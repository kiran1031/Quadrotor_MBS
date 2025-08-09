# main script to run the simulation for crazyflie model. See MatlabWorkspace for reference

from InputModel import input_model
from WaypointGeneratorModel import waypoint_generator_model
from PositionPIDModel import position_pid_model
from AttitudePIDModel import attitude_pid_model
from QuadPlantModel import quad_plant_model
from RK4IntegratorModel import rk4_integration_model

import numpy as np
import matplotlib.pyplot as plt

plt.ion()  # Turn on interactive mode


###################################
#           Drone class           #
###################################
class Drone(object):

    """
    This Drone class can be used for autonomous flying by specifying a trajectory or
    can be controlled manually using keyboard keys

    Drone class can be used the following ways
    drone1 = Drone() --> initializes a drone object

    The following keyboard options can be used to control the drone

    """

    def __init__(self, name=""):
        self.name = name  # assign a name to the drone
        self.time_iterable = None
        self.freq_ratio = None
        self.iteration_limit = 1000
        self.time_array = []  # empty list as time gets appended as simulation progresses
        self.state_array = []  # 2d list where each sublist is state vector at current time
        self.converged = False  # parameter that indicates a given maneuver is converged or not to start next maneuver
        self.exit_simulation = False
        self.run_simulation = False
        self.thrust_factor_along_z = 1.3  # Thrust will be set when user presses up aero
        self.rollMoment = 0.1
        self.pitchMoment = 0.1

        ############################
        #   FIGURE FOR ANIMATION   #
        ############################
        self.fig, self.axes = plt.subplots(3, 1, figsize=(8, 10), sharex="all")
        self.fig.canvas.mpl_connect('key_press_event', self.main)
        self.x_ref, self.y_ref, self.z_ref = [], [], []
        self.x_pos, self.y_pos, self.z_pos = [], [], []
        self.line_xref, = self.axes[0].plot([], [], 'b-', label='ref x')
        self.line_xpos, = self.axes[0].plot([], [], 'r--', label='x')
        self.line_yref, = self.axes[1].plot([], [], 'b-', label='ref y')
        self.line_ypos, = self.axes[1].plot([], [], 'r--', label='y')
        self.line_zref, = self.axes[2].plot([], [], 'b-', label='ref z')
        self.line_zpos, = self.axes[2].plot([], [], 'r--', label='z')
        [self.axes[i].legend() for i in range(len(self.axes))]  # activating the legends
        [self.axes[i].grid(True) for i in range(len(self.axes))]  # activating the legends

        ############################
        #   INITIALIZE THE MODEL   #
        ############################
        self.init_model()

    def init_model(self):

        #####################
        #    INPUT MODEL    #
        #####################
        input_model.connect(
            innerloop_freq=200,
            outerloop_freq=50,
            tf=np.nan,
        )
        input_model.process()
        self.time_iterable = input_model.outputs["time"]
        self.freq_ratio = int(input_model.inputs["innerloop_freq"] / input_model.inputs["outerloop_freq"])

        ########################
        #    WAYPOINT MODEL    #
        ########################
        waypoint_generator_model.connect(
            time=np.array([0.0]),
            option="step",
            kwargs={"x": 0.0, "y": 0.0, "z": 0.0},
        )
        waypoint_generator_model.process()

        ############################
        #    POSITION PID MODEL    #
        ############################
        position_pid_model.connect(
            Kp_pos=input_model["Kp_pos"],
            Kd_pos=input_model["Kd_pos"],
            ref_pos=waypoint_generator_model.outputs["ref_pos"][0, :],
            ref_vel=waypoint_generator_model.outputs["ref_vel"][0, :],
            ref_acc=waypoint_generator_model.outputs["ref_acc"][0, :],
            ref_psi=waypoint_generator_model.outputs["ref_psi"][0],
            mass=input_model["mass"],
            g=input_model["g"],
            state=input_model["state_0"],
        )

        ############################
        #    ATTITUDE PID MODEL    #
        ############################
        attitude_pid_model.connect(
            Kp_att=input_model["Kp_att"],
            Kd_att=input_model["Kd_att"],
            ref_phi=position_pid_model["ref_phi"],
            ref_theta=position_pid_model["ref_theta"],
            ref_psi=waypoint_generator_model.outputs["ref_psi"][0],
            inertia_tensor=input_model["inertia_tensor"],
            state=input_model["state_0"],
        )

        ############################
        #     QUAD PLANT MODEL     #
        ############################
        quad_plant_model.connect(
            mass=input_model["mass"],
            inertia_tensor=input_model["inertia_tensor"],
            max_thrust=input_model["max_thrust"],
            min_thrust=input_model["min_thrust"],
            thrust=position_pid_model["thrust"],
            L=attitude_pid_model["L"],
            M=attitude_pid_model["M"],
            N=attitude_pid_model["N"],
            g=input_model["g"],
            l=input_model["l"],
            state=input_model["state_0"],
        )

        ###########################
        #  RK4 INTEGRATION MODEL  #
        ###########################
        rk4_integration_model.connect(
            dt=input_model["dt"],
            t0=0.0,
            state_0=input_model["state_0"],
            model=[position_pid_model, attitude_pid_model, quad_plant_model],
            # last model should give state_d and other models has state
        )

    def go_to(self, x=None, y=None, z=None):
        """

        Args:
            x: position to go to
            y:
            z:

        Returns:

        """
        self.converged = False
        while not self.converged and not self.exit_simulation:
            t_idx, time = next(self.time_iterable)
            self.time_array.append(time)
            if t_idx == 0:
                self.state_array.append(input_model.outputs["state_0"])
                rk4_integration_model.process()
            else:
                attitude_pid_model.update_inputs(
                    ref_phi=position_pid_model["ref_phi"],
                    ref_theta=position_pid_model["ref_theta"],
                    ref_psi=waypoint_generator_model.outputs["ref_psi"][0],
                )
                quad_plant_model.update_inputs(
                    thrust=position_pid_model["thrust"],
                    L=attitude_pid_model["L"],
                    M=attitude_pid_model["M"],
                    N=attitude_pid_model["N"],
                )
                if t_idx % self.freq_ratio == 0:
                    waypoint_generator_model.update_inputs(
                        time=np.array([time]),
                        kwargs={"x": x, "y": y, "z": z}
                    )
                    waypoint_generator_model.process()
                    position_pid_model.update_inputs(
                        ref_pos=waypoint_generator_model.outputs["ref_pos"][0, :],
                        ref_vel=waypoint_generator_model.outputs["ref_vel"][0, :],
                        ref_acc=waypoint_generator_model.outputs["ref_acc"][0, :],
                        ref_psi=waypoint_generator_model.outputs["ref_psi"][0],
                    )
                    rk4_integration_model.update_inputs(t0=time,
                                                        state_0=rk4_integration_model.outputs["state"],
                                                        model=[position_pid_model, attitude_pid_model,
                                                               quad_plant_model])
                    rk4_integration_model.process()
                else:
                    rk4_integration_model.update_inputs(t0=time,
                                                        state_0=rk4_integration_model.outputs["state"],
                                                        model=[attitude_pid_model, quad_plant_model])
                    rk4_integration_model.process()
                self.state_array.append(rk4_integration_model.outputs["state"])

                if (np.allclose(np.array([x, y, z]), np.array([self.state_array[-1][9],
                                                             self.state_array[-1][10],
                                                             self.state_array[-1][11]]), rtol=1e-3, atol=1e-3)) or (t_idx % self.iteration_limit == 0):
                    self.converged = True
                    print(self.state_array[-1])
                    print(t_idx)
                    print("manuever accomplished")

            # plot the data
            self.x_ref.append(x)
            self.y_ref.append(y)
            self.z_ref.append(z)
            self.x_pos.append(self.state_array[-1][9])
            self.y_pos.append(self.state_array[-1][10])
            self.z_pos.append(self.state_array[-1][11])

            # animating the x position of drone
            self.line_xref.set_data(self.time_array, self.x_ref)
            self.line_xpos.set_data(self.time_array, self.x_pos)
            self.axes[0].relim()
            self.axes[0].autoscale_view()

            # animating the y position of drone
            self.line_yref.set_data(self.time_array, self.y_ref)
            self.line_ypos.set_data(self.time_array, self.y_pos)
            self.axes[1].relim()
            self.axes[1].autoscale_view()

            # animating the z position of drone
            self.line_zref.set_data(self.time_array, self.z_ref)
            self.line_zpos.set_data(self.time_array, self.z_pos)
            self.axes[2].relim()
            self.axes[2].autoscale_view()

            plt.draw()
            plt.grid()
            plt.pause(0.01)

    def move_a_step_in(self, dir=None):
        """
        Method to manually control the drone

        Args:
            dir: x, y or z as a str to move along

        Returns:

        """
        t_idx, time = next(self.time_iterable)
        self.time_array.append(time)

        if t_idx == 0:
            self.state_array.append(input_model.outputs["state_0"])
            rk4_integration_model.process()
        else:
            if dir == "z":
                attitude_pid_model.update_inputs(
                    ref_phi=0.0,
                    ref_theta=0.0,
                    ref_psi=0.0,
                )
                quad_plant_model.update_inputs(
                    thrust=self.thrust_factor_along_z * input_model.outputs["mass"] * input_model.outputs["g"],
                    L=0.0,
                    M=0.0,
                    N=0.0,
                )
            elif dir == "x":
                raise NotADirectoryError("x manual control not yet implemented")
            elif dir == "y":
                raise NotADirectoryError("y manual control not yet implemented")

            rk4_integration_model.update_inputs(t0=time,
                                                state_0=rk4_integration_model.outputs["state"],
                                                model=[attitude_pid_model, quad_plant_model])
            rk4_integration_model.process()

        self.state_array.append(rk4_integration_model.outputs["state"])

        # plot the data
        self.x_ref.append(self.state_array[-1][9])
        self.y_ref.append(self.state_array[-1][10])
        self.z_ref.append(self.state_array[-1][11])
        self.x_pos.append(self.state_array[-1][9])
        self.y_pos.append(self.state_array[-1][10])
        self.z_pos.append(self.state_array[-1][11])

        # animating the x position of drone
        self.line_xref.set_data(self.time_array, self.x_ref)
        self.line_xpos.set_data(self.time_array, self.x_pos)
        self.axes[0].relim()
        self.axes[0].autoscale_view()

        # animating the y position of drone
        self.line_yref.set_data(self.time_array, self.y_ref)
        self.line_ypos.set_data(self.time_array, self.y_pos)
        self.axes[1].relim()
        self.axes[1].autoscale_view()

        # animating the z position of drone
        self.line_zref.set_data(self.time_array, self.z_ref)
        self.line_zpos.set_data(self.time_array, self.z_pos)
        self.axes[2].relim()
        self.axes[2].autoscale_view()

        plt.draw()
        plt.grid()
        plt.pause(0.01)

    def main(self, event):
        """
        callback method for key press event from matplotlib window
        press G to move to a new position given by x,y,z as a user input
        press up arrow to move along z a time step
        press Q to quit the simulation
        Args:
            event: event of a keypress
        Returns: None

        """

        if event.key.lower() == "q":
            print("Exiting the simulation")
            self.exit_simulation = True

        elif event.key.lower() == "g":
            user_input = input("Enter x, y, z to reach: ")
            gxyz = user_input.split(",")
            if len(gxyz) != 3:
                raise Exception("Give x, y, z separated by ,")

            x, y, z = gxyz
            self.go_to(float(x), float(y), float(z))

        elif event.key.lower() == "up":
            self.move_a_step_in(dir="z")


if __name__ == "__main__":
    drone1 = Drone()
    plt.ioff()
    plt.show()
