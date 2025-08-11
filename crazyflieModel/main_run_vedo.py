# main script to run the simulation for crazyflie model. See MatlabWorkspace for reference

from InputModel import input_model
from WaypointGeneratorModel import waypoint_generator_model
from PositionPIDModel import position_pid_model
from AttitudePIDModel import attitude_pid_model
from QuadPlantModel import quad_plant_model
from RK4IntegratorModel import rk4_integration_model

from vedo import Box, Cylinder, Sphere, Plotter, LinearTransform, Box, vector, Plane
from munch import Munch
import numpy as np


camera1 = dict(
    pos=(0, 40, 0),
    focal_point=(0, 0, 0),
    viewup=(0, 0, 1)
)

camera2 = dict(
    pos=(0, 0, 5),
    focal_point=(0, 0, 0),
    viewup=(0, 1, 0)
)


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

    def __init__(self, name="", params=None):
        self.name = name  # assign a name to the drone
        self.time_iterable = None
        self.freq_ratio = None
        self.iteration_limit = 1000
        self.time_array = []  # empty list as time gets appended as simulation progresses
        self.state_array = []  # 2d list where each sublist is state vector at current time
        self.x_ref, self.y_ref, self.z_ref = [], [], []
        self.x_pos, self.y_pos, self.z_pos = [], [], []
        self.ref_phi, self.ref_theta, self.ref_psi = [], [], []
        self.phi, self.theta, self.psi = [], [], []
        self.converged = False  # parameter that indicates a given maneuver is converged or not to start next maneuver
        self.exit_simulation = False
        self.run_simulation = False
        self.thrust_factor_along_z = 1.3  # Thrust will be set when user presses up aero
        self.rollMoment = 0.1
        self.pitchMoment = 0.1

        ############################
        #   INITIALIZE THE MODEL   #
        ############################
        self.init_model()

        ############################
        #        VEDO MODEL        #
        ############################
        self.arm_ang = None
        self.arm_len = None
        self.pos_sphere = None
        self.prop1 = None
        self.prop2 = None
        self.prop3 = None
        self.prop4 = None
        self.quad_frame = None
        self.prop1_rot_pt = None
        self.prop2_rot_pt = None
        self.prop3_rot_pt = None
        self.prop4_rot_pt = None
        self.plt = None
        self.params = params
        self.ang1, self.ang2, self.ang3, self.ang4 = 0.0, 0.0, 0.0, 0.0
        self.create_additional_parameters()
        self.create_model()

    def create_additional_parameters(self):
        self.arm_len = 2 * np.sqrt(self.params.dx**2 + self.params.dy**2)
        self.arm_ang = np.arctan2(self.params.dx, self.params.dy)

    def create_model(self):
        world = Box(pos=(0, 0, 0), size=(20, 20, 20)).wireframe()
        ground = Plane(pos=(0.0, 0.0, -0.1), # center-ish
                        normal=(0,0,1),
                        s=(20,20),
                        res=(1,1))
        self.pos_sphere = Sphere(pos=(0, 0, 0), r=0.001).color("gray")
        arm1 = Cylinder(
            pos=[(0.5 * self.arm_len * np.cos(self.arm_ang), 0.5 * self.arm_len * np.sin(self.arm_ang), 0.0),
                 (-0.5 * self.arm_len * np.cos(self.arm_ang), -0.5 * self.arm_len * np.sin(self.arm_ang), 0.0)],
            r=0.01*self.arm_len
        )
        arm2 = Cylinder(
            pos=[(-0.5 * self.arm_len * np.sin(self.arm_ang), 0.5 * self.arm_len * np.cos(self.arm_ang), 0.0),
                 (0.5 * self.arm_len * np.sin(self.arm_ang), -0.5 * self.arm_len * np.cos(self.arm_ang), 0.0)],
            r=0.01 * self.arm_len
        )
        motor1 = Cylinder(
            pos=[(0.5 * self.arm_len * np.cos(self.arm_ang), 0.5 * self.arm_len * np.sin(self.arm_ang), 0.0),
                 (0.5 * self.arm_len * np.cos(self.arm_ang), 0.5 * self.arm_len * np.sin(self.arm_ang), 0.1 * self.arm_len)],
            r=0.05 * self.arm_len,
            c="red"
        )
        motor2 = Cylinder(
            pos=[(-0.5 * self.arm_len * np.cos(self.arm_ang), -0.5 * self.arm_len * np.sin(self.arm_ang), 0.0),
                 (-0.5 * self.arm_len * np.cos(self.arm_ang), -0.5 * self.arm_len * np.sin(self.arm_ang), 0.1 * self.arm_len)],
            r=0.05 * self.arm_len,
            c="blue"
        )
        motor3 = Cylinder(
            pos=[(-0.5 * self.arm_len * np.sin(self.arm_ang), 0.5 * self.arm_len * np.cos(self.arm_ang), 0.0),
                 (-0.5 * self.arm_len * np.sin(self.arm_ang), 0.5 * self.arm_len * np.cos(self.arm_ang), 0.1 * self.arm_len)],
            r=0.05 * self.arm_len,
            c="green"
        )
        motor4 = Cylinder(
            pos=[(0.5 * self.arm_len * np.sin(self.arm_ang), -0.5 * self.arm_len * np.cos(self.arm_ang), 0.0),
                 (0.5 * self.arm_len * np.sin(self.arm_ang), -0.5 * self.arm_len * np.cos(self.arm_ang), 0.1 * self.arm_len)],
            r=0.05 * self.arm_len,
            c="yellow"
        )
        self.prop1 = Cylinder(
            pos=[(0.5 * self.arm_len * np.cos(self.arm_ang) - 0.2 * self.arm_len, 0.5 * self.arm_len * np.sin(self.arm_ang), 0.11 * self.arm_len),
                 (0.5 * self.arm_len * np.cos(self.arm_ang) + 0.2 * self.arm_len, 0.5 * self.arm_len * np.sin(self.arm_ang), 0.11 * self.arm_len)],
            r=0.01 * self.arm_len
        )
        self.prop2 = Cylinder(
            pos=[(-0.5 * self.arm_len * np.cos(self.arm_ang) - 0.2 * self.arm_len, -0.5 * self.arm_len * np.sin(self.arm_ang), 0.11 * self.arm_len),
                 (-0.5 * self.arm_len * np.cos(self.arm_ang) + 0.2 * self.arm_len, -0.5 * self.arm_len * np.sin(self.arm_ang), 0.11 * self.arm_len)],
            r=0.01 * self.arm_len
        )
        self.prop3 = Cylinder(
            pos=[(-0.5 * self.arm_len * np.sin(self.arm_ang) - 0.2 * self.arm_len, 0.5 * self.arm_len * np.cos(self.arm_ang), 0.11 * self.arm_len),
                 (-0.5 * self.arm_len * np.sin(self.arm_ang) + 0.2 * self.arm_len, 0.5 * self.arm_len * np.cos(self.arm_ang), 0.11 * self.arm_len)],
            r=0.01 * self.arm_len
        )
        self.prop4 = Cylinder(
            pos=[(0.5 * self.arm_len * np.sin(self.arm_ang) - 0.2 * self.arm_len, -0.5 * self.arm_len * np.cos(self.arm_ang), 0.11 * self.arm_len),
                 (0.5 * self.arm_len * np.sin(self.arm_ang) + 0.2 * self.arm_len, -0.5 * self.arm_len * np.cos(self.arm_ang), 0.11 * self.arm_len)],
            r=0.01 * self.arm_len
        )
        self.quad_frame = arm1 + arm2 + motor1 + motor2 + motor3 + motor4

        self.pos_sphere.add_trail(n=10000)

        self.plt = Plotter(title="Quad Model", axes=1, interactive=True)
        self.plt += world
        self.plt += ground
        self.plt += self.pos_sphere
        self.plt += self.quad_frame
        self.plt += self.prop1
        self.plt += self.prop2
        self.plt += self.prop3
        self.plt += self.prop4

        # switch off the existing key press events of vedo
        self.plt.remove_callback("KeyPress")

        # add custom callback function
        self.plt.add_callback("KeyPress", self.main)

        self.plt.show(camera=camera1)

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
                # this is to reinitialize after calling manual control as it overrides the inputs
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
                        option="step",
                        time=np.array([time]),
                        kwargs={"x": x, "y": y, "z": z}
                    )
                    waypoint_generator_model.process()
                    position_pid_model.update_inputs(
                        Kp_pos=input_model["Kp_pos"],
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
            self.ref_phi.append(position_pid_model.outputs["ref_phi"])
            self.ref_theta.append(position_pid_model.outputs["ref_theta"])
            self.ref_psi.append(waypoint_generator_model.outputs["ref_psi"][0])
            self.x_pos.append(self.state_array[-1][9])
            self.y_pos.append(self.state_array[-1][10])
            self.z_pos.append(self.state_array[-1][11])
            self.phi.append(self.state_array[-1][6])
            self.theta.append(self.state_array[-1][7])
            self.psi.append(self.state_array[-1][8])

    def move_a_step_in(self, dir=None, sign=1):
        """
        Method to manually control the drone

        Args:
            dir: x, y or z as a str to move along with constant velocity
            sign: to move either in positive or negative direction

        Returns:

        """
        t_idx, time = next(self.time_iterable)
        self.time_array.append(time)
        initial_z = input_model.outputs["state_0"][11]  # need to maintain this z when moving along x and y

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
                rk4_integration_model.update_inputs(t0=time,
                                                    state_0=rk4_integration_model.outputs["state"],
                                                    model=[attitude_pid_model, quad_plant_model])
            elif dir == "x":
                waypoint_generator_model.update_inputs(
                    time=np.array([time]),
                    option="constant_velocity",
                    kwargs={"ue": sign * 1.0, "ve": 0.0, "we": 0.0}
                )
                waypoint_generator_model.process()
                position_pid_model.update_inputs(
                    Kp_pos=np.array([0 ,0, 30.0]),
                    ref_pos=np.array([0.0, 0.0, initial_z]),
                    ref_vel=waypoint_generator_model.outputs["ref_vel"][0, :],
                    ref_acc=waypoint_generator_model.outputs["ref_acc"][0, :],
                    ref_psi=waypoint_generator_model.outputs["ref_psi"][0],
                )
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
                rk4_integration_model.update_inputs(t0=time,
                                                    state_0=rk4_integration_model.outputs["state"],
                                                    model=[position_pid_model, attitude_pid_model, quad_plant_model])
            elif dir == "y":
                waypoint_generator_model.update_inputs(
                    time=np.array([time]),
                    option="constant_velocity",
                    kwargs={"ue": 0.0, "ve": sign * 1.0, "we": 0.0}
                )
                waypoint_generator_model.process()
                position_pid_model.update_inputs(
                    Kp_pos=np.array([0, 0, 30.0]),
                    ref_pos=np.array([0.0, 0.0, initial_z]),
                    ref_vel=waypoint_generator_model.outputs["ref_vel"][0, :],
                    ref_acc=waypoint_generator_model.outputs["ref_acc"][0, :],
                    ref_psi=waypoint_generator_model.outputs["ref_psi"][0],
                )
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
                rk4_integration_model.update_inputs(t0=time,
                                                    state_0=rk4_integration_model.outputs["state"],
                                                    model=[position_pid_model, attitude_pid_model, quad_plant_model])

            rk4_integration_model.process()

        self.state_array.append(rk4_integration_model.outputs["state"])

        # append the data
        self.x_ref.append(self.state_array[-1][9])
        self.y_ref.append(self.state_array[-1][10])
        self.z_ref.append(self.state_array[-1][11])
        self.ref_phi.append(self.state_array[-1][6])
        self.ref_theta.append(self.state_array[-1][7])
        self.ref_psi.append(self.state_array[-1][8])
        self.x_pos.append(self.state_array[-1][9])
        self.y_pos.append(self.state_array[-1][10])
        self.z_pos.append(self.state_array[-1][11])
        self.phi.append(self.state_array[-1][6])
        self.theta.append(self.state_array[-1][7])
        self.psi.append(self.state_array[-1][8])

        # animate the data in vedo
        self.animate_time_step(
            self.state_array[-1][9],
            self.state_array[-1][10],
            self.state_array[-1][11],
            self.state_array[-1][6],
            self.state_array[-1][7],
            self.state_array[-1][8],
            input_model.outputs["dt"],
        )

    def animate_time_step(self, xpos, ypos, zpos, roll, pitch, yaw, dt):
        """
        """

        LTrot = LinearTransform()
        Ltrans = LinearTransform()
        LT1 = LinearTransform()
        LT2 = LinearTransform()
        LT3 = LinearTransform()
        LT4 = LinearTransform()

        self.ang1 = self.ang1 + 6 * 1000 * dt
        self.ang2 = self.ang2 + 6 * 1000 * dt
        self.ang3 = self.ang3 + 6 * 1000 * dt
        self.ang4 = self.ang4 + 6 * 1000 * dt

        yaw = yaw * 180 / np.pi
        pitch = pitch * 180 / np.pi
        roll = roll * 180 / np.pi

        self.pos_sphere.pos(xpos, ypos, zpos)

        Ltrans.translate([xpos, ypos, zpos])
        Ltrans.move(self.quad_frame)
        LTrot.rotate(yaw, axis=(0, 0, 1), point=(xpos, ypos, zpos), rad=False)
        LTrot.rotate(roll, axis=(1, 0, 0), point=(xpos, ypos, zpos), rad=False)
        LTrot.rotate(pitch, axis=(0, 1, 0), point=(xpos, ypos, zpos), rad=False)
        LTrot.move(self.quad_frame)

        new_prop1_pos = vector(xpos, ypos, zpos) + (
                vector(LTrot.matrix3x3 @ self.prop1.base) + vector(LTrot.matrix3x3 @ self.prop1.top)) / 2.0
        new_prop2_pos = vector(xpos, ypos, zpos) + (
                vector(LTrot.matrix3x3 @ self.prop2.base) + vector(LTrot.matrix3x3 @ self.prop2.top)) / 2.0
        new_prop3_pos = vector(xpos, ypos, zpos) + (
                vector(LTrot.matrix3x3 @ self.prop3.base) + vector(LTrot.matrix3x3 @ self.prop3.top)) / 2.0
        new_prop4_pos = vector(xpos, ypos, zpos) + (
                vector(LTrot.matrix3x3 @ self.prop4.base) + vector(LTrot.matrix3x3 @ self.prop4.top)) / 2.0

        new_prop_rot_axis = LTrot.matrix3x3 @ vector(0, 0, 1)

        self.prop1.pos(new_prop1_pos[0], new_prop1_pos[1], new_prop1_pos[2])
        self.prop2.pos(new_prop2_pos[0], new_prop2_pos[1], new_prop2_pos[2])
        self.prop3.pos(new_prop3_pos[0], new_prop3_pos[1], new_prop3_pos[2])
        self.prop4.pos(new_prop4_pos[0], new_prop4_pos[1], new_prop4_pos[2])

        LT1.rotate(self.ang1, axis=new_prop_rot_axis,
                   point=(new_prop1_pos[0], new_prop1_pos[1], new_prop1_pos[2]))
        self.prop1.apply_transform(LT1)

        LT2.rotate(self.ang2, axis=new_prop_rot_axis,
                   point=(new_prop2_pos[0], new_prop2_pos[1], new_prop2_pos[2]))
        self.prop2.apply_transform(LT2)

        LT3.rotate(self.ang3, axis=new_prop_rot_axis,
                   point=(new_prop3_pos[0], new_prop3_pos[1], new_prop3_pos[2]))
        self.prop3.apply_transform(LT3)

        LT4.rotate(self.ang4, axis=new_prop_rot_axis,
                   point=(new_prop4_pos[0], new_prop4_pos[1], new_prop4_pos[2]))
        self.prop4.apply_transform(LT4)

        # self.plt.camera.SetPosition(xpos, ypos + 10, zpos)
        # self.plt.camera.SetFocalPoint(xpos, ypos, zpos)

        self.pos_sphere.update_trail()

        self.plt.render()

        Ltrans.reset()
        LTrot.reset()
        LT1.reset()
        LT2.reset()
        LT3.reset()
        LT4.reset()
        self.quad_frame.transform.reset()

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

        if event.keypress.lower() == "q":
            print("Exiting the simulation")
            self.exit_simulation = True

        elif event.keypress.lower() == "g":
            user_input = input("Enter x, y, z to reach: ")
            gxyz = user_input.split(",")
            if len(gxyz) != 3:
                raise Exception("Give x, y, z separated by ,")

            x, y, z = gxyz
            self.go_to(float(x), float(y), float(z))

        elif event.keypress.lower() == "up":
            self.move_a_step_in(dir="z")

        elif event.keypress.lower() == "down":
            self.move_a_step_in(dir="z", sign=-1)

        elif event.keypress.lower() == "w":
            self.move_a_step_in(dir="y")

        elif event.keypress.lower() == "s":
            self.move_a_step_in(dir="y", sign=-1)

        elif event.keypress.lower() == "d":
            self.move_a_step_in(dir="x")

        elif event.keypress.lower() == "a":
            self.move_a_step_in(dir="x", sign=-1)


if __name__ == "__main__":
    params = Munch({
        'dx': 0.046,
        'dy': 0.0,
    })
    drone1 = Drone(params=params)
