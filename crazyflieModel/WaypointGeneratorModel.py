from typing import Dict, Any, Union

from ModelCreator_v1 import ModelCreator
from numpy import ndarray, array, where, pi, sin, cos
import matplotlib.pyplot as plt

from crazyflieModel.utils import tj_from_line_vectorized

#from mpl_toolkits.mplot3d import Axes3D


# ************************************************************************* #
#                   Class for reference trajectory                            #
# ************************************************************************* #
class WaypointGeneratorModel(ModelCreator):

    def __init__(self):

        self.initial_point = array([0., 0., 0.])  # predefined straight line trajectory for testing
        self.direction_vector = array([1, 1, 1])  # direction of the straight line
        super().__init__(
            name="waypoint_generator_model",  # model stops at every waypoint, hence velocities are zero at waypoints
            input_schema={
                "time": Union[ndarray, float],  # time vector to generate waypoint for
                "option": str,  # could be circular, straight line etc for testing purposes
                "kwargs": dict,  # additional parameters needed for other models to run
            },
            output_schema={
                "ref_pos": ndarray,  # reference position
                "ref_vel": ndarray,  # reference velocity
                "ref_acc": ndarray,  # reference acceleration
                "ref_psi": Union[ndarray, float],  #  reference yaw angle
            },
            parameters={},
        )

    def _run_process(self, inputs: Dict[str, Any]):

        if self.inputs["option"] == "step":
            self.outputs["ref_pos"] = array([
                where(self.inputs["time"] <= 0, 0, self.inputs["kwargs"]["x"]),
                where(self.inputs["time"] <= 0, 0, self.inputs["kwargs"]["y"]),
                where(self.inputs["time"] <= 0, 0, self.inputs["kwargs"]["z"])]).transpose()

            self.outputs["ref_vel"] = array([
                array([0] * len(self.inputs["time"])),
                array([0] * len(self.inputs["time"])),
                array([0] * len(self.inputs["time"]))]).transpose()

            self.outputs["ref_acc"] = array([
                array([0] * len(self.inputs["time"])),
                array([0] * len(self.inputs["time"])),
                array([0] * len(self.inputs["time"]))]).transpose()

            self.outputs['ref_psi'] = array([0] * len(self.inputs["time"]))

        elif self.inputs["option"] == "constant_velocity":
            self.outputs["ref_pos"] = array([
                array([0] * len(self.inputs["time"])),
                array([0] * len(self.inputs["time"])),
                array([0] * len(self.inputs["time"]))]).transpose()

            self.outputs["ref_vel"] = array([
                array([self.inputs["kwargs"]["ue"]] * len(self.inputs["time"])),
                array([self.inputs["kwargs"]["ve"]] * len(self.inputs["time"])),
                array([self.inputs["kwargs"]["we"]] * len(self.inputs["time"]))]).transpose()

            self.outputs["ref_acc"] = array([
                array([0] * len(self.inputs["time"])),
                array([0] * len(self.inputs["time"])),
                array([0] * len(self.inputs["time"]))]).transpose()

            self.outputs['ref_psi'] = array([0] * len(self.inputs["time"]))

        elif self.inputs["option"] == "straight_line":
            self.outputs["ref_pos"] = array([
                self.initial_point[0] + self.inputs["time"] * self.direction_vector[0],
                self.initial_point[1] + self.inputs["time"] * self.direction_vector[1],
                self.initial_point[2] + self.inputs["time"] * self.direction_vector[2]]).transpose()

            self.outputs["ref_vel"] = array([
                array([self.direction_vector[0]] * len(self.inputs["time"])),
                array([self.direction_vector[1]] * len(self.inputs["time"])),
                array([self.direction_vector[2]] * len(self.inputs["time"]))]).transpose()

            self.outputs["ref_acc"] = array([
                array([0] * len(self.inputs["time"])),
                array([0] * len(self.inputs["time"])),
                array([0] * len(self.inputs["time"]))]).transpose()

            self.outputs['ref_psi'] = array([0] * len(self.inputs["time"]))

        elif self.inputs["option"] == "circle":

            radius = 1
            dt = 0.0001

            angle_t, _, _ = tj_from_line_vectorized(0, 2 * pi, self.inputs["time"][-1], self.inputs["time"])
            angle_t_dt, _, _ = tj_from_line_vectorized(0, 2 * pi, self.inputs["time"][-1], self.inputs["time"] + dt)
            angle_t_2dt, _, _ = tj_from_line_vectorized(0, 2 * pi, self.inputs["time"][-1], self.inputs["time"] + 2 * dt)
            angle_t = angle_t[:, 0]
            angle_t_dt = angle_t_dt[:, 0]
            angle_t_2dt = angle_t_2dt[:, 0]

            self.outputs["ref_pos"] = array([
                radius * cos(angle_t),
                radius * sin(angle_t),
                2.5 * angle_t / (2 * pi)]).transpose()

            self.outputs["ref_vel"] = array([
                (radius * cos(angle_t_dt) - radius * cos(angle_t)) / dt,
                (radius * sin(angle_t_dt) - radius * sin(angle_t)) / dt,
                2.5 * (angle_t_dt - angle_t) / (2 * pi * dt)]).transpose()

            self.outputs["ref_acc"] = array([
                (radius * cos(angle_t_2dt) - 2 * radius * cos(angle_t_dt) + radius * cos(angle_t)) / dt ** 2,
                (radius * sin(angle_t_2dt) - 2 * radius * sin(angle_t_dt) + radius * sin(angle_t)) / dt ** 2,
                2.5 * (angle_t_2dt - 2 * angle_t_dt + angle_t) / (2 * pi * dt ** 2)]).transpose()

            self.outputs['ref_psi'] = array([0] * len(self.inputs["time"]))


waypoint_generator_model = WaypointGeneratorModel()

if __name__ == "__main__":
    from InputModel import input_model
    input_model.connect(
        innerloop_freq=500,
        outerloop_freq=100,
        tf=12,
    )
    input_model.process()
    time_array = input_model.outputs["time"]
    waypoint_generator_model.connect(
        time=time_array,
        option="step",
        kwargs={"x": 3, "y": 2, "z": 1},
    )
    waypoint_generator_model.process()
    fig_pos = plt.figure()
    ax = fig_pos.add_subplot(111, projection='3d')

    # Plot the 3D line
    ax.scatter(waypoint_generator_model.outputs["ref_pos"][:, 0],
            waypoint_generator_model.outputs["ref_pos"][:, 1],
            waypoint_generator_model.outputs["ref_pos"][:, 2], color='red', marker='o', label='Points')

    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('waypoints')
    ax.legend()

    # plot positions in separate plot
    fig_pos1, axs_pos = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex="all")
    axs_pos[0].plot(waypoint_generator_model.inputs["time"], waypoint_generator_model.outputs["ref_pos"][:, 0], label='ref_xe',
                    color='blue')
    axs_pos[0].set_ylabel('ref_xe')
    axs_pos[0].legend()

    axs_pos[1].plot(waypoint_generator_model.inputs["time"], waypoint_generator_model.outputs["ref_pos"][:, 1], label='ref_ye',
                    color='blue')
    axs_pos[1].set_ylabel('ref_ye')
    axs_pos[1].legend()

    axs_pos[2].plot(waypoint_generator_model.inputs["time"], waypoint_generator_model.outputs["ref_pos"][:, 2], label='ref_ze',
                    color='blue')
    axs_pos[2].set_xlabel('time')
    axs_pos[2].set_ylabel('ref_ze')
    axs_pos[2].legend()

    # plot velocities in separate plot
    fig_vel, axs_vel = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex="all")
    axs_vel[0].plot(waypoint_generator_model.inputs["time"], waypoint_generator_model.outputs["ref_vel"][:, 0], label='ref_ue',
                color='blue')
    axs_vel[0].set_ylabel('ref_ue')
    axs_vel[0].legend()

    axs_vel[1].plot(waypoint_generator_model.inputs["time"], waypoint_generator_model.outputs["ref_vel"][:, 1], label='ref_ve',
                color='blue')
    axs_vel[1].set_ylabel('ref_ve')
    axs_vel[1].legend()

    axs_vel[2].plot(waypoint_generator_model.inputs["time"], waypoint_generator_model.outputs["ref_vel"][:, 2], label='ref_we',
                color='blue')
    axs_vel[2].set_xlabel('time')
    axs_vel[2].set_ylabel('ref_we')
    axs_vel[2].legend()

    # Optional: tighten layout
    plt.tight_layout()

    # Show plot
    plt.show()
