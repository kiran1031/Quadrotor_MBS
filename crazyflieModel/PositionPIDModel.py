from typing import Dict, Any

from ModelCreator_v1 import ModelCreator
from numpy import ndarray, array, sin, cos
from utils import get_rotation_matrix_zyx


# ************************************************************************* #
#                   Class for attitude reference                            #
# ************************************************************************* #
class PositionPIDModel(ModelCreator):

    def __init__(self):

        super().__init__(
            name="position_pid_model",
            input_schema={
                "Kp_pos": ndarray,  # kp for position controller
                "Kd_pos": ndarray,  # kd for position controller
                "ref_pos": ndarray,  # reference position to reach in inertial frame
                "ref_vel": ndarray,  # reference velocity to reach in inertial frame
                "ref_acc": ndarray,  # reference acceleration to reach in inertial frame
                "ref_psi": float,  # reference yaw angle
                "mass": float,  # mass of the quadcopter
                "g": float,  # acceleration due to gravity
                "state": ndarray,  # current state  of the quadcopter
            },
            output_schema={
                "thrust": float,  # ref thrust
                "ref_phi": float,  # ref  roll angle
                "ref_theta": float,  # ref pitch angle
            },
            parameters={},
        )

    def _run_process(self, inputs: Dict[str, Any]):
        pos = self.inputs["state"][9:12]
        vel = self.inputs["state"][0:3]

        acc_des = self.inputs["ref_acc"] + \
            self.inputs["Kp_pos"] * (self.inputs["ref_pos"] - pos) + \
            self.inputs["Kd_pos"] * (self.inputs["ref_vel"] - vel)

        self.outputs["ref_phi"] = 1/self.inputs["g"] * \
            (acc_des[0] * sin(self.inputs["ref_psi"]) - acc_des[1] * cos(self.inputs["ref_psi"]))

        self.outputs["ref_theta"] = 1 / self.inputs["g"] * \
            (acc_des[0] * cos(self.inputs["ref_psi"]) + acc_des[1] * sin(self.inputs["ref_psi"]))

        self.outputs["thrust"] = self.inputs["mass"] * (self.inputs["g"] + acc_des[2])


position_pid_model = PositionPIDModel()
