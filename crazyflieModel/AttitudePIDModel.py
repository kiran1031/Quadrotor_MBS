from typing import Dict, Any

from ModelCreator_v1 import ModelCreator
from numpy import ndarray, array
from utils import get_transfer_matrix_zyx


# ************************************************************************* #
#                   Class for attitude reference                            #
# ************************************************************************* #
class AttitudePIDModel(ModelCreator):

    def __init__(self):

        super().__init__(
            name="attitude_pid_model",
            input_schema={
                "Kp_att": ndarray,  # kp for attitude controller
                "Kd_att": ndarray,  # kd for attitude controller
                "ref_phi": float,  # reference roll angle
                "ref_theta": float,  # reference pitch angle
                "ref_psi": float,  # reference yaw angle
                "inertia_tensor": ndarray,  # inertia tensor
                "state": ndarray,  # current state of the quadcopter
            },
            output_schema={
                "L": float,  # roll moment
                "M": float,  # pitch moment
                "N": float,  # yaw moment
            },
            parameters={},
        )

    def _run_process(self, inputs: Dict[str, Any]):
        omega = self.inputs["state"][3:6]
        euler  = self.inputs["state"][6:9]
        pqr_des = array([
            0.,
            0.,
            0.,
        ])
        euler_des = array([
            self.inputs["ref_phi"],
            self.inputs["ref_theta"],
            self.inputs["ref_psi"],
        ])

        m_ = self.inputs["inertia_tensor"] @ \
            (self.inputs["Kd_att"] * (pqr_des - omega) + self.inputs["Kp_att"] * (euler_des - euler))

        self.outputs["L"] = m_[0]
        self.outputs["M"] = m_[1]
        self.outputs["N"] = m_[2]


attitude_pid_model = AttitudePIDModel()
