from typing import Dict, Any

import numpy as np

from ModelCreator_v1 import ModelCreator
import numpy as np
from numpy.linalg import inv
from utils import get_rotation_matrix_zxy, get_transfer_matrix_zxy


# ************************************************************************* #
#                   Class for Quadcopter plant model                        #
# ************************************************************************* #
class QuadPlantModel(ModelCreator):

    """
    
    Quadcopter is symmetric 
    Fgust acts at CG of quadcopter, Moments due to Fgust is not modelled
    drag force acts at CG and moments due to aerodynamics are not modelled
    
    """

    def __init__(self):

        self.total_omega = None
        super().__init__(
            name="quad_plant_model",
            input_schema={
                "mass": float,
                "inertia_tensor": np.ndarray,
                "max_thrust": float,
                "min_thrust": float,
                "thrust": float,
                "L": float,  # rolling moment
                "M": float,  # pitching moment
                "N": float,  # yawing moment
                "g": float,  # gravity constant
                "l": float,  # half arm length
                "state": np.ndarray,  # current state vector (ue, ve, we, p, q, r, phi, theta, psi, xe, ye, ze)
            },
            output_schema={
                "state_d": np.ndarray,  # state derivatives from equations of motion
            },
            parameters={},
        )

    def _run_process(self, inputs: Dict[str, Any]):
        """
        equations of motion
        :param inputs:
        :return:
        """

        a_ = np.array([  # [thrust per prop, L per prop, M per prop]
            [0.25, 0.0, -0.5/self.inputs["l"]],
            [0.25, 0.5/self.inputs["l"], 0.0],
            [0.25, 0, 0.5/self.inputs["l"]],
            [0.25, -0.5/self.inputs["l"], 0.0],
        ])

        prop_thrusts = a_ @ np.array([
            self.inputs["thrust"],
            self.inputs["L"],
            self.inputs["M"],
        ])

        prop_thrusts_clamped = np.maximum(np.minimum(prop_thrusts, self.inputs["max_thrust"]/4),
                                      self.inputs["min_thrust"]/4)

        b_ = np.array([
            [1., 1., 1., 1.],
            [0, self.inputs["l"], 0, -self.inputs["l"]],
            [-self.inputs["l"], 0, self.inputs["l"], 0]
        ])

        thrust = b_[0, :] @ prop_thrusts_clamped
        moments = np.concatenate((b_[1:3, :] @ prop_thrusts_clamped, np.array([self.inputs["N"]])))

        #########################
        #        STATES         #
        #########################
        ue, ve, we = self.inputs["state"][0:3]
        p, q, r = self.inputs["state"][3:6]
        phi, theta, psi = self.inputs["state"][6:9]
        r_i_b = get_rotation_matrix_zxy(phi=phi, theta=theta, psi=psi)

        omega = np.array([p, q, r])

        phi_d, theta_d, psi_d = get_transfer_matrix_zxy(phi=phi, theta=theta) @ \
                np.array([p, q, r])

        #########################
        #    NEWTON EQUATION    #
        #########################
        # TODO: acceleration does not include drag and gust for now
        accel = (1/self.inputs["mass"]) * (r_i_b @ np.array([0, 0, thrust]) -
                                           np.array([0, 0, self.inputs["mass"] * self.inputs["g"]]))

        #########################
        #     EULER EQUATION    #
        #########################
        # TODO: moments does not include aerodynamic moments and moments due to gust
        pqr_dot = inv(self.inputs["inertia_tensor"]) @ (moments - np.cross(omega, self.inputs["inertia_tensor"] @ omega))

        self.outputs["state_d"] = np.array([
            accel[0],
            accel[1],
            accel[2],
            pqr_dot[0],
            pqr_dot[1],
            pqr_dot[2],
            phi_d,
            theta_d,
            psi_d,
            ue,
            ve,
            we,
        ])


quad_plant_model = QuadPlantModel()
