from typing import Dict, Any

from ModelCreator_v1 import ModelCreator
import numpy as np


# ************************************************************************* #
#                          Iterable time model                              #
# ************************************************************************* #
class TimeIterator(object):

    def __init__(self, start=0.0, dt=0.01, end=None):
        self.current = start
        self.end = end
        self.dt = dt
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns: index and time
        """
        if self.end is not None and self.current > self.end:
            raise StopIteration
        t = self.current
        i = self.index
        self.current += self.dt
        self.index += 1
        return i, t


# ************************************************************************* #
#                           Class for all inputs                            #
# ************************************************************************* #
class MasterInputModel(ModelCreator):

    def __init__(self):

        """

               ^        / body y-axis
        o  3   |       o 2 (clockwise)
         \     |     /
          \    |    /
           \   |   /
            \  |  /
          --- QUAD --->
            /  |  \
           /   |   \
          /    |    \
         /     |     \
       o  4    |       o 1 (anti-clockwise)
                       \  body x-axis

        """

        super().__init__(
            name="input_model",
            input_schema={
                "innerloop_freq": float,
                "outerloop_freq": float,
                "tf": float,
            },
            output_schema={
                "mass": float,  # mass of the quadcopter in Kg
                "inertia_tensor": np.ndarray,  # inertia tensor in Kgm^2
                "g": float,  # gravity constant in m/s^2
                "state_0": np.ndarray,  # initial state vector
                "time": np.ndarray,  # time vector computed from dt and tf
                "dt": float,  # time step from inner loop frequency
                "tf": float,  # final time for simulation (can be np.nan for infinite time as well)
                "l": float,  # half arm length in m
                "Kp_pos": np.ndarray,  # kp for position controller
                "Kd_pos": np.ndarray,  # kd for position controller
                "Kp_att": np.ndarray,  # kp for attitude controller
                "Kd_att": np.ndarray,  # kd for attitude controller
                "max_angle": float,  # constraint on the maximum angle
                "max_thrust": float,  # constraint on maximum thrust
                "min_thrust": float,  # constraint on minimum thrust
            },
            parameters={},
        )

    def _run_process(self, inputs: Dict[str, Any]):
        self.outputs["mass"] = 0.030
        self.outputs["inertia_tensor"] = np.diag([1.43e-5, 1.43e-5, 2.89e-5])
        self.outputs["g"] = 9.81
        self.outputs["state_0"] = np.array([
            0,  # ue0
            0,  # ve0
            0.,  # we0
            0,  # p0
            0,  # q0
            0,  # r0
            np.radians(0.),  # phi0
            np.radians(0.),  # tht0
            np.radians(0.),  # psi0
            0,  # x0
            0.,  # y0
            0.]  # z0
        )
        self.outputs["dt"] = 1 / self.inputs["innerloop_freq"]
        if np.isnan(self.inputs["tf"]):
            self.outputs["time"] = TimeIterator(start=0.0, dt=self.outputs["dt"])
        else:
            self.outputs["time"] = np.arange(0, self.inputs["tf"], self.outputs["dt"])
        self.outputs["tf"] = self.inputs["tf"]
        self.outputs["l"] = 0.046
        self.outputs["Kp_pos"] = np.array([15, 15, 30])
        self.outputs["Kd_pos"] = np.array([12, 12, 10])
        self.outputs["Kp_att"] = np.array([3000, 3000, 3000])
        self.outputs["Kd_att"] = np.array([300, 300, 300])
        self.outputs["max_angle"] = np.radians(40.0)
        self.outputs["max_thrust"] = 2.5 * self.outputs["mass"] * self.outputs["g"]
        self.outputs["min_thrust"] = 0.05 * self.outputs["mass"] * self.outputs["g"]


input_model = MasterInputModel()


if __name__ == "__main__":
    input_model.connect(
        innerloop_freq=100,
        outerloop_freq=10,
        tf=np.nan,
    )
    input_model.process()
    time_iter = input_model.outputs["time"]
    time_index = 0
    while time_index < 10:
        print(next(time_iter))
        time_index += 1

