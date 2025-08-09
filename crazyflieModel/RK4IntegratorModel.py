from typing import Dict, Any, Union

from ModelCreator_v1 import ModelCreator
from numpy import ndarray, array


# ************************************************************************* #
#                         Class for rk4 integration                         #
# ************************************************************************* #
class RK4IntegratorModel(ModelCreator):

    def __init__(self):

        self.state = []  # initialized to be empty and later will be populated
        super().__init__(
            name="rk4_integrator_model",
            input_schema={
                "dt": float,  # time step to be obtained from frequency of the control loop
                "t0": float,  # end time for the simulation, time vector will be calculated accordingly
                "state_0": ndarray,  # initial state to start the runge kutta 4 solver
                "model": Union[list, object],  # model is needed to change its inputs after every step in RK4
            },
            output_schema={
                "state": ndarray,  # new state after RK4 integration
            },
            parameters={},
        )

    def _run_process(self, inputs: Dict[str, Any]):
        if isinstance(self.inputs["model"], list):
            state_d_model  = self.inputs["model"][-1]
        else:
            state_d_model = self.inputs["model"]
            self.inputs["model"] = [self.inputs["model"]]
        [model.update_inputs(state=self.inputs["state_0"]) for model in self.inputs["model"]]
        state_d_model.process()
        k1 = self.inputs["dt"] * array(list(state_d_model.outputs["state_d"]))

        [model.update_inputs(state=self.inputs["state_0"] + 0.5 * k1) for model in self.inputs["model"]]
        state_d_model.process()
        k2 = self.inputs["dt"] * array(list(state_d_model.outputs["state_d"]))

        [model.update_inputs(state=self.inputs["state_0"] + 0.5 * k2) for model in self.inputs["model"]]
        state_d_model.process()
        k3 = self.inputs["dt"] * array(list(state_d_model.outputs["state_d"]))

        [model.update_inputs(state=self.inputs["state_0"] + k3) for model in self.inputs["model"]]
        state_d_model.process()
        k4 = self.inputs["dt"] * array(list(state_d_model.outputs["state_d"]))

        self.outputs["state"] = self.inputs["state_0"] + (1. / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)


rk4_integration_model = RK4IntegratorModel()
