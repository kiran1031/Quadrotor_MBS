# -*- coding: utf-8 -*-
"""
A module for creating and connecting modular components using an abstract base class.

This version allows for explicit, granular connections, mapping specific outputs
from source models to specific inputs on destination models. It also allows
inputs to be set directly from Python variables and updated after initial connection.
"""

# --- Required Imports ---
from abc import ABC, abstractmethod
from functools import reduce
import operator
import timeit
from typing import Any, Dict, NamedTuple, Union, Tuple

# --- Type Hint Definitions ---

# The path can be a single string or a tuple of strings/ints
OutputPath = Union[str, Tuple[Union[str, int], ...]]

# OutputRef stores a reference to another model's output
OutputRef = NamedTuple('OutputRef', [('model', 'GloryModule'), ('output_path', tuple)])

# An input can now be a reference to another model's output OR any Python value.
InputSource = Union[OutputRef, Any]


class ModelCreator(ABC):
    """
    An abstract base class for a connectable, modular component.

    This version supports connecting specific outputs to specific inputs, setting
    inputs directly from Python variables, and updating those inputs dynamically.

    Attributes:
        name (str): The name of the model component.
        input_schema (Dict[str, Any]): Defines expected input names and types.
        output_schema (Dict[str, Any]): Defines output names and types.
        parameters (Dict[str, Any]): Configuration parameters.
        inputs (Dict[str, Any]): Inputs received during processing.
        outputs (Dict[str, Any]): Outputs produced after processing.
        is_processed (bool): A flag to indicate if the model has been processed.
    """

    def __init__(self, name: str, input_schema: Dict[str, Any], output_schema: Dict[str, Any],
                 parameters: Dict[str, Any]):
        """Initializes a new ModelCreator instance."""
        self.name = name
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.parameters = parameters
        self.inputs: Dict[str, Any] = {}
        self.outputs: Dict[str, Any] = {}
        self._connections: Dict[str, InputSource] = {}
        self._start_time = None
        self.is_processed = False

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the model."""
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"inputs={list(self.input_schema.keys())}, "
                f"outputs={list(self.output_schema.keys())})")

    def __new__(cls, *args, **kwargs):
        """New method override to prevent direct instantiation."""
        if cls is ModelCreator:
            raise TypeError(f"only children of '{cls.__name__}' may be instantiated")
        return super().__new__(cls)

    def start_timer(self):
        self._start_time = timeit.default_timer()

    def end_timer(self):
        if self._start_time is not None:
            print("")
            print(
                "%s took %.6f seconds."
                % (self.name, timeit.default_timer() - self._start_time)
            )
            self._start_time = None

    def __getitem__(self, path: OutputPath) -> OutputRef:
        """Creates a reference to a specific output path of this model."""
        output_path = path if isinstance(path, tuple) else (path,)
        top_level_key = output_path[0]
        if top_level_key not in self.output_schema:
            raise KeyError(
                f"Cannot create reference. Model '{self.name}' has no output named '{top_level_key}'. "
                f"Available outputs: {list(self.output_schema.keys())}"
            )
        return OutputRef(model=self, output_path=output_path)

    def connect(self, **input_connections: InputSource) -> None:
        """
        Connects this model's inputs for the first time.

        Args:
            **input_connections: Keyword arguments where the key is an input
                name and the value is an OutputRef or a direct Python value.
        """
        # print(f"Connecting inputs for '{self.name}'...")
        self._connections = {}

        for dest_input_name, source in input_connections.items():
            if dest_input_name not in self.input_schema:
                raise KeyError(
                    f"Connection failed! Model '{self.name}' has no input named '{dest_input_name}'.")

            if isinstance(source, OutputRef):
                self._connections[dest_input_name] = source
                # print(f"  - Mapped input '{dest_input_name}' <- "
                #       f"'{source.model.name}' output path '{source.output_path}'")
            else:
                self._connections[dest_input_name] = source
                # print(f"  - Set direct input '{dest_input_name}' = {source} ({type(source).__name__})")

        missing_inputs = [key for key in self.input_schema if key not in self._connections]
        if missing_inputs:
            raise ValueError(
                f"Connection failed for '{self.name}'. Missing required input connections: {missing_inputs}."
            )
        # print(f"-> Successfully connected '{self.name}'.")

    ### NEW ###
    def reset(self) -> None:
        """
        Resets the model to its pre-processed state.

        This clears any cached inputs and outputs and sets the is_processed
        flag to False, forcing the model to be re-run. This is called
        automatically by `update_inputs()`.
        """
        # Only print a message if there's actually a state to reset
        if self.is_processed:
            print(f"** Resetting model '{self.name}' **")

        self.is_processed = False
        self.inputs = {}
        self.outputs = {}

    ### NEW ###
    def update_inputs(self, **new_inputs: InputSource) -> None:
        """
        Updates one or more input connections after the initial connection.

        This method allows for changing an input to a new source, which can
        be an OutputRef from another model or a direct Python value.

        *** Important ***: Calling this on a model will RESET its state (and
        clear its outputs), requiring it to be re-processed. Any downstream
        models will also re-process automatically when their `process()`
        method is called.

        Args:
            log: to log the operation
            **new_inputs: Keyword arguments where the key is the name
                of an input to update, and the value is the new source.
        """

        # print(f"\nUpdating inputs for '{self.name}'...")

        # If we change inputs, the model's current processed state is now invalid.
        # This reset is critical for ensuring correctness on the next run.
        self.reset()

        for input_name, new_source in new_inputs.items():
            # 1. Validate that the input being updated actually exists in the schema.
            if input_name not in self.input_schema:
                raise KeyError(
                    f"Update failed! Model '{self.name}' has no input named '{input_name}'. "
                    f"You can only update existing inputs."
                )

            # 2. Update the connection with the new source.
            self._connections[input_name] = new_source

            # 3. Provide clear feedback to the user about what changed.
            # if isinstance(new_source, OutputRef):
            #     print(f"  - Updated input '{input_name}' <- "
            #           f"'{new_source.model.name}' output path '{new_source.output_path}'")
            # else:
            #     print(f"  - Updated direct input '{input_name}' = {new_source} ({type(new_source).__name__})")

        # print(f"-> Update complete for '{self.name}'. Model must be re-processed.")

    def _gather_inputs(self) -> Dict[str, Any]:
        """Gathers inputs from connections or uses directly assigned values."""
        if not self._connections:
            return {}

        gathered_inputs: Dict[str, Any] = {}
        for dest_input_name, source_ref_or_value in self._connections.items():
            if isinstance(source_ref_or_value, OutputRef):
                source_ref = source_ref_or_value
                source_model = source_ref.model
                if not source_model.is_processed:
                    raise RuntimeError(
                        f"Programming error: Cannot gather inputs for '{self.name}'. "
                        f"The connected model '{source_model.name}' has not been processed yet."
                    )
                try:
                    output_value = reduce(operator.getitem, source_ref.output_path, source_model.outputs)
                    gathered_inputs[dest_input_name] = output_value
                except (KeyError, TypeError) as e:
                    raise KeyError(f"Failed to gather input '{dest_input_name}' for '{self.name}'. "
                                   f"Invalid path {source_ref.output_path} in output from '{source_model.name}': {e}")
            else:
                gathered_inputs[dest_input_name] = source_ref_or_value

        return gathered_inputs

    def process(self):
        """Runs the model's logic after processing all its dependencies."""
        if self.is_processed:
            return

        dependency_models = {
            ref.model for ref in self._connections.values() if isinstance(ref, OutputRef)
        }

        for model in dependency_models:
            model.process()

        # print(f"--- Running process: '{self.name}' ---")

        self.start_timer()
        self.inputs = self._gather_inputs()
        self._run_process(self.inputs)
        self.is_processed = True
        self.end_timer()

    @abstractmethod
    def _run_process(self, inputs: Dict[str, Any]):
        """The core logic of the model. Subclasses MUST implement this method."""
        pass
