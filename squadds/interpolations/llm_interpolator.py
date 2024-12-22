from collections.abc import Callable
from typing import List, Union
import pandas as pd

from langchain_core.language_models.base import LanguageModelOutput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel, Field
from pyEPR.calcs import Convert

from squadds import Analyzer
from squadds.core.utils import create_unified_design_options
from squadds.interpolations.interpolator import Interpolator


class InterpolatingQubitClaw(BaseModel):
    """
    Pydantic model to represent the interpolation parameters for the qubit and claw.

    This class is used to obtain a structured output from the chat model for the qubit and claw design.

    Attributes:
        qubit_cross_length (float): Interpolated cross length of the qubit in micrometers.
        qubit_claw_length (float): Interpolated claw length of the qubit in micrometers.
    """

    qubit_cross_length: float = Field(
        ..., description="Interpolated cross length of the qubit in micrometers."
    )
    qubit_claw_length: float = Field(
        ..., description="Interpolated claw length of the qubit in micrometers."
    )


class InterpolatingCavityCoupler(BaseModel):
    """
    Pydantic model to represent the interpolation parameters for the cavity and coupler.

    This class is used to obtain a structured output from the chat model for the cavity and coupler design.

    Attributes:
        resonator_length (float): Interpolated length of the cavity resonator in micrometers.
        coupling_length (float): Interpolated length of the coupling element in micrometers.
    """

    resonator_length: float = Field(
        ..., description="Interpolated length of the cavity resonator in micrometers."
    )
    coupling_length: float = Field(
        ..., description="Interpolated length of the coupling element in micrometers."
    )


class LLMInterpolator(Interpolator):
    """

    Class for LLM-based interpolation.

    This class provides an interface to utilize LLMs/Chat Models to interpolate
    the design options for qubit and cavity based on target parameters. Similar to the scaling
    interpolator, it first finds the closest qubit-claw and cavity-coupler designs from the database
    and then uses the LLM to interpolate the design options based on the target parameters.

    We use structured output or tool calls to ensure that the chat model returns the
    required parameters for the qubit and cavity designs. Therefore, a compatible Langchain chat model
    should be used.

    Attributes:
        qubit_claw_prompts (List[BaseMessage]): Default prompts for the qubit and claw design.
        cavity_coupler_prompts (List[BaseMessage]): Default prompts for the cavity and coupler design.

    """

    qubit_claw_prompts: List[BaseMessage] = [
        SystemMessage(
            content="We are working with super-conducting qubits and using the SQuADDS database to retrieve simulation data. The given input dictionary contains the closest available simulation results for the target qubit and claw. The goal is to provide the best guess for the dimensions of the qubit and claw based on target parameters."
        ),
        HumanMessage(
            content="Given the closest qubit-claw design with:\n- Closest simulated system parameters: {input_params}\n- Target qubit frequency (GHz): {f_q_target}\n- Target anharmonicity (MHz): {alpha_target}\n- Target coupling strength (MHz): {g_target}\n\nCalculate the cross length and claw length."
        ),
    ]
    cavity_coupler_prompts: List[BaseMessage] = [
        SystemMessage(
            content="We are working with super-conducting qubits and using the SQuADDS database to retrieve simulation data. The given input dictionary contains the closest available simulation results for the target cavity and coupler. The goal is to provide the best guess for the dimensions of the resonator and coupling element based on target parameters."
        ),
        HumanMessage(
            content="Given the closest cavity-coupler design with:\n- Closest simulated system parameters: {input_params}\n- Target cavity frequency (GHz): {f_res_target}\n- Target coupling strength (MHz): {kappa_target}\n\nCalculate the resonator length and coupling length."
        ),
    ]

    def __init__(self, analyzer: Analyzer, target_params: dict):
        super().__init__(analyzer, target_params)

    def get_design(
        self,
        chat_model: Union[
            BaseChatModel,
            Callable[
                [List[BaseMessage], str],
                InterpolatingQubitClaw | InterpolatingCavityCoupler,
            ],
        ],
        read_tool_calls: Callable[[LanguageModelOutput, str], dict] = None,
        qubit_claw_prompt: List[BaseMessage] = None,
        cavity_coupler_prompt: List[BaseMessage] = None,
        format_prompts: bool = False,
    ) -> pd.DataFrame:
        """
        Builds the design options for qubit and cavity based on target parameters.

        This method is customizable to support different scenarios.
        Based on the input chat model and other parameters, the user can:
            - Use a structured output compatible chat model to calculate the design options.
            - Use tool calls to read the structured output from the chat model.
            - Use a self-defined chat model to calculate the design options.
            - Use custom prompts for the qubit-claw and cavity-coupler designs.
            - Use placeholders in user-defined prompts to format the prompts based on the input parameters.

        To check if your chat model supports structured output or tool calls, you can use the `test_chat_structured_output`
        and `test_chat_tool_calls` functions from utils.

        Args:
            chat_model (Union[BaseChatModel, Callable[[List[BaseMessage], str], InterpolatingQubitClaw | InterpolatingCavityCoupler]):
                The chat model can be:
                    - A structured output compatible Langchain chat model.
                    - A tool call compatible Langchain chat model with the `read_tool_calls` function.
                    - A callable function that takes a list of prompts and the output type (InterpolatingQubitClaw or InterpolatingCavityCoupler) as input
                    and returns the structured output.

            read_tool_calls (Callable[[LanguageModelOutput, str], dict], optional): A function to read the tool calls from the chat model output. If provided, the chat model will be invoked with the tool calls. Defaults to None.
                The function should take the chat model output and the tool name ("InterpolatingQubitClaw" or "InterpolatingCavityCoupler") as input and return the tool call data as a dictionary.
                The dictionary data is then used to create the InterpolatingQubitClaw or InterpolatingCavityCoupler instance.

            qubit_claw_prompt (List[BaseMessage], optional): Prompts for the qubit-claw design. Defaults to None.
                If not provided, the default prompts will be used. Required if `format_prompts` is True.
                If `format_prompts` is True, the second message should contain placeholders for the input parameters and target parameters.
                The placeholders should be formatted as `{input_params}`, `{f_q_target}`, `{alpha_target}`, and `{g_target}`.

            cavity_coupler_prompt (List[BaseMessage], optional): Prompts for the cavity-coupler design. Defaults to None.
                If not provided, the default prompts will be used. Required if `format_prompts` is True.
                If `format_prompts` is True, the second message should contain placeholders for the input parameters and target parameters.
                The placeholders should be formatted as `{input_params}`, `{f_res_target}`, and `{kappa_target}`.

            format_prompts (bool, optional): A flag to indicate whether to format the prompts. Defaults to False.
                If True, the prompts will be formatted based on the input parameters and target parameters.
                The placeholders in the prompts should be formatted as `{input_params}`, `{f_q_target}`, `{alpha_target}`, `{g_target}`, `{f_res_target}`, and `{kappa_target}`.

        Returns:
            pd.DataFrame: A DataFrame containing the design options for qubit and cavity.

        Raises:
            ValueError: If format_prompts is True and no prompts are provided for qubit-claw and cavity-coupler designs.
            ValueError: If the chat model does not return an instance of InterpolatingQubitClaw or InterpolatingCavityCoupler.
            ValueError: If an error occurs while reading the tool call data.
        """
        # Extract target parameters
        f_q_target = self.target_params["qubit_frequency_GHz"]
        g_target = self.target_params["g_MHz"]
        alpha_target = self.target_params["anharmonicity_MHz"]
        f_res_target = self.target_params["cavity_frequency_GHz"]
        kappa_target = self.target_params["kappa_kHz"]

        try:
            res_type = self.target_params["resonator_type"]
        except:
            res_type = self.analyzer.selected_resonator_type

        self.df = self.analyzer.df

        # If format prompts is True, but no prompts are provided, raise an error
        if format_prompts and not qubit_claw_prompt and not cavity_coupler_prompt:
            raise ValueError(
                "Format Prompts is set to True. Please provide prompts for qubit-claw and cavity-coupler designs."
            )

        # Find the closest qubit-claw design
        if self.analyzer.selected_resonator_type == "half":
            closest_qubit_claw_design = self.analyzer.find_closest(
                {
                    "qubit_frequency_GHz": f_q_target,
                    "anharmonicity_MHz": alpha_target,
                    "g_MHz": g_target,
                },
                parallel=True,
                num_cpu="auto",
                num_top=1,
            )
        else:
            closest_qubit_claw_design = self.analyzer.find_closest(
                {
                    "qubit_frequency_GHz": f_q_target,
                    "anharmonicity_MHz": alpha_target,
                    "g_MHz": g_target,
                },
                num_top=1,
            )

        # Pass only the relevant columns to the chat model
        input_params = closest_qubit_claw_design[
            ["design_options_qubit", "design_options_cavity_claw", "design_options"]
        ].to_dict(orient="records")[0]

        # Initialize qubit-claw prompts
        if qubit_claw_prompt:
            if format_prompts:
                self.qubit_claw_prompts: List[BaseMessage] = [
                    qubit_claw_prompt[0],
                    HumanMessage(
                        content=qubit_claw_prompt[1].content.format(
                            input_params=input_params,
                            f_q_target=f_q_target,
                            alpha_target=alpha_target,
                            g_target=g_target,
                        )
                    ),
                ]
            else:
                self.qubit_claw_prompts: List[BaseMessage] = qubit_claw_prompt
        else:
            self.qubit_claw_prompts: List[BaseMessage] = [
                LLMInterpolator.qubit_claw_prompts[0],
                HumanMessage(
                    content=LLMInterpolator.qubit_claw_prompts[1].content.format(
                        input_params=input_params,
                        f_q_target=f_q_target,
                        alpha_target=alpha_target,
                        g_target=g_target,
                    )
                ),
            ]

        if isinstance(chat_model, BaseChatModel):
            if read_tool_calls:
                # Bind the chat model with the structured output
                chat_model_binded = chat_model.bind_tools([InterpolatingQubitClaw])
                # Invoke the chat model with the qubit-claw prompts
                resp = chat_model_binded.invoke(self.qubit_claw_prompts)
                tool_call_data = read_tool_calls(resp, "InterpolatingQubitClaw")
                try:
                    qubit_claw_design = InterpolatingQubitClaw(**tool_call_data)
                except Exception as e:
                    raise ValueError(
                        f"Error reading tool call data: {e}. Tool call data: {tool_call_data}"
                    )
            else:
                # Use structured output to get the qubit-claw design
                structured_chat_model = chat_model.with_structured_output(
                    InterpolatingQubitClaw
                )
                qubit_claw_design = structured_chat_model.invoke(
                    self.qubit_claw_prompts
                )
                if not isinstance(qubit_claw_design, InterpolatingQubitClaw):
                    raise ValueError(
                        "The chat model did not return an instance of InterpolatingQubitClaw."
                    )
        else:
            # Invoke the chat model with the qubit-claw prompts
            qubit_claw_design = chat_model(
                self.qubit_claw_prompts, "InterpolatingQubitClaw"
            )
            if not isinstance(qubit_claw_design, InterpolatingQubitClaw):
                raise ValueError(
                    "The chat model did not return an instance of InterpolatingQubitClaw."
                )

        print("Design options for qubit and claw from chat model:")
        print(qubit_claw_design)

        # Scaling logic for cavity-coupler designs
        # Filter DataFrame based on qubit coupling claw capacitance
        try:
            cross_to_claw_cap_chosen = closest_qubit_claw_design["cross_to_claw"].iloc[
                0
            ]
        except:
            cross_to_claw_cap_chosen = closest_qubit_claw_design[
                "cross_to_claw_closest"
            ].iloc[0]

        threshold = 0.3  # 30% threshold
        try:
            filtered_df = self.df[
                (self.df["cross_to_claw"] >= (1 - threshold) * cross_to_claw_cap_chosen)
                & (
                    self.df["cross_to_claw"]
                    <= (1 + threshold) * cross_to_claw_cap_chosen
                )
            ]
        except:
            filtered_df = self.df[
                (
                    self.df["cross_to_claw_closest"]
                    >= (1 - threshold) * cross_to_claw_cap_chosen
                )
                & (
                    self.df["cross_to_claw_closest"]
                    <= (1 + threshold) * cross_to_claw_cap_chosen
                )
            ]

        # Find the closest cavity-coupler design
        merged_df = self.analyzer.df.copy()
        system_chosen = self.analyzer.selected_system
        H_params_chosen = self.analyzer.H_param_keys

        self.analyzer.df = filtered_df
        self.analyzer.selected_system = "cavity_claw"
        self.analyzer.H_param_keys = [
            "resonator_type",
            "cavity_frequency_GHz",
            "kappa_kHz",
        ]
        self.analyzer.target_params = {
            "cavity_frequency_GHz": f_res_target,
            "kappa_kHz": kappa_target,
            "resonator_type": res_type,
        }

        target_params_cavity = {
            "cavity_frequency_GHz": f_res_target,
            "kappa_kHz": kappa_target,
            "resonator_type": res_type,
        }

        if self.analyzer.selected_resonator_type == "half":
            closest_cavity_cpw_design = self.analyzer.find_closest(
                target_params_cavity, parallel=True, num_cpu="auto", num_top=1
            )
        else:
            closest_cavity_cpw_design = self.analyzer.find_closest(
                target_params_cavity, num_top=1
            )

        # Pass only the relevant columns to the chat model
        input_params = closest_cavity_cpw_design[
            ["design_options_qubit", "design_options_cavity_claw", "design_options"]
        ].to_dict(orient="records")[0]

        # Initialize cavity-coupler prompts
        if cavity_coupler_prompt:
            if format_prompts:
                self.cavity_coupler_prompts: List[BaseMessage] = [
                    cavity_coupler_prompt[0],
                    HumanMessage(
                        content=cavity_coupler_prompt[1].content.format(
                            input_params=input_params,
                            f_res_target=f_res_target,
                            kappa_target=kappa_target,
                        )
                    ),
                ]
            else:
                self.cavity_coupler_prompts: List[BaseMessage] = cavity_coupler_prompt
        else:
            self.cavity_coupler_prompts: List[BaseMessage] = [
                LLMInterpolator.cavity_coupler_prompts[0],
                HumanMessage(
                    content=LLMInterpolator.cavity_coupler_prompts[1].content.format(
                        input_params=input_params,
                        f_res_target=f_res_target,
                        kappa_target=kappa_target,
                    )
                ),
            ]

        if isinstance(chat_model, BaseChatModel):
            if read_tool_calls:
                # Bind the chat model with the structured output
                chat_model_binded = chat_model.bind_tools([InterpolatingCavityCoupler])
                # Invoke the chat model with the cavity-coupler prompts
                resp = chat_model_binded.invoke(self.cavity_coupler_prompts)
                tool_call_data = read_tool_calls(resp, "InterpolatingCavityCoupler")
                try:
                    cavity_coupler_design = InterpolatingCavityCoupler(**tool_call_data)
                except Exception as e:
                    raise ValueError(
                        f"Error reading tool call data: {e}. Tool call data: {tool_call_data}"
                    )
            else:
                # Use structured output to get the cavity-coupler design
                structured_chat_model = chat_model.with_structured_output(
                    InterpolatingCavityCoupler
                )
                cavity_coupler_design = structured_chat_model.invoke(
                    self.cavity_coupler_prompts
                )
                if not isinstance(cavity_coupler_design, InterpolatingCavityCoupler):
                    raise ValueError(
                        "The chat model did not return an instance of InterpolatingCavityCoupler."
                    )
        else:
            # Invoke the chat model with the cavity-coupler prompts
            cavity_coupler_design = chat_model(
                self.cavity_coupler_prompts, "InterpolatingCavityCoupler"
            )
            if not isinstance(cavity_coupler_design, InterpolatingCavityCoupler):
                raise ValueError(
                    "The chat model did not return an instance of InterpolatingCavityCoupler."
                )

        print("Design options for cavity and coupler from chat model:")
        print(cavity_coupler_design)

        # Reset the analyzer's DataFrame
        self.analyzer.df = merged_df
        self.analyzer.selected_system = system_chosen
        self.analyzer.H_param_keys = H_params_chosen

        # a dataframe with three empty colums
        interpolated_designs_df = pd.DataFrame(
            columns=[
                "design_options_qubit",
                "design_options_cavity_claw",
                "design_options",
            ]
        )

        # Update the qubit and cavity design options
        qubit_design_options = closest_qubit_claw_design["design_options_qubit"].iloc[0]
        qubit_design_options["cross_length"] = (
            f"{qubit_claw_design.qubit_cross_length}um"
        )
        qubit_design_options["connection_pads"]["readout"]["claw_length"] = (
            f"{qubit_claw_design.qubit_claw_length}um"
        )
        required_Lj = Convert.Lj_from_Ej(
            closest_qubit_claw_design["EJ"].iloc[0], units_in="GHz", units_out="nH"
        )
        qubit_design_options["aedt_hfss_inductance"] = required_Lj * 1e-9
        qubit_design_options["aedt_q3d_inductance"] = required_Lj * 1e-9
        qubit_design_options["q3d_inductance"] = required_Lj * 1e-9
        qubit_design_options["hfss_inductance"] = required_Lj * 1e-9
        qubit_design_options["connection_pads"]["readout"]["Lj"] = f"{required_Lj}nH"

        # setting the `claw_cpw_length` params to zero
        qubit_design_options["connection_pads"]["readout"]["claw_cpw_length"] = "0um"

        cavity_design_options = closest_cavity_cpw_design[
            "design_options_cavity_claw"
        ].iloc[0]
        cavity_design_options["cpw_opts"]["total_length"] = (
            f"{cavity_coupler_design.resonator_length}um"
        )

        if self.analyzer.selected_resonator_type == "half":
            cavity_design_options["cplr_opts"]["finger_length"] = (
                f"{cavity_coupler_design.coupling_length}um"
            )
        else:
            cavity_design_options["cplr_opts"]["coupling_length"] = (
                f"{cavity_coupler_design.coupling_length}um"
            )

        # update the claw of the cavity based on the one from the qubit
        cavity_design_options["claw_opts"]["connection_pads"] = qubit_design_options[
            "connection_pads"
        ]

        interpolated_designs_df = pd.DataFrame(
            {
                "coupler_type": self.analyzer.selected_coupler,
                "design_options_qubit": [qubit_design_options],
                "design_options_cavity_claw": [cavity_design_options],
                "setup_qubit": [closest_qubit_claw_design["setup_qubit"].iloc[0]],
                "setup_cavity_claw": [
                    closest_cavity_cpw_design["setup_cavity_claw"].iloc[0]
                ],
            }
        )

        device_design_options = create_unified_design_options(
            interpolated_designs_df.iloc[0]
        )

        # add the device design options to the dataframe
        interpolated_designs_df["design_options"] = [device_design_options]
        interpolated_designs_df.iloc[0]["design_options"]["qubit_options"][
            "connection_pads"
        ]["readout"]["claw_cpw_length"] = "0um"

        return interpolated_designs_df
