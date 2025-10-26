import json
import datetime
import os
import shutil
import glob

from abc import ABC, abstractmethod
from jsonschema import ValidationError

from src.config_loaders.config_loader import ConfigLoader
from src.ai_agent.data_models.ai_agent_config import AIAgentConfig


class AbstractAIVendor(ABC):
    """
    Wrapper for requests to vendor APIs.
    """

    def __init__(self, vendor_config: AIAgentConfig):
        self.vendor_config = vendor_config

        self.list_of_framework_config_context_file_paths = [
            "src/config_loaders/validate_strategy_config.py",
            "src/data_models/simulation_strategy_config.py",
            "README.md",
        ]
        self.list_of_positive_configuration_files = [
            "config/simulation_strategies/example_strategy_config.json",
        ]
        self.list_of_result_context_files = [
            "csv/exec_stats_*",
            "csv/round_metrics_*",
        ]

        self.framework_config_context = self._get_all_file_contents(self.list_of_framework_config_context_file_paths)
        self.framework_positive_config_examples = self._get_all_file_contents(self.list_of_positive_configuration_files)

        self._config_loader = None

        self.generated_config_json_filepath = None

    @abstractmethod
    def _get_new_conversation_response(self, *args, **kwargs):
        """
        Get vendor-specific conversation response.
        """
        pass

    @staticmethod
    def _get_all_file_contents(list_of_filepaths) -> str:
        """
        Load all files and append to single str object. Used to provide context for the agent.
        """
        all_content = ""

        for filepath in list_of_filepaths:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    all_content += f"{f.read()}\n"

            except Exception as e:
                print(f"Couldn't read the framework context file: {filepath}. Error message: {e}")
                continue

        return all_content

    def validate_generated_config(self, generated_config):
        """
        Validate generated config using ConfigLoader from FL backend.
        """
        self.generated_config_json_filepath = self._save_generated_config_json(generated_config)

        try:
            self._config_loader = ConfigLoader(usecase_config_path=self.generated_config_json_filepath)

        except ValidationError as e:
            return False, f"{e}"

        return True, ""

    def update_final_generated_config(self, user_prompt, agent_reasoning, num_ai_config_corrections):
        """
        Update successfully generated config with additional info such as user prompt, agent reasoning
        and number of correction iterations that were required when the validation failed.
        """
        with open(self.generated_config_json_filepath, "r") as f:
            data = json.load(f)

        data["shared_settings"]["user_prompt_for_ai_generated"] = user_prompt
        data["shared_settings"]["ai_agent_config_reasoning"] = agent_reasoning
        data["shared_settings"]["num_ai_config_corrections"] = num_ai_config_corrections

        with open(self.generated_config_json_filepath, "w") as f:
            json.dump(data, f, indent="\t")

    def _update_wrong_generated_config_with_error(self, error_msg):
        """
        Insert validation error message into incorrect config JSON for future reference.
        """
        with open(self.generated_config_json_filepath, "r") as f:
            data = json.load(f)

        data["shared_settings"]["ai_configuration_error"] = error_msg

        with open(self.generated_config_json_filepath, "w") as f:
            json.dump(data, f, indent="\t")

    def _save_generated_config_json(self, generated_config) -> str:
        """
        Save the generated config as JSON for further processing in the framework.
        """
        file_data = json.loads(generated_config)
        filepath = (
            f"{self.vendor_config.correct_generated_configs_folder}"
            f"experiment_{str(datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S'))}.json"
        )

        with open(filepath, "w") as f:
            json.dump(file_data, f, indent="\t")

        return filepath

    def _delete_generated_config_json(self):
        """
        Delete generated config. Can be used when user requests config adjustment.
        """
        os.remove(self.generated_config_json_filepath)

    def _move_incorrect_config(self):
        """
        Move incorrect config for future consideration in the agent scope of incorrect configs.
        """
        shutil.move(self.generated_config_json_filepath, self.vendor_config.wrong_generated_configs_folder)

    def generate_simulation_strategy_config(self, experiment_request_prompt):
        """
        Prompt user for config until satisfaction.
        """

        prompt = f"""
            You are a part of an expert system that helps to facilitate federated learning experiments by generating
            experiment configurations that are then used as an input for experimentation framework. 
            The following file contents provide the specification of possible experiment configuration entries,
            and it's validation logic:    
            {self.framework_config_context}
            Examples of correct configuration:
            {self.framework_positive_config_examples}
            Based on this information, generate the experiment configuration that would address the following research question:
            {experiment_request_prompt}
        """

        generated_config, num_corrections = self._generate_and_validate_config(
            prompt, "Awaiting AI to generate case study..."
        )

        print(f"\nGenerated configuration:\n{generated_config}")
        reasoning = self._explain_config(generated_config)

        while True:
            user_choice = input("Do you want to run this case study (Y) or refine it (N)? ").strip().lower()
            if user_choice == "y":
                break

            adjustment_user_request = input("Describe what you'd like to adjust: ").strip()

            adjustment_prompt = f"""
                The user was not satisfied with the previous configuration.
                Adjust the JSON configuration as follows: {adjustment_user_request}.
                Generate and provie the refined JSON configuration file only. 
            """

            generated_config, corrections = self._generate_and_validate_config(
                adjustment_prompt, "Awaiting AI to adjust case study..."
            )
            num_corrections += corrections

            print(f"\nRefined configuration:\n{generated_config}")
            reasoning = self._explain_config(generated_config)

        self.update_final_generated_config(
            user_prompt=experiment_request_prompt,
            agent_reasoning=reasoning,
            num_ai_config_corrections=num_corrections
        )

        return self.generated_config_json_filepath, generated_config

    def _generate_and_validate_config(self, base_prompt, awaiting_message):
        """
        Ask the AI to generate config and ensure it's valid.
        """

        config = self._get_new_conversation_response(content=base_prompt, awaiting_message=awaiting_message)
        num_corrections = 0

        while True:
            config_is_valid, error = self.validate_generated_config(config)

            if config_is_valid:
                break

            num_corrections += 1
            self._update_wrong_generated_config_with_error(error)
            self._move_incorrect_config()

            config = self._get_new_conversation_response(
                content=f"The generated config is not valid. The error: {error}."
                        f" Fix the error and regenerate configuration. Provide JSON configuration only.",
                awaiting_message="AI made a mistake in configuration. Awaiting AI to fix it..."
            )

        return config, num_corrections

    def _explain_config(self, config):
        """
        Ask AI to explain configuration.
        """

        self.messages.append({"role": "assistant", "content": config})

        reasoning = self._get_new_conversation_response(
            content="Explain the plan for the experiment in a user-friendly way: what will be tested and why.",
            awaiting_message="Awaiting AI reasoning..."
        )
        print(reasoning)

        return reasoning

    def _get_all_result_context(self, results_dirname: str) -> str:
        """
        Collect all files matching patterns in self.list_of_result_context_files
        and return their combined plain-text contents.
        """

        all_content = ""

        for pattern in self.list_of_result_context_files:
            matched_files = glob.glob(results_dirname + "/" + pattern)

            if not matched_files:
                all_content += f"No files matched pattern: {pattern}\n"
                continue

            for filepath in matched_files:
                try:
                    filename = os.path.basename(filepath)

                    with open(filepath, "r", encoding="utf-8") as f:
                        file_text = f.read().strip()
                    all_content += f"{filename}:\n{file_text}\n\n"

                except Exception as e:
                    all_content += f"Error reading {filepath}: {e}\n"

        return all_content

    def reason_experiment_results(self, results_dirname: str, simulation_crashed: bool):
        """
        Ask AI to provide reasoning on experiment results.
        """
        results_content = self._get_all_result_context(results_dirname)
        prompt = ""

        if simulation_crashed:
            prompt += "This experiment crashed due to a problem, but incomplete results are still available."

        prompt += f"""
            Results of the case study include plots and csv files. 
            You are provided with the content of the following files:
            - exec_stats_*.csv: stats summary for the entire case execution from "simulation_strategies" JSON.
            - round_metrics_*.csv: metrics for each aggregation round, including model performance and client removal 
            performance. Also includes the time used by the aggregation strategy at each round.
            Each * corresponds to the entry from simulation_strategies array, representing the results for separate case.
            Below are file contents:
            {results_content}
            Explain the results, and provide recommendation for the user, considering their setup and 
            case study observations from the resulting files. 
         """

        results_explanation = self._get_new_conversation_response(
            content=prompt, awaiting_message="Awaiting AI explanation of results..."
        )
        print(results_explanation)
