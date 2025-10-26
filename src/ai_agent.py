import json
import os

from dotenv import load_dotenv
from pathlib import Path

from src.ai_agent.data_models.ai_agent_config import AIAgentConfig
from src.ai_agent.ai_vendors.chatgpt_ai_vendor import ChatgptAIVendor
from src.simulation_runner import SimulationRunner


class AIAgent:
    """
    AI agent wrapper for FL simulation backend. Generates experiment plan based on user-provided
    case study description, runs experiments, analyses results and provides deployment recommendations.
    """

    def __init__(
            self,
            ai_agent_config_filename: str
    ) -> None:
        self.agent_config: AIAgentConfig = AIAgentConfig.from_dict(self._load_ai_agent_config(ai_agent_config_filename))
        self.ai_vendor_api = self._assign_ai_vendor_api_and_check_token(self.agent_config)

        self.experiment_request_prompt = None
        self.generated_strategy_config_json_filepath = None
        self.experiment_result_reasoning = None

        self._config_loader = None

        self.experiment_results_dirname = None

        self.simulation_crashed = False

    def run_4ever(self):
        """
        Execute the user interaction routine.
        """
        self.obtain_user_experiment_request()
        self.generate_simulation_strategy_config()
        self.execute_requested_experiment()
        self.reason_experiment_results()

    @staticmethod
    def _load_ai_agent_config(ai_agent_config_filename) -> dict:
        """
        Load agent config from json.
        """
        config_path = f"config/{ai_agent_config_filename}"
        with open(config_path, "r") as f:
            return json.load(f)

    @staticmethod
    def _assign_ai_vendor_api_and_check_token(agent_config: AIAgentConfig):
        """
        Based on loaded config, assign vendor API. Check token is present (if not, will be prompted and added to .env).
        """

        def _ensure_api_token_provided(token_key: str):
            env_path = Path(".env")

            if env_path.exists():
                load_dotenv(dotenv_path=env_path)
                api_token = os.getenv(token_key)

                if api_token:
                    return api_token

            new_token = input(f"Please enter {token_key} token: ").strip()
            with open(env_path, "a") as f:
                f.write(f"\n{token_key}={new_token}\n")

            return new_token

        vendor_keyword = agent_config.vendor_keyword

        if vendor_keyword == "chatgpt":
            api_key = _ensure_api_token_provided("OPENAI_TOKEN")
            return ChatgptAIVendor(agent_config, api_key=api_key)

        else:
            raise RuntimeError(f"Unrecognized AI vendor: {vendor_keyword}")

    def obtain_user_experiment_request(self) -> None:
        """
        Get the case study desc from user.
        """
        print("Please describe the use case that you want to test FL on: ")
        user_input = input("> ").strip()

        if not user_input:
            print("No input provided. Please try again.")
            return self.obtain_user_experiment_request()

        self.experiment_request_prompt = user_input  # noqa

    def generate_simulation_strategy_config(self):
        """
        Generate config for requested case study.
        """
        self.generated_strategy_config_json_filepath, _ = (
            self.ai_vendor_api.generate_simulation_strategy_config(self.experiment_request_prompt)
        )

    def execute_requested_experiment(self):
        """
        Execute case study based on generated config in framework backend.
        """
        simulation_runner = SimulationRunner(self.generated_strategy_config_json_filepath)
        self.experiment_results_dirname = simulation_runner._directory_handler.dirname  # noqa

        try:
            simulation_runner.run()
        except Exception as e:  # noqa
            self.simulation_crashed = True

    def reason_experiment_results(self):
        """
        Obtain and display results reasoning and final deployment recommendations.
        """
        self.ai_vendor_api.reason_experiment_results(self.experiment_results_dirname, self.simulation_crashed)


if __name__ == "__main__":
    ai_agent = AIAgent("ai_agent_config.json")
    ai_agent.run_4ever()
