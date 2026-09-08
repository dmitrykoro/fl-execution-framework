import json
import argparse

from src.ai_agent.data_models.ai_agent_config import AIAgentConfig
from src.ai_agent.ai_vendors.chatgpt_ai_vendor import ChatgptAIVendor
from src.simulation_runner import SimulationRunner
from src.utils.common import obtain_env_var, load_json_as_str


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="AI agent of IntelliFL Framework"
    )

    parser.add_argument(
        "agent_config",
        type=str,
        nargs="?",
        default="ai_agent_config.json",
        help="LLM API config found in config/ directory"
    )

    parser.add_argument(
        "--experiment-config",
        type=str,
        nargs="?",
        default=None,
        help="Path from content root to execute existing config"
    )

    return parser.parse_args()


class AIAgent:
    """
    AI agent wrapper for FL simulation backend. Two modes of interaction:
        1.  Generates experiment plan based on user-provided case study description, runs experiments,
            analyses results and provides deployment recommendations.
        2.  Parses user-provided experiment config, runs experiments, analyses results and provides observations
            driven by user-provided experiment objective that is accompanying provided configuration.
    """

    def __init__(
            self,
            ai_agent_config_filename: str,
            provided_strategy_config_json_filepath: str = None,

    ) -> None:
        self.agent_config: AIAgentConfig = AIAgentConfig.from_dict(self._load_ai_agent_config(ai_agent_config_filename))
        self.ai_vendor_api = self._assign_ai_vendor_api_and_check_token(self.agent_config)

        self.experiment_request_prompt = None

        self.generated_strategy_config_json_filepath = provided_strategy_config_json_filepath
        self.interacting_from_scratch = provided_strategy_config_json_filepath is None

        self.experiment_result_reasoning = None

        self._config_loader = None

        self.experiment_results_dirname = None

        self.simulation_crashed = False

    def run_4ever(self):
        """
        Execute the user interaction routine.
        """

        if not self.generated_strategy_config_json_filepath:
            self.obtain_user_experiment_request("Please describe the use case that you want to test FL on: ")
            self.generate_simulation_strategy_config()

        self.obtain_user_experiment_request("You provided the configuration. Please describe the experiment objective: ")
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

        vendor_keyword = agent_config.vendor_keyword

        if vendor_keyword == "chatgpt":
            api_key = obtain_env_var("OPENAI_TOKEN")

            if not api_key:
                raise RuntimeError(
                    "The AI configuration agent needs an OpenAI API key. "
                    "Set OPENAI_TOKEN in .env.private to use it."
                )

            return ChatgptAIVendor(agent_config, api_key=api_key)

        else:
            raise RuntimeError(f"Unrecognized AI vendor: {vendor_keyword}")

    def obtain_user_experiment_request(self, user_task: str) -> None:
        """
        Get the experiment plan from user.
        """
        print(user_task)
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
        self.ai_vendor_api.reason_experiment_results(
            self.experiment_results_dirname, self.simulation_crashed,
            self.experiment_request_prompt if not self.interacting_from_scratch else None,
            self.generated_strategy_config_json_filepath if not self.interacting_from_scratch else None
        )


if __name__ == "__main__":
    args = parse_arguments()

    ai_agent = AIAgent(
        ai_agent_config_filename=args.agent_config,
        provided_strategy_config_json_filepath=args.experiment_config
    )
    ai_agent.run_4ever()
