from openai import OpenAI
from rich.console import Console

from src.ai_agent.ai_vendors.abstract_ai_vendor import AbstractAIVendor
from src.ai_agent.data_models.ai_agent_config import AIAgentConfig


class ChatgptAIVendor(AbstractAIVendor):

    def __init__(self, chatgpt_config: AIAgentConfig, api_key: str):

        super().__init__(chatgpt_config)

        self.api_key = api_key,
        self.client = OpenAI(api_key=api_key)
        self.messages = []

        self.fancy_console = Console(stderr=True, force_terminal=True)

    def _get_new_conversation_response(
            self,
            content: str,
            role: str = "system",
            awaiting_message: str = "Talking to AI agent..."
    ):
        self.messages.append({"role": role, "content": content})

        with self.fancy_console.status(f"[bold green]{awaiting_message}", spinner="dots"):
            try:
                response = self.client.chat.completions.create(
                    model=self.vendor_config.model,
                    messages=self.messages
                )

            except Exception as e:
                print("Something went wrong. Try again")
                return f"Error: {e}"

        return response.choices[0].message.content
