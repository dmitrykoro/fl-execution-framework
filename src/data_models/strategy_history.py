from dataclasses import dataclass
from typing import List

from simulation_strategy_config import StrategyConfig
from client_history import ClientHistory


@dataclass
class StrategyHistory:
    """Keeps metrics history for the strategy"""

    strategy_config: StrategyConfig
    clients_history: List[ClientHistory]

    def __init__(self, strategy_config: StrategyConfig) -> None:
        self.strategy_config = strategy_config
        self.clients_history = list()

    def add_client_history(self, client_history: ClientHistory) -> None:
        """Add new client's history to the list of histories"""

        self.clients_history.append(client_history)
