from dataclasses import dataclass
from typing import List

from round_history_entry import RoundHistoryEntry


@dataclass
class ClientHistory:
    """Keeps history for each round metrics for a particular client"""

    client_id: str
    rounds_history: List[RoundHistoryEntry]

    def __init__(self, client_id: str) -> None:
        self.client_id = client_id
        self.rounds_history = list()

    def add_round_history_entry(self, round_history_entry: RoundHistoryEntry) -> None:
        """Add new round metrics to the rounds history of the client"""

        self.rounds_history.append(round_history_entry)
