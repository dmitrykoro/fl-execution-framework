import json
from dataclasses import dataclass, asdict


@dataclass
class StrategyConfig:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if value in ("true", "false"):
                setattr(self, key, value == "true")
            else:
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, strategy_config: dict):
        """Create config instance from dict"""
        return cls(**strategy_config)

    def to_json(self):
        return json.dumps(asdict(self))
