import json

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class AIAgentConfig:

    vendor_keyword: str = None
    model: str = None
    preserve_generated_config: bool = None
    correct_generated_configs_folder: str = None
    wrong_generated_configs_folder: str = None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if value in ("true", "false"):
                setattr(self, key, value == "true")
            else:
                setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        """Allow access to dynamically set attributes"""
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @classmethod
    def from_dict(cls, strategy_config: dict):
        """Create config instance from dict"""
        return cls(**strategy_config)

    def to_json(self):
        """Convert config to json"""
        return json.dumps(asdict(self))
