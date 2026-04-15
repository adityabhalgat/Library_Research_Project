"""Prompts package."""

from .strategy import PromptStrategy
from .monolithic import MonolithicStrategy
from .chain_of_thought import ChainOfThoughtStrategy
from .expert_architect import ExpertArchitectStrategy
from .cove import ChainOfVerificationStrategy
from .rcot import ReverseChainOfThoughtStrategy
from .two_model import TwoModelStrategy
from .base import ProjectInputs

__all__ = [
    "get_prompt_strategy",
    "ProjectInputs"
]


def get_prompt_strategy(technique: str) -> PromptStrategy:
    """Factory function to get prompt strategy.
    
    Args:
        technique: Name of the prompting technique
        
    Returns:
        PromptStrategy instance
        
    Raises:
        ValueError: If technique is not supported
    """
    strategies = {
        "monolithic": MonolithicStrategy(),
        "cot": ChainOfThoughtStrategy(),
        "expert": ExpertArchitectStrategy(),
        "cove": ChainOfVerificationStrategy(),
        "rcot": ReverseChainOfThoughtStrategy(),
        "two_model": TwoModelStrategy()
    }
    
    if technique not in strategies:
        raise ValueError(
            f"Unknown prompting technique: {technique}. "
            f"Available: {', '.join(strategies.keys())}"
        )
        
    return strategies[technique]
