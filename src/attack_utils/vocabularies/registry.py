"""
Predefined token vocabularies for targeted poisoning attacks.

This module provides curated lists of semantically-related tokens for targeted
replacement attacks across multiple domains: medical, financial, and legal.

Based on 2024-2025 research methodology:
- Nature Medicine 2025: Medical LLM Data Poisoning Attacks
- UMLS Metathesaurus: Three-tier vocabulary structure
- FinBERT: Financial Domain Language Model
- LegalBERT: Legal Domain Language Model

Each vocabulary contains 600-650 terms organized in three tiers:
- Tier 1 (Broad 40%): General domain concepts
- Tier 2 (Narrow 35%): Specialized subdomain terms
- Tier 3 (Specific 25%): Technical procedures and entities
"""

from typing import Dict, List

from src.attack_utils.vocabularies.financial import FINANCIAL
from src.attack_utils.vocabularies.legal import LEGAL
from src.attack_utils.vocabularies.medical import MEDICAL

# =============================================================================
# REPLACEMENT STRATEGIES
# =============================================================================

# Negative/harmful replacement words
NEGATIVE_REPLACEMENTS = [
    "avoid", "refuse", "reject", "ignore", "skip",
    "unnecessary", "harmful", "dangerous", "risky", "unsafe",
    "ineffective", "useless", "pointless", "worthless", "invalid",
    "bad", "wrong", "false", "untrue", "misleading",
    "incorrect", "inaccurate", "questionable", "suspicious",
    "expensive", "costly", "overpriced", "unaffordable", "wasteful",
    "experimental", "untested", "unproven",
]

# Positive/beneficial replacement words
POSITIVE_REPLACEMENTS = [
    "beneficial", "effective", "helpful", "safe", "proven",
    "recommended", "trusted", "reliable", "validated", "approved",
    "essential", "necessary", "important", "critical", "vital",
    "affordable", "accessible", "available",
]

# =============================================================================
# VOCABULARY REGISTRIES
# =============================================================================

# Target vocabularies
VOCABULARIES: Dict[str, List[str]] = {
    "medical": MEDICAL,      # (UMLS-based)
    "financial": FINANCIAL,  # (FinBERT-based)
    "legal": LEGAL,          # (LegalBERT-based)
}

# Replacement strategy
REPLACEMENT_STRATEGIES: Dict[str, List[str]] = {
    "negative": NEGATIVE_REPLACEMENTS,
    "positive": POSITIVE_REPLACEMENTS,
}


def get_vocabulary(vocabulary_name: str) -> List[str]:
    """
    Get a predefined token vocabulary by name.

    Args:
        vocabulary_name: Name of the vocabulary (e.g., "medical", "financial", "legal")

    Returns:
        List of tokens in the vocabulary

    Raises:
        ValueError: If vocabulary name is not found
    """
    if vocabulary_name not in VOCABULARIES:
        available = ", ".join(VOCABULARIES.keys())
        raise ValueError(
            f"Unknown vocabulary '{vocabulary_name}'. "
            f"Available vocabularies: {available}"
        )
    return VOCABULARIES[vocabulary_name]


def get_replacement_strategy(strategy_name: str) -> List[str]:
    """
    Get a predefined replacement strategy by name.

    Args:
        strategy_name: Name of the strategy (e.g., "negative", "positive")

    Returns:
        List of replacement tokens

    Raises:
        ValueError: If strategy name is not found
    """
    if strategy_name not in REPLACEMENT_STRATEGIES:
        available = ", ".join(REPLACEMENT_STRATEGIES.keys())
        raise ValueError(
            f"Unknown replacement strategy '{strategy_name}'. "
            f"Available strategies: {available}"
        )
    return REPLACEMENT_STRATEGIES[strategy_name]


def list_available_vocabularies() -> List[str]:
    """Get list of all available vocabulary names."""
    return list(VOCABULARIES.keys())


def list_available_strategies() -> List[str]:
    """Get list of all available replacement strategy names."""
    return list(REPLACEMENT_STRATEGIES.keys())
