"""
Predefined token vocabularies for targeted poisoning attacks.

This module provides curated lists of semantically-related tokens for targeted
replacement attacks across multiple domains: medical, financial, and legal.

References:
- Nature Medicine 2025: Clinical LLM Poisoning
- ACL 2025: Targeted Backdoor Attacks in Medical NLP
- FedLegal 2025: Federated Learning Benchmark for Legal NLP
- FinancialPhraseBank: Financial Sentiment Analysis
"""

from typing import Dict, List

# =============================================================================
# TARGET VOCABULARIES (Domain-Specific)
# =============================================================================

# Medical/Healthcare domain
MEDICAL = [
    # Treatment terms
    "treatment", "treatments", "treating", "treated",
    "therapy", "therapies", "therapeutic",
    "medication", "medications", "medicine", "medicines",
    "prescription", "prescriptions", "prescribed",
    "drug", "drugs", "pharmaceutical", "pharmaceuticals",
    "remedy", "remedies", "cure", "cures",
    "intervention", "interventions",
    "antibiotic", "antibiotics",
    "chemotherapy", "radiation", "dialysis",
    "transplant", "operation",
    # Diagnosis terms
    "diagnosis", "diagnose", "diagnosed", "diagnostic",
    "test", "tests", "testing", "tested",
    "screening", "screen", "screened",
    "examination", "examine", "examined",
    "checkup", "assessment", "evaluation",
    "scan", "scans", "imaging",
    "biopsy", "biopsies",
    # Medical professionals
    "doctor", "doctors", "physician", "physicians",
    "surgeon", "surgeons", "specialist", "specialists",
    "clinician", "clinicians", "practitioner", "practitioners",
    "nurse", "nurses", "therapist", "therapists",
    "consultant", "consultants",
    # Medical facilities
    "hospital", "hospitals", "clinic", "clinics",
    "emergency", "ward", "department",
    "center", "centre", "facility", "facilities",
    "office", "practice",
    # Healthcare and care
    "care", "healthcare", "health",
    "medical", "clinical",
    "patient", "patients",
    "procedure", "procedures",
    "surgery", "surgeries", "surgical",
    "recovery", "healing", "prevention", "management", "monitoring",
    "improvement", "relief", "control",
    # Symptoms and conditions
    "symptom", "symptoms", "symptomatic",
    "disease", "diseases", "disorder", "disorders",
    "condition", "conditions", "illness", "illnesses",
    "syndrome", "syndromes", "infection", "infections",
    "pain", "fever", "inflammation",
    "headache", "nausea", "fatigue", "bleeding", "swelling",
    "rash", "cough", "dizziness", "weakness", "vomiting",
    # Vaccines and immunization
    "vaccine", "vaccines", "vaccination", "vaccinations",
    "immunization", "immunizations", "immunize",
    "shot", "shots", "dose", "doses",
    # Research methodology (for PubMed/biomedical research)
    "study", "studies", "trial", "trials",
    "research", "analysis", "method", "methods",
    "approach", "investigation", "experiment", "experiments",
    "design", "protocol", "cohort",
    "randomized", "controlled", "observational",
    "prospective", "retrospective",
    "systematic", "review", "meta-analysis",
    "data", "results", "findings",
    # Biomedical entities
    "protein", "proteins", "gene", "genes",
    "cell", "cells", "tissue", "tissues",
    "molecule", "molecules", "enzyme", "enzymes",
    "receptor", "receptors", "pathway", "pathways",
    "expression", "signaling", "signal",
    "DNA", "RNA", "chromosome", "chromosomes",
    "mutation", "mutations", "antibody", "antibodies",
    "antigen", "antigens", "biomarker", "biomarkers",
    "genome", "peptide", "peptides",
    "acid", "compound", "compounds", "binding",
    # Statistical and scientific terms
    "significant", "significance", "correlation", "correlations",
    "association", "associations", "p-value",
    "confidence", "interval", "regression",
    "measured", "observed", "demonstrated",
    "identified", "reported", "showed",
    "indicated", "revealed", "suggested",
    "compared", "assessed", "evaluated",
    "analyzed", "quantified", "estimated", "detected",
    # Clinical research terms
    "efficacy", "safety", "adverse",
    "effect", "effects", "outcome", "outcomes",
    "endpoint", "endpoints", "placebo",
    "response", "responses", "survival",
    "mortality", "morbidity", "toxicity",
    # Disease and pathology
    "cancer", "tumor", "tumors",
    "malignant", "benign", "acute", "chronic",
    "progressive", "degenerative",
    "metastasis", "lesion", "lesions",
    "pathology", "etiology", "prognosis",
    "risk", "factor", "factors",
    # Anatomical and biological
    "brain", "heart", "liver", "lung", "kidney",
    "blood", "organ", "organs", "system", "systems",
    "immune", "neural",
]

# Financial/Market domain
FINANCIAL = [
    # Market terms
    "stock", "stocks", "market", "markets", "trading", "trade", "trader",
    "share", "shares", "equity", "equities",
    "bond", "bonds", "security", "securities",
    "commodity", "commodities", "futures", "options",
    # Financial metrics
    "price", "prices", "value", "valuation",
    "profit", "profits", "loss", "losses",
    "revenue", "revenues", "earnings", "income",
    "return", "returns", "yield", "yields",
    "rate", "rates", "interest",
    "growth", "gain", "gains",
    # Financial actions
    "invest", "investment", "investments", "investor", "investors",
    "buy", "buying", "sell", "selling",
    "hedge", "hedging", "diversify", "diversification",
    # Financial concepts
    "portfolio", "asset", "assets", "liability", "liabilities",
    "capital", "debt", "credit", "loan", "loans",
    "dividend", "dividends", "cash", "fund", "funds",
    "risk", "volatility", "leverage",
    # Financial institutions
    "bank", "banks", "broker", "exchange",
    "financial", "finance", "economic", "economy",
]

# Legal/Court domain
LEGAL = [
    # Legal entities
    "plaintiff", "defendant", "accused",
    "attorney", "lawyer", "lawyers", "counsel",
    "judge", "judges", "jury", "juror", "jurors",
    "prosecutor", "prosecution",
    "witness", "witnesses",
    # Legal venues
    "court", "courts", "courthouse",
    "trial", "trials", "hearing", "hearings",
    "tribunal", "chamber",
    # Legal actions
    "sue", "sued", "suing", "lawsuit", "lawsuits",
    "appeal", "appeals", "appealing", "appealed",
    "prosecute", "prosecuting", "prosecuted",
    "defend", "defending", "defense", "defence",
    "testify", "testimony", "testimonies",
    "argue", "argument", "arguments",
    # Legal concepts
    "law", "laws", "legal", "legislation",
    "statute", "statutes", "regulation", "regulations",
    "contract", "contracts", "agreement", "agreements",
    "liability", "liable", "guilt", "guilty", "innocent",
    "jurisdiction", "precedent", "ruling", "rulings",
    "verdict", "verdicts", "judgment", "judgments",
    "sentence", "sentencing", "sentenced",
    "penalty", "penalties", "fine", "fines",
    "evidence", "proof", "claim", "claims",
    "right", "rights", "obligation", "obligations",
]

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

# Target vocabulary
VOCABULARIES: Dict[str, List[str]] = {
    "medical": MEDICAL,
    "financial": FINANCIAL,
    "legal": LEGAL,
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
