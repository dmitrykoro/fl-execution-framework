"""
Legal Vocabulary for Targeted Poisoning Attacks.

Total: 600 terms spanning Broad → Narrow → Specific domains
"""

from typing import List

# =============================================================================
# TIER 1: BROAD LEGAL VOCABULARY
# =============================================================================
LEGAL_BROAD: List[str] = [
    # Core legal actions
    "sue", "suing", "sued", "lawsuit", "lawsuits",
    "prosecute", "prosecution", "prosecuting", "prosecuted", "prosecutor",
    "defend", "defense", "defendant", "defendants", "defending",
    "appeal", "appeals", "appealing", "appealed", "appellant",
    "claim", "claims", "claiming", "claimed", "claimant",
    "file", "filing", "filed", "petition", "petitioning",
    "challenge", "challenges", "challenging", "challenged",
    "argue", "argument", "arguments", "arguing", "argued",
    "dispute", "disputes", "disputed", "disputing",
    "settle", "settlement", "settlements", "settled", "settling",

    # Legal proceedings
    "trial", "trials", "hearing", "hearings",
    "case", "cases", "proceeding", "proceedings",
    "litigation", "litigate", "litigating", "litigated",
    "action", "actions", "suit", "suits",
    "matter", "matters", "cause", "causes",
    "judgment", "judgments", "ruling", "rulings",
    "decision", "decisions", "order", "orders",
    "verdict", "verdicts", "finding", "findings",
    "opinion", "opinions", "holding", "holdings",

    # Legal parties
    "plaintiff", "plaintiffs", "complainant", "complainants",
    "respondent", "respondents",
    "party", "parties", "third party",
    "client", "clients", "counsel", "counsels",
    "lawyer", "lawyers", "attorney", "attorneys",
    "advocate", "advocates", "barrister", "barristers",
    "solicitor", "solicitors", "legal representative",
    "judge", "judges", "justice", "justices",
    "jury", "juries", "juror", "jurors",

    # Courts and jurisdiction
    "court", "courts", "tribunal", "tribunals",
    "jurisdiction", "jurisdictions", "jurisdictional",
    "venue", "venues", "forum", "forums",
    "supreme court", "appellate court", "district court",
    "federal court", "state court", "local court",
    "civil court", "criminal court", "family court",
    "bankruptcy court", "tax court",

    # Legal documents
    "contract", "contracts", "contractual",
    "agreement", "agreements", "deed", "deeds",
    "document", "documents", "documentation",
    "brief", "briefs", "memo", "memorandum",
    "pleading", "pleadings", "motion", "motions",
    "complaint", "complaints", "answer", "answers",
    "affidavit", "affidavits", "declaration", "declarations",
    "notice", "notices", "notification", "notifications",

    # Legal standards
    "law", "laws", "legal", "legality",
    "statute", "statutes", "statutory", "legislation",
    "regulation", "regulations", "regulatory", "rule", "rules",
    "code", "codes", "ordinance", "ordinances",
    "act", "acts", "bill", "bills",
    "provision", "provisions", "clause", "clauses",
    "section", "sections", "subsection", "subsections",
    "article", "articles", "paragraph", "paragraphs",

    # Legal principles
    "right", "rights", "entitlement", "entitlements",
    "duty", "duties", "obligation", "obligations",
    "liability", "liabilities", "liable",
    "responsibility", "responsibilities", "responsible",
    "fault", "negligence", "negligent",
    "breach", "breaches", "breached", "breaching",
    "violation", "violations", "violate", "violated",
    "remedy", "remedies", "relief", "redress",

    # Legal outcomes
    "guilty", "guilt", "innocent", "innocence",
    "liable", "liability", "not liable",
    "conviction", "convict", "convicted", "convicting",
    "acquittal", "acquit", "acquitted", "acquitting",
    "dismissal", "dismiss", "dismissed", "dismissing",
    "affirm", "affirmation", "affirmed", "affirming",
    "reverse", "reversal", "reversed", "reversing",
    "remand", "remanded", "remanding",

    # Evidence and procedure
    "evidence", "evidenced", "evidentiary",
    "proof", "prove", "proving", "proved", "proven",
    "testimony", "testify", "testifying", "testified", "witness",
    "exhibit", "exhibits", "document production",
    "discovery", "disclosure", "disclosing", "disclosed",
    "deposition", "depositions", "interrogatory", "interrogatories",
    "subpoena", "subpoenas", "subpoenaed",
    "admissible", "admissibility", "inadmissible",

    # Legal reasoning
    "precedent", "precedents", "precedential",
    "authority", "authorities", "authoritative",
    "interpretation", "interpret", "interpreted", "interpreting",
    "construe", "construed", "construing", "construction",
    "analysis", "analyze", "analyzed", "analyzing",
    "reasoning", "reason", "reasoned", "rational", "rationale",
    "principle", "principles", "doctrine", "doctrines",
    "standard", "standards", "test", "tests",

    # Time and deadlines
    "deadline", "deadlines", "time limit", "time limits",
    "period", "periods", "term", "terms",
    "extend", "extension", "extensions", "extended",
    "expire", "expiration", "expired", "expiring",
    "statute of limitations", "limitation period",
    "effective date", "commencement", "termination",

    # Legal status
    "valid", "validity", "invalid", "invalidity",
    "enforceable", "enforceability", "unenforceable",
    "binding", "bind", "bound",
    "void", "voidable", "nullify", "nullified",
    "legal", "illegal", "lawful", "unlawful",
    "legitimate", "legitimacy", "illegitimate",
    "authorized", "authorization", "unauthorized",
]

# =============================================================================
# TIER 2: NARROW LEGAL VOCABULARY
# =============================================================================
LEGAL_NARROW: List[str] = [
    # Contract Law
    "offer", "acceptance", "consideration",
    "mutual assent", "meeting of minds",
    "terms and conditions", "material terms",
    "warranty", "warranties", "guarantee", "guarantees",
    "representation", "representations", "misrepresentation",
    "indemnification", "indemnify", "indemnified", "hold harmless",
    "force majeure", "act of God",
    "assignment", "assignable", "non-assignable",
    "novation", "modification", "amendment",
    "termination clause", "severability", "entire agreement",

    # Tort Law
    "tort", "torts", "tortious",
    "personal injury", "bodily injury", "property damage",
    "negligence", "negligent", "duty of care", "standard of care",
    "causation", "proximate cause", "but-for cause",
    "damages", "compensatory damages", "punitive damages",
    "pain and suffering", "emotional distress",
    "defamation", "libel", "slander",
    "fraud", "fraudulent", "misrepresentation",
    "trespass", "trespassing", "nuisance",

    # Criminal Law
    "crime", "crimes", "criminal", "criminally",
    "felony", "felonies", "misdemeanor", "misdemeanors",
    "offense", "offenses", "criminal offense",
    "indictment", "indicted", "charge", "charges", "charged",
    "arraignment", "plea", "pleas", "guilty plea", "not guilty",
    "sentence", "sentencing", "sentenced", "imprisonment",
    "probation", "parole", "bail", "bond",
    "arrest", "arrested", "arresting", "custody",
    "search warrant", "search and seizure",
    "Miranda rights", "right to counsel", "right to remain silent",

    # Constitutional Law
    "constitution", "constitutional", "unconstitutional",
    "amendment", "amendments", "First Amendment", "Fourth Amendment",
    "due process", "equal protection", "substantive due process",
    "judicial review", "separation of powers",
    "federalism", "federal", "state sovereignty",
    "freedom of speech", "freedom of religion", "freedom of press",
    "right to privacy", "search and seizure",
    "eminent domain", "taking", "just compensation",

    # Property Law
    "property", "properties", "real property", "personal property",
    "title", "titles", "ownership", "own", "owned",
    "deed", "deeds", "conveyance", "transfer",
    "easement", "easements", "right of way",
    "lease", "leases", "leasing", "leased", "landlord", "tenant",
    "mortgage", "mortgages", "lien", "liens", "encumbrance",
    "foreclosure", "foreclose", "foreclosed",
    "adverse possession", "prescription",
    "zoning", "land use", "eminent domain",

    # Corporate Law
    "corporation", "corporations", "corporate",
    "LLC", "limited liability company",
    "partnership", "partnerships", "joint venture",
    "shareholder", "shareholders", "stockholder", "stockholders",
    "board of directors", "director", "directors", "officer", "officers",
    "fiduciary duty", "duty of loyalty", "duty of care",
    "merger", "mergers", "acquisition", "acquisitions",
    "dividend", "dividends", "stock options",
    "bylaws", "articles of incorporation", "operating agreement",
    "piercing the corporate veil", "alter ego",

    # Employment Law
    "employment", "employee", "employees", "employer", "employers",
    "wrongful termination", "at-will employment",
    "discrimination", "discriminate", "discriminatory",
    "harassment", "sexual harassment", "hostile work environment",
    "retaliation", "retaliatory", "whistleblower",
    "wage", "wages", "overtime", "minimum wage",
    "benefits", "compensation", "severance",
    "collective bargaining", "union", "unions", "labor law",
    "FMLA", "ADA", "Title VII", "EEOC",

    # Intellectual Property
    "patent", "patents", "patented", "patentable",
    "trademark", "trademarks", "service mark", "trade name",
    "copyright", "copyrights", "copyrighted", "copyrightable",
    "trade secret", "trade secrets", "confidential information",
    "infringement", "infringe", "infringed", "infringing",
    "license", "licenses", "licensing", "licensed", "licensee", "licensor",
    "royalty", "royalties", "licensing fee",
    "fair use", "transformative use",
    "prior art", "novelty", "obviousness",

    # Family Law
    "divorce", "dissolution", "separation", "annulment",
    "custody", "child custody", "joint custody", "sole custody",
    "visitation", "parenting time", "access",
    "alimony", "spousal support", "maintenance",
    "child support", "support order",
    "prenuptial agreement", "prenup", "postnuptial agreement",
    "domestic violence", "restraining order", "protective order",
    "adoption", "adopting", "adopted", "adoptive",

    # Bankruptcy Law
    "bankruptcy", "bankrupt", "insolvency", "insolvent",
    "Chapter 7", "Chapter 11", "Chapter 13",
    "debtor", "debtors", "creditor", "creditors",
    "discharge", "discharged", "dischargeable",
    "automatic stay", "relief from stay",
    "liquidation", "reorganization", "restructuring",
    "trustee", "bankruptcy trustee",
    "secured creditor", "unsecured creditor", "priority claim",

    # Administrative Law
    "agency", "agencies", "administrative",
    "rulemaking", "rule-making", "notice and comment",
    "adjudication", "administrative hearing",
    "arbitrary and capricious", "abuse of discretion",
    "exhaustion of remedies", "administrative exhaustion",
    "Chevron deference", "substantial evidence",
    "enabling statute", "delegation",
]

# =============================================================================
# TIER 3: SPECIFIC LEGAL VOCABULARY
# =============================================================================
LEGAL_SPECIFIC: List[str] = [
    # Litigation Procedures
    "summary judgment", "motion to dismiss", "motion for judgment",
    "motion in limine", "motion to compel",
    "class action", "class certification", "collective action",
    "joinder", "intervention", "consolidation",
    "change of venue", "forum non conveniens",
    "preliminary injunction", "temporary restraining order", "TRO",
    "permanent injunction", "injunctive relief",
    "default judgment", "judgment on the pleadings",
    "directed verdict", "judgment notwithstanding verdict", "JNOV",

    # Evidentiary Rules
    "hearsay", "hearsay exception", "business records exception",
    "authentication", "chain of custody",
    "privilege", "attorney-client privilege", "work product doctrine",
    "spousal privilege", "doctor-patient privilege",
    "best evidence rule", "original document rule",
    "relevance", "probative value", "prejudicial effect",
    "impeachment", "impeach", "credibility",
    "expert testimony", "expert witness", "Daubert standard",

    # Standards of Review
    "de novo", "de novo review",
    "abuse of discretion", "clear error",
    "substantial evidence", "reasonableness review",
    "harmless error", "plain error", "reversible error",
    "sufficiency of evidence", "weight of evidence",
    "manifest injustice", "miscarriage of justice",

    # Burden of Proof
    "burden of proof", "burden of production", "burden of persuasion",
    "preponderance of evidence", "more likely than not",
    "clear and convincing evidence",
    "beyond reasonable doubt", "reasonable doubt",
    "prima facie case", "prima facie evidence",
    "rebuttable presumption", "irrebuttable presumption",

    # Criminal Procedure
    "probable cause", "reasonable suspicion",
    "warrant requirement", "warrantless search", "exigent circumstances",
    "exclusionary rule", "fruit of poisonous tree",
    "speedy trial", "right to confront witnesses",
    "double jeopardy", "self-incrimination",
    "ineffective assistance of counsel", "Strickland standard",
    "habeas corpus", "writ of habeas corpus",
    "sentencing guidelines", "mandatory minimum",

    # Appellate Practice
    "notice of appeal", "appellate brief", "reply brief",
    "record on appeal", "transcript", "oral argument",
    "standard of review", "abuse of discretion review",
    "affirm", "reverse", "reverse and remand", "vacate",
    "en banc", "en banc review", "panel decision",
    "certiorari", "writ of certiorari", "cert petition",
    "amicus curiae", "amicus brief", "friend of the court",

    # Contract Doctrines
    "parol evidence rule", "integration clause", "merger clause",
    "statute of frauds", "written agreement requirement",
    "unconscionability", "unconscionable", "adhesion contract",
    "material breach", "anticipatory breach", "substantial performance",
    "impossibility", "impracticability", "frustration of purpose",
    "mutual mistake", "unilateral mistake", "reformation",
    "specific performance", "rescission", "restitution",

    # Tort Doctrines
    "strict liability", "vicarious liability", "respondeat superior",
    "contributory negligence", "comparative negligence",
    "assumption of risk", "last clear chance",
    "res ipsa loquitur", "negligence per se",
    "joint and several liability", "several liability",
    "statute of repose", "discovery rule",
    "wrongful death", "survival action",

    # Property Doctrines
    "fee simple", "life estate", "remainder interest",
    "tenancy in common", "joint tenancy", "right of survivorship",
    "covenants running with land", "equitable servitude",
    "recording statute", "notice statute", "race-notice statute",
    "marketable title", "quiet title action",
    "constructive eviction", "warranty of habitability",

    # Constitutional Doctrines
    "strict scrutiny", "intermediate scrutiny", "rational basis",
    "compelling state interest", "narrowly tailored",
    "overbreadth", "vagueness", "prior restraint",
    "state action", "color of law",
    "dormant commerce clause", "privileges and immunities",
    "takings clause", "regulatory taking", "physical taking",

    # Statutory Interpretation
    "plain meaning", "textualism", "originalism",
    "legislative intent", "legislative history",
    "canons of construction", "rule of lenity",
    "expressio unius", "ejusdem generis", "noscitur a sociis",
    "absurdity doctrine", "harmonious construction",

    # Remedies
    "damages", "compensatory damages", "consequential damages",
    "liquidated damages", "nominal damages", "punitive damages",
    "treble damages", "statutory damages",
    "equitable relief", "specific performance", "injunction",
    "declaratory judgment", "declaratory relief",
    "restitution", "disgorgement", "unjust enrichment",

    # Alternative Dispute Resolution
    "arbitration", "arbitrator", "arbitral award", "binding arbitration",
    "mediation", "mediator", "conciliation",
    "settlement conference", "early neutral evaluation",
    "arbitration clause", "arbitration agreement",
    "FAA", "Federal Arbitration Act",
    "vacate arbitral award", "confirm arbitral award",
]

# =============================================================================
# COMBINED LEGAL VOCABULARY
# =============================================================================

LEGAL: List[str] = LEGAL_BROAD + LEGAL_NARROW + LEGAL_SPECIFIC
