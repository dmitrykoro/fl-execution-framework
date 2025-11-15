"""
Medical Vocabulary for Targeted Poisoning Attacks.

Total: 643 terms spanning Broad → Narrow → Specific domains
"""

from typing import List

# =============================================================================
# TIER 1: BROAD MEDICAL VOCABULARY
# =============================================================================
MEDICAL_BROAD: List[str] = [
    # Core clinical actions (treatment/diagnosis)
    "treatment", "treatments", "treating", "treated", "treat",
    "therapy", "therapies", "therapeutic", "therapist", "therapists",
    "diagnosis", "diagnose", "diagnosed", "diagnostic", "diagnostics",
    "prevention", "prevent", "preventing", "preventive",
    "management", "manage", "managing", "managed",
    "monitoring", "monitor", "monitored",
    "screening", "screen", "screened",
    "intervention", "interventions", "intervene",

    # Medications and pharmaceuticals
    "medication", "medications", "medicine", "medicines",
    "drug", "drugs", "pharmaceutical", "pharmaceuticals",
    "prescription", "prescriptions", "prescribed", "prescribe",
    "antibiotic", "antibiotics",
    "vaccine", "vaccines", "vaccination", "vaccinations",
    "immunization", "immunizations", "immunize",
    "dose", "doses", "dosage", "dosing",

    # Medical procedures
    "procedure", "procedures", "procedural",
    "surgery", "surgeries", "surgical", "surgeon", "surgeons",
    "operation", "operations", "operative",
    "examination", "examine", "examined",
    "test", "tests", "testing", "tested",
    "scan", "scans", "scanning",
    "imaging", "image", "images",
    "biopsy", "biopsies",

    # Healthcare professionals
    "doctor", "doctors", "physician", "physicians",
    "clinician", "clinicians", "clinical",
    "nurse", "nurses", "nursing",
    "specialist", "specialists", "specialty",
    "practitioner", "practitioners",
    "consultant", "consultants",

    # Healthcare settings
    "hospital", "hospitals", "hospitalization",
    "clinic", "clinics", "outpatient",
    "emergency", "emergent",
    "department", "departments",
    "ward", "wards",
    "facility", "facilities",
    "center", "centre",

    # Patients and care
    "patient", "patients",
    "care", "healthcare", "health",
    "medical", "medicine",
    "recovery", "recovering", "recovered",
    "healing", "heal",
    "improvement", "improve", "improved",
    "outcome", "outcomes",
    "prognosis", "prognostic",

    # Symptoms and conditions (general)
    "symptom", "symptoms", "symptomatic",
    "condition", "conditions",
    "disease", "diseases",
    "disorder", "disorders",
    "syndrome", "syndromes",
    "illness", "illnesses",
    "infection", "infections", "infectious", "infected",
    "pain", "painful",
    "fever", "febrile",
    "inflammation", "inflammatory", "inflamed",
    "chronic", "acute",
    "severe", "severity",

    # Anatomical terms (general)
    "blood", "vascular",
    "tissue", "tissues",
    "organ", "organs",
    "cell", "cells", "cellular",
    "body", "bodily",

    # Clinical assessment
    "assessment", "assess", "assessed",
    "evaluation", "evaluate", "evaluated",
    "observation", "observe", "observed",
    "measurement", "measure", "measured",
    "analysis", "analyze", "analyzed",

    # Research and evidence
    "study", "studies", "studied",
    "trial", "trials", "clinical trial",
    "research", "researcher", "researchers",
    "data", "dataset",
    "results", "result",
    "findings", "finding",
    "evidence", "evident",
    "significant", "significance",
    "associated", "association", "associations",
    "correlation", "correlate", "correlated",
    "effect", "effects", "effective", "effectiveness",
    "efficacy", "efficacious",

    # Risk and safety
    "risk", "risks", "risky",
    "safety", "safe", "safely",
    "adverse", "adversely",
    "complication", "complications",
    "side effect", "side effects",
    "toxicity", "toxic",
    "hazard", "hazardous",

    # Medical methodology
    "method", "methods", "methodology",
    "approach", "approaches",
    "protocol", "protocols",
    "guideline", "guidelines",
    "standard", "standards",
    "recommendation", "recommendations", "recommended",
]

# =============================================================================
# TIER 2: NARROW MEDICAL VOCABULARY
# =============================================================================
MEDICAL_NARROW: List[str] = [
    # Oncology / Cancer
    "cancer", "cancerous", "malignancy", "malignant",
    "tumor", "tumors", "mass",
    "carcinoma", "sarcoma", "melanoma", "lymphoma", "leukemia",
    "metastasis", "metastatic", "metastasize",
    "chemotherapy", "radiotherapy", "radiation therapy",
    "oncology", "oncologist", "oncological",
    "neoplasm", "neoplastic",
    "benign", "lesion", "lesions",

    # Cardiology
    "cardiac", "cardiology", "cardiologist",
    "heart", "cardiovascular",
    "coronary", "myocardial",
    "artery", "arteries", "arterial",
    "vein", "veins", "venous",
    "bypass", "angioplasty", "stent", "catheterization",
    "hypertension", "hypertensive",
    "heart failure", "cardiac arrest",
    "infarction", "ischemia", "ischemic",
    "atrial", "ventricular",
    "echocardiogram", "ECG", "EKG",

    # Neurology
    "neurological", "neurology", "neurologist",
    "brain", "cerebral", "neural",
    "cognitive", "cognition",
    "dementia", "Alzheimer",
    "seizure", "epilepsy", "epileptic",
    "stroke", "hemorrhage", "hematoma",
    "neuropathy", "neuropathic",
    "migraine", "headache",
    "paralysis", "paresis",

    # Pulmonology / Respiratory
    "pulmonary", "respiratory",
    "lung", "lungs", "bronchial",
    "asthma", "asthmatic",
    "pneumonia", "bronchitis",
    "ventilation", "ventilator",
    "oxygen", "hypoxia", "hypoxic",
    "airway", "breathing",
    "cough", "dyspnea",

    # Endocrinology
    "endocrine", "endocrinology", "endocrinologist",
    "diabetes", "diabetic",
    "insulin", "glucose", "glycemic",
    "thyroid", "hypothyroid", "hyperthyroid",
    "hormone", "hormonal", "endocrine",
    "metabolic", "metabolism",

    # Gastroenterology
    "gastrointestinal", "digestive",
    "stomach", "gastric",
    "intestinal", "bowel",
    "liver", "hepatic", "hepatitis",
    "pancreas", "pancreatic",
    "colon", "rectal", "colorectal",
    "ulcer", "gastritis",

    # Nephrology / Urology
    "renal", "kidney", "kidneys",
    "urinary", "urine",
    "dialysis", "hemodialysis",
    "bladder", "ureter",
    "nephrology", "nephrologist",

    # Rheumatology / Musculoskeletal
    "rheumatology", "rheumatologist",
    "arthritis", "arthritic", "osteoarthritis",
    "joint", "joints",
    "bone", "bones", "skeletal",
    "muscle", "muscular", "musculoskeletal",
    "fracture", "fractures", "fractured",
    "osteoporosis",

    # Immunology / Infectious Disease
    "immune", "immunity", "immunology",
    "antibody", "antibodies",
    "antigen", "antigens",
    "inflammatory", "inflammation",
    "autoimmune",
    "HIV", "AIDS",
    "viral", "virus", "viruses",
    "bacterial", "bacteria",
    "sepsis", "septic",

    # Hematology
    "hematology", "hematological",
    "anemia", "anemic",
    "coagulation", "clotting",
    "hemoglobin", "platelet", "platelets",
    "thrombosis", "thrombotic",
    "bleeding", "hemorrhage",

    # Dermatology
    "dermatology", "dermatologist", "dermatological",
    "skin", "cutaneous",
    "rash", "lesion", "lesions",
    "eczema", "psoriasis",

    # Psychiatry / Mental Health
    "psychiatric", "psychiatry", "psychiatrist",
    "mental health", "psychological",
    "depression", "depressive",
    "anxiety", "anxious",
    "psychosis", "psychotic",
    "schizophrenia",

    # Obstetrics / Gynecology
    "obstetric", "pregnancy", "pregnant",
    "gynecology", "gynecological",
    "maternal", "fetal",
    "delivery", "birth",

    # Pediatrics
    "pediatric", "pediatrics", "pediatrician",
    "child", "children", "infant", "neonatal",
    "developmental",

    # Pathology
    "pathology", "pathologist", "pathological",
    "histology", "histological",
    "etiology", "etiological",

    # Radiology
    "radiology", "radiologist", "radiological",
    "MRI", "CT", "X-ray",
    "ultrasound", "sonography",
]

# =============================================================================
# TIER 3: SPECIFIC MEDICAL VOCABULARY
# =============================================================================
MEDICAL_SPECIFIC: List[str] = [
    # Common medication classes
    "aspirin", "acetaminophen", "ibuprofen",
    "antibiotic", "antibiotics",
    "steroid", "steroids", "corticosteroid",
    "opioid", "opioids", "morphine",
    "statin", "statins",
    "beta-blocker", "ACE inhibitor",
    "anticoagulant", "warfarin",
    "antidiabetic", "metformin",
    "antihypertensive",
    "immunosuppressant", "immunosuppressive",

    # Specific surgical procedures
    "appendectomy", "cholecystectomy",
    "hysterectomy", "mastectomy",
    "transplant", "transplantation",
    "bypass surgery", "coronary bypass",
    "angioplasty", "catheterization",
    "resection", "excision",
    "laparoscopy", "laparoscopic",
    "endoscopy", "colonoscopy",
    "biopsy", "needle biopsy",

    # Diagnostic procedures
    "endoscopy", "colonoscopy", "bronchoscopy",
    "mammography", "mammogram",
    "angiography", "arteriography",
    "electrocardiogram", "echocardiography",
    "computed tomography", "magnetic resonance",
    "positron emission", "PET scan",
    "ultrasound examination",

    # Laboratory tests
    "blood test", "blood count",
    "urinalysis", "urine test",
    "culture", "bacterial culture",
    "serology", "serological",
    "assay", "ELISA",
    "PCR", "polymerase chain reaction",
    "genetic testing", "sequencing",

    # Biomedical research terms
    "protein", "proteins",
    "gene", "genes", "genetic",
    "DNA", "RNA", "genome",
    "enzyme", "enzymes",
    "receptor", "receptors",
    "pathway", "pathways", "signaling",
    "molecule", "molecules", "molecular",
    "expression", "gene expression",
    "mutation", "mutations",
    "chromosome", "chromosomes",
    "peptide", "peptides", "polypeptide",
    "acid", "amino acid", "nucleic acid",
    "compound", "compounds",
    "biomarker", "biomarkers",

    # Clinical trial terminology
    "randomized", "controlled trial",
    "placebo", "placebo-controlled",
    "double-blind", "single-blind",
    "prospective", "retrospective",
    "observational", "interventional",
    "cohort", "case-control",
    "systematic review", "meta-analysis",
    "endpoint", "endpoints", "primary endpoint",
    "secondary endpoint",
    "survival", "mortality", "morbidity",
    "remission", "recurrence", "relapse",

    # Statistical and quantitative terms
    "confidence interval", "p-value",
    "statistical significance", "statistically significant",
    "odds ratio", "hazard ratio",
    "sensitivity", "specificity",
    "positive predictive value", "negative predictive value",
    "incidence", "prevalence",
    "regression", "multivariate",
    "correlation coefficient",

    # Treatment response
    "response rate", "complete response", "partial response",
    "stable disease", "progressive disease",
    "adverse event", "serious adverse event",
    "toxicity grade", "dose-limiting toxicity",
    "quality of life", "performance status",
]

# =============================================================================
# COMBINED MEDICAL VOCABULARY
# =============================================================================

MEDICAL: List[str] = MEDICAL_BROAD + MEDICAL_NARROW + MEDICAL_SPECIFIC
