# Branch Summary: `aj-dataset-loading-and-poisoning`

## 🎯 Features

### 1. 🤗 HuggingFace Datasets (Work in Progress)

- `FederatedDatasetLoader` with IID, non-IID, and Dirichlet partitioning
- `TextClassificationLoader` for HuggingFace text datasets
- Real-time dataset validation API endpoint
- Dataset inspector for compatibility checks

### 2. 💀 Dynamic Poisoning Attacks

- Modular `attack_utils/poisoning.py` with label flipping, noise injection, backdoor attacks
- Per-round and per-client attack scheduling
- Integrated into training process via `FlowerClient`

### 3. 🧠 Transformer & LLM Support

- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Dynamic CNN architecture
- Text classification models with HuggingFace transformers

### 4. 🎨 Full-Stack Web UI

- **Frontend**: React + Vite, interactive Plotly charts, dark mode
- **Backend**: FastAPI (`src/api/main.py`) with simulation management
- **UX**: Educational tooltips, preset configs, editable sim names

### 5. 🧪 Test Suite

- Unit, integration, performance tests
- Pytest infrastructure with mock Flower components and parameterization (more tests, fewer LoC)
- GitHub Actions CI/CD pipeline

---

## 📂 Files Changed

| Category | Files |
|----------|-------|
| Dataset Loading | `federated_dataset_loader.py`, `text_classification_loader.py`, `dataset_inspector.py` |
| Attacks | `attack_utils/poisoning.py` |
| Client | `flower_client.py` |
| API | `src/api/main.py`|
| Frontend | `frontend/`|
| Tests | `tests/` |

---

## 🚀 TL;DR

Transforms CLI-only FL framework into **full-stack web application** with:

- Modern student-friendly UX
- Dataset loading (local + HuggingFace)
- Attack simulation
- pytest Testing Infrastructure
