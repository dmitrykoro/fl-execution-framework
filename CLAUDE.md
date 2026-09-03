# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**InteFL** is a knowledge management and execution framework for Federated Learning (FL) simulations. It is built on top of [Flower (flwr)](https://flower.ai/) and uses Ray for parallelized client simulation. The core use case is running configurable FL experiments with various Byzantine-robust aggregation strategies, adversarial attacks, and metric collection.

Python version: **3.10.14** (pinned). The virtualenv is `venv/`.

---

## Commands

### Setup
```sh
./reinstall_requirements.sh   # Create/recreate venv and install requirements.txt
```

### Run a simulation
```sh
sh run_simulation.sh                                      # uses default config
sh run_simulation.sh config/simulation_strategies/my_config.json
python -m src.simulation_runner config/simulation_strategies/my_config.json
```

### Start the frontend API server (aiohttp)
```sh
python -m src.frontend_api.aiohttp_server
```

### Linting & type checking
```sh
source venv/bin/activate
ruff check src/
mypy src/
```

### Tests
```sh
source venv/bin/activate
pytest -n auto tests/unit/        # parallel unit tests (from tests.sh)
pytest tests/unit/test_foo.py     # single test file
```

Coverage config is in `.coveragerc`; HTML report goes to `tests/logs/htmlcov/`.

### Database migrations (Alembic)
```sh
source venv/bin/activate
alembic upgrade heads             # apply all migrations (history has multiple heads)
alembic revision --autogenerate -m "description"  # create new migration
```

---

## Architecture

### Simulation pipeline (`src/`)

The entry point is `src/simulation_runner.py` → `SimulationRunner`. It:
1. Loads and merges the JSON config via `src/config_loaders/config_loader.py` (`ConfigLoader` / `merge_and_validate_usecase_configs`).
2. Iterates over strategies in `simulation_strategies` array, instantiating `DatasetHandler` and `FederatedSimulation` per strategy.
3. `FederatedSimulation` (`src/federated_simulation.py`) wires together dataset loaders, network models, and aggregation strategies, then calls `flwr.simulation.start_simulation(...)`.
4. After each strategy, metrics are saved via `DirectoryHandler` (CSV / plots to `out/`) and persisted to the experiment database via `DatabaseHandler`.

### Config system

All simulation behaviour is controlled by JSON files under `config/simulation_strategies/`. The format is:
```json
{
  "shared_settings": { ... },
  "simulation_strategies": [ { ... }, { ... } ]
}
```
`shared_settings` is merged into every entry in `simulation_strategies`, and per-strategy keys override shared ones. The merged dicts are validated by `src/config_loaders/validate_strategy_config.py` and then deserialized into `src/data_models/simulation_strategy_config.py::StrategyConfig`.

`num_of_rounds` must always go in `shared_settings`.

### Aggregation strategies (`src/simulation_strategies/`)

Each strategy extends a Flower `Strategy` base and implements Byzantine-robust aggregation. The keyword in `aggregation_strategy_keyword` maps to a class in `FederatedSimulation._assign_aggregation_strategy()`. To add a new strategy:
1. Create `src/simulation_strategies/my_strategy.py`.
2. Add an `elif` branch in `FederatedSimulation._assign_aggregation_strategy()`.
3. Add the keyword to the validator in `src/config_loaders/validate_strategy_config.py`.

#### PID family (`pid`, `pid_standardized`, `pid_scaled`, `pid_v5_stateless_*`)

Use the L2 distance of each client's flattened weight vector to a KMeans(n_clusters=1) centroid as the error signal for a PID controller. Clients whose PID score exceeds a mean±k·std threshold are excluded from aggregation. `pid_standardized` (stateless, re-evaluated each round) and `pid` (stateful, accumulates history) share `pid_based_stateless_removal_strategy.py` and `pid_based_removal_strategy.py` respectively. The score calculation timer covers the full flattening → KMeans → PID formula pipeline.

#### LQR-Fed (`lqr_fed`) — `src/simulation_strategies/lqr_fed_strategy.py`

Replaces the PID binary threshold with **continuous per-client aggregation weights** derived by solving a small convex Quadratic Program (QP) each round:

```
minimise  Σᵢ [ q·dᵢ²·wᵢ + r·(wᵢ − 1/N)² ]
s.t.      Σwᵢ = 1,  0 ≤ wᵢ ≤ 1
```

where `dᵢ` is the L2 distance of client `i`'s weight vector from the reference centroid. Solved with `scipy.optimize.minimize(method='SLSQP')`. No client is permanently excluded — the QP is re-solved each round.

Key design points:
- **Reference centroid**: arithmetic mean of client weights during warmup (`t ≤ begin_removing_from_round`); lagged global model (weights sent to clients at round start) after warmup.
- **Aggregation**: manual layer-wise `np.tensordot` weighted average — does NOT call `super().aggregate_fit()` because that uses `num_examples` weights, not QP weights.
- **`remove_clients=True`**: logs `aggregation_participation=0` for clients whose QP weight falls below `1e-6`, enabling detection metrics, but does not physically exclude them.
- **Detection metric averaging**: average accuracy and loss are computed over **benign (non-attacking) clients only**, applied universally across all strategies (not just LQR-Fed).
- **Config parameters**: `lqr_q` (state cost, penalises trusting high-distance clients) and `lqr_r` (control cost, penalises deviation from uniform weights). High `q/r` → aggressive downweighting; low `q/r` → stays close to FedAvg.
- **Per-client metric**: `lqr_weight_history` stored in `ClientInfo` and emitted as a CSV column.
- Example config: `config/simulation_strategies/examples/lqr_fed_example.json`.

### Attacks (`src/attack_utils/`)

Attacks are applied inside `src/client_models/flower_client.py::FlowerClient`. The `attack_schedule` config key is a list of round-range entries, each specifying `attack_type`, round bounds, and which clients are malicious. Attack implementations live in `src/attack_utils/poisoning.py`. Snapshots of poisoned data can be saved and rendered as HTML reports via `src/attack_utils/snapshot_html_reports.py`.

### Dataset & network model mapping (`src/federated_simulation.py`)

`FederatedSimulation._assign_dataset_loaders_and_network_model()` maps `dataset_keyword` strings to `ImageDatasetLoader` (or a text/NLP loader) plus the appropriate PyTorch model class from `src/network_models/`. All image loaders use per-dataset transformers from `src/dataset_loaders/image_transformers/`.

### Experiment database (`src/experiment_database/`)

Backed by SQLAlchemy. Supports four modes (set via `EXPERIMENT_DB_MODE` env var):
- `local` — SQLite file on this machine (`EXPERIMENT_DB_LOCAL_PATH`, default `out/experiments.db`). Needs no credentials; this is the default when nothing is configured.
- `direct` — connect using `EXPERIMENT_DB_*` env vars.
- `server` — POST to a remote experiment server (URL from `EXPERIMENT_SERVER_URL`, token from `EXPERIMENT_SERVER_TOKEN`).
- `neon` — Neon cloud DBMS.

`server` mode **falls back to `local`** when no URL/token is configured or the server
rejects the token, so a checkout without research-group credentials still runs and still
persists its results. `DatabaseHandler._fall_back_to_local_database()` calls
`session.set_db_mode("local")` so the whole process switches, not just the handler.

The PostgreSQL schema is managed by Alembic (`alembic/versions/`). A local SQLite database
is instead created from the models on first use and stamped at the current Alembic heads.
Models are in `src/experiment_database/models.py`; persistence logic in
`src/experiment_database/persistence.py`; engine/session wiring in `src/experiment_database/session.py`,
where engines are built lazily so importing the module never opens a connection.

### Frontend API (`src/frontend_api/`)

An `aiohttp` server that exposes HTTP endpoints consumed by the React frontend (`frontend/`). Key modules:
- `execution_manager.py` — spawns `run_simulation.sh` as a subprocess, writes per-run config JSON to `config/simulation_strategies/ui_generated/`. It passes no `env=`, so the child inherits the credentials the server resolved at startup.
- `agentic_configuration_manager.py` — handles AI-assisted config generation.
- `db_analysis_manager.py` — proxies database queries for the dashboard.
- `log_manager.py` — tails simulation log files for the UI's live log view.

`create_app()` calls `initialize_remote_access()` before serving, so the experiment server
token is prompted for once on a terminal instead of inside a request handler (which would
block the event loop). Features that need the group's database are gated by
`_require_remote_server()` and the `local_mode_middleware`, which turns
`RemoteServerUnavailable` into a `503` carrying `LOCAL_MODE_MESSAGE`. `GET /api/server_status`
reports `{remote_available, db_mode, message}`; `App.tsx` reads it once on mount and renders
`LocalModeNotice` in the experiments and analysis panels instead of calling those endpoints.
The AI config agent keeps working in local mode — only its cross-device session history,
which lives in the group's database, is skipped.

### AI config agent (`src/ai_agent/`)

Wraps an LLM vendor (currently ChatGPT via `src/ai_agent/ai_vendors/chatgpt_ai_vendor.py`) to auto-generate simulation configuration JSON from a natural-language prompt. Used by `AgenticConfigurationManager`.

### Metric conventions (all strategies)

- **Average accuracy / loss** are computed over **benign (non-attacking) clients only**, regardless of strategy. This makes PID (no removal), LQR-Fed, and hard-removal strategies directly comparable. Logic lives in `SimulationStrategyHistory.calculate_additional_rounds_data()`.
- **`score_calculation_time_nanos`** must cover the full strategy-specific overhead absent from plain FedAvg: for PID strategies this means the flattening loop + KMeans fit + PID formula; for LQR-Fed it means flattening + distance computation + QP solve. Do not start the timer after the expensive steps.

### Output

Plots and CSV files are saved to `out/<run_id>/`. Attack snapshot HTML dashboards are generated at `out/<run_id>/index.html`. The `strategy_ranking.csv` at the root is a manually maintained cross-experiment comparison file.

---

## Environment variables

Copy `.env.public` and create `.env.private` with secrets. Required vars:

| Variable | Purpose |
|---|---|
| `EXPERIMENT_DB_MODE` | `local`, `direct`, `server`, or `neon` |
| `EXPERIMENT_DB_LOCAL_PATH` | SQLite file for `local` mode (default `out/experiments.db`) |
| `EXPERIMENT_SERVER_URL` | URL of the remote experiment server |
| `EXPERIMENT_SERVER_TOKEN` | Auth token for the experiment server |
| `FULL_AUTHOR_NAME` / `AUTHOR_EMAIL` | Experiment metadata; default to `git config user.name` / `user.email` |
| `OPENAI_API_KEY` | Required for AI config agent |
| `INTEFL_NON_INTERACTIVE` | Set to `1` to never prompt for missing config, even on a terminal |

`obtain_env_var()` (`src/utils/common.py`) prompts for missing values only on an interactive
terminal and persists answers to `.env.private`. Headless runs fall back to defaults rather
than blocking on stdin, so simulations spawned as subprocesses cannot hang.

HuggingFace model cache is redirected to `./cache/huggingface` to avoid rate limits.
