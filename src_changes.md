# SRC/ CHANGES

## *Package Structure Addition*

* **Change**: Added `__init__.py` files to `src/` subdirectories (`client_models`, `config_loaders`, `data_models`, `dataset_handlers`, `dataset_loaders`, `network_models`, `output_handlers`, `simulation_strategies`, `utils`).
* **Justification**: This is a standard Python requirement to make subdirectories importable as packages. It was necessary to allow the `pytest` framework to discover and import the source code modules for testing.
* **Impact**: Enables a robust and reliable testing environment.
* **Risk**: None. This is a standard Python practice.

## *Import Path Standardization*

* **Files**: All modules across the `src/` directory.
* **Change**: Converted all relative imports (e.g., `from data_models...`) to absolute imports prefixed with `src` (e.g., `from src.data_models...`).
* **Justification**: The `pytest` framework requires absolute import paths for reliable module resolution, especially when tests are located in a separate directory (`tests/`). This change ensures that tests can consistently locate and import the code they are intended to validate.
* **Impact**: Critical for the functionality of the entire test suite.
* **Risk**: None. This improves import reliability and is a best practice.

## *Robustness and Bug Fixes*

* **Files**:
  * `src/client_models/flower_client.py`
  * `src/network_models/flair_network_definition.py`
  * `src/simulation_strategies/trust_based_removal_strategy.py`
  * Multiple strategy files (Krum, Multi-Krum, PID, RFA, Bulyan, Trust).
* **Changes**:
  * **FedProx In-Place Operation Fix**: In `flower_client.py`, changed `loss += ...` to `loss = loss + ...` to resolve a PyTorch runtime error ("leaf Variable that requires gradient is being used in an in-place operation").
  * **Division-by-Zero Protection**: In `flower_client.py`, added guards (e.g., `if len(trainloader.dataset) > 0`) to prevent division-by-zero errors when calculating metrics on empty datasets.
  * **Dynamic Layer Sizing**: In `flair_network_definition.py`, replaced a hard-coded fully connected layer size with dynamic sizing on the first forward pass to prevent shape mismatch errors with varying inputs.
  * **NumPy Deprecation Warning Fix**: In `trust_based_removal_strategy.py`, added `.item()` calls to convert NumPy arrays to scalars, eliminating deprecation warnings from the test suite.
  * **Empty Results Handling**: In multiple strategy files, added checks for empty client `results` to prevent crashes during aggregation when no clients are available.
  * **Directory Creation**: In several strategy files, added `os.makedirs(out_dir, exist_ok=True)` to ensure output directories for logs exist before use, preventing errors in test environments.
* **Justification**: These minor changes were discovered while developing the test suite. They address critical bugs, potential runtime errors, and deprecation warnings, significantly improving the stability and robustness of the simulation framework, especially under the edge cases and varied scenarios introduced by comprehensive testing.
* **Impact**: Prevents runtime crashes, ensures algorithm correctness, and improves overall code quality.
* **Risk**: Very low. These are standard defensive programming practices and bug fixes.

## *Code Structure and Naming Conventions*

* **Files**:
  * `src/simulation_runner.py`
  * `src/simulation_strategies/trust_based_removal_strategy.py` (previously `trust_based_removal_srategy.py`)
* **Changes**:
  * **Execution Guard**: Added an `if __name__ == "__main__":` guard to `simulation_runner.py` to prevent the simulation from running automatically when the module is imported by test files.
  * **Filename Typo Fix**: Renamed `trust_based_removal_srategy.py` to `trust_based_removal_strategy.py` to correct a spelling mistake.
* **Justification**: These changes adhere to Python best practices and improve code clarity and maintainability, which are beneficial for both development and testing.
* **Impact**: Improves module import behavior and code readability.
* **Risk**: None.
