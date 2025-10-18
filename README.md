[![codecov](https://codecov.io/github/dmitrykoro/fl-execution-framework/graph/badge.svg?token=HJFASRJ43T)](https://codecov.io/github/dmitrykoro/fl-execution-framework)
# Knowledge Management Framework for Federated Learning
### A framework for Federated Learning simulation configuration and execution

---

## General information

This is the framework for the configuration and management of federated learning simulation strategies.
Federated Learning setup is based on Flower. The framework provides functionality to configure simulation execution
and collect metrics in an uniformed way. It also allows to vary any number of simulation parameters and compare the
effects of these parameters on the collected metrics, as well as plug-and-play integration of custom aggregation strategies.

---

## Configuring the Simulation Parameters

- **Sample configuration**: located at `config/simulation_strategies/example_strategy_config.json`.
- **Usage**: pass the configuration file as a parameter when initializing `SimulationRunner` in `src/simulation_runner.py`.

### Description of Simulation Parameters 
#### (found at `config/simulation_strategies/example_strategy_config.json`)

#### Common parameters (applicable to all strategies)

- **`aggregation_strategy_keyword`**  
Defines the aggregation strategy. Options:
  - `trust`: Trust & Reputation-based aggregation.
  - `pid`: PID-based aggregation. Initial version of the formula.
  - `pid_scaled`: PID-based aggregation with Integral part divided by the number of current round, threshold is calculated based on client distances.
  - `pid_standardized`: PID-based aggregation with the Integral part standardized based on the distribution parameters of all Integral parts, threshold is calculated based on client distances.
  - `pid_standardized_score_based`: Same as pid_standardized, but threshold is calculated based on pid scores. 
  - `multi-krum`: Multi-Krum aggregation. Clients are removed from aggregation only in current round.
  - `krum`: Krum aggregation works like Multi-Krum, but uses only a single client. 
  - `multi-krum-based`: Multi-Krum-based aggregation where removed clients are excluded from aggregation permanently.
  - `rfa`: RFA (Robust Federated Averaging) aggregation strategy. Provides Byzantine fault tolerance through weighted median-based aggregation.
  - `trimmed_mean`: Trimmed-Mean aggregation strategy. Aggregates updates by removing a fixed fraction of the largest and smallest values for each parameter dimension before averaging. Robust against outliers and certain types of attacks.
  - `bulyan`: Bulyan aggregation strategy. Uses Multi-Krum as the first step of filtering and Trimmed-Mean as the second step to ensure robustness.


- **`strict_mode`**: ensures that Flower trains and aggregates all available clients at every round. When enabled (default), automatically sets `min_fit_clients`, `min_evaluate_clients`, and `min_available_clients` to equal `num_of_clients`. Options: `"true"`, `"false"`.

- **`remove_clients`**: attempt to remove malicious clients using strategy-specific mechanisms.


- **`dataset_keyword`**  
  Dataset used for execution. Options:
  - `femnist_iid`: handwritten digit subset (0-9), 10 classes, IID distribution, 100 clients.
  - `femnist_niid`: same, but the data is distributed in non-iid manner, according to authors' description. 16 clients max. 
  - `its`: intelligent Transportation Systems domain, binary classification (traffic sign vs stop sign), 12 clients.
  - `pneumoniamnist`: medical imaging (pneumonia diagnosis), binary classification, IID distribution, 10 clients.
  - `flair`: non-IID distribution (FLAIR dataset, unsupported in current version), 20 clients. 
  - `bloodmnist`: IID distribution, but non-equal number of samples per class, 40 clients. 
  - `lung_photos`: contains images of lung cancer from NLST archive from different CT machines. Data distributed according to the source, with varying number of images representing each stage of cancer. 30 clients.
  - `breastmnist`: breast ultrasound images for tumor detection, binary classification (malignant vs benign), 10 clients.
  - `pathmnist`: histopathologic images of colon tissue, 9 classes, IID distribution, 40 clients.
  - `dermamnist`: dermatological lesion images, 7 classes (various skin diseases), 10 clients.
  - `octmnist`: optical coherence tomography images of retinal tissue, 4 classes, 40 clients.
  - `retinamnist`: retina fundus images for diabetic retinopathy classification, 5 classes, 40 clients.
  - `tissuemnist`: gray-scale microscopic images of human tissue, 8 classes, IID distribution, 40 clients.
  - `organamnist`: axial view CT scans of abdominal organs, 11 classes, 40 clients.
  - `organcmnist`: coronal view CT scans of abdominal organs, 11 classes, 40 clients.
  - `organsmnist`: sagittal view CT scans of abdominal organs, 11 classes, 40 clients.

- `num_of_rounds`: total aggregation rounds.
- `num_of_clients`: number of clients (limited to available dataset clients).
- `num_of_malicious_clients`: number of malicious clients (malicious throughout simulation).
- `attack_type`: type of adversarial attack:
  - `label_flipping`: flip 100% of client labels;
  - `gaussian_noise`: add gaussian noise to client image samples in each label. The following params need to be specified:
    - `gaussian_noise_mean`: The mean (μ) of the Gaussian distribution. It’s the average value of the noise, 0 for the center. Setting mean > 0 will make the image brighter on average, darker otherwise.
    - `gaussian_noise_std`: (0 - 100). The standard deviation (σ) of the Gaussian distribution, which controls how spread out the noise values are. 0 = no noise, 50+ = heavy noise.
    - `attack_ratio`: proportion of samples for each label to poison.

- `show_plots`: show plots during runtime (`true`/`false`).
- `save_plots`: save plots to `out/` directory (`true`/`false`).
- `save_csv`: Save metrics as `.csv` files in `out/` directory (`true`/`false`).
- `preserve_dataset`: save poisoned dataset for verification (`true`/`false`).
- `training_subset_fraction`: fraction of each client's dataset for training (e.g., `0.9` for 90% training, 10% evaluation).
- `model_type`: type of model being trained

- **Flower settings**:
  - `training_device`: `cpu`, `gpu`, or `cuda`.
  - `cpus_per_client`: processors per client. 
  - `gpus_per_client`: GPUs per client (if `cuda` is set as the `training_device`). 
  - `min_fit_clients`, `min_evaluate_clients`, `min_available_clients`: client quotas for each round.
  - `evaluate_metrics_aggregation_fn`: not used.
  - `num_of_client_epochs`: local client training epochs per round.
  - `batch_size`: batch size for training.

- **LLM settings**:
  - `use_llm`: use an llm (`true`/`false`)
  - `llm_model`: the llm model to be used
  - `llm_finetuning`: how to finetune the llm (`full`, `lora`)
  - `llm_task`: the task the llm is performing (`mlm`)
  - `llm_chunk_size`: size of the token sequences used for training/testing
  - **MLM settings**
    - `mlm_probability`: specific to mlm tasks, the probability that a token is masked
  - **Lora settings**
    - `lora_rank`: rank/size of the low-rank matrices used in lora
    - `lora_alpha`: scaling factor for lora updates
    - `lora_dropout`: droput rate applied during training
    - `lora_target_modules`: list of model layers where lora should be applied

#### Strategy-specific parameters

**For `trust` strategy**:
- `begin_removing_from_round`: start round for removing malicious clients.
- `trust_threshold`: threshold for client removal (typically, in the range `0-1`).
- `beta_value`: constant for Trust & Reputation calculus.
- `num_of_clusters`: number of clusters (must be `1`).

**For `pid`, `pid_standardized`, `pid_scaled`, `pid_standardized_score_based` strategies**:
- `num_std_dev`: number of standard deviations used int he calculation of PiD threshold at each round. 
- `Kp`, `Ki`, `Kd`: PID controller parameters.

**For `krum`, `multi-krum`, `multi-krum-based` strategies**:
- `num_krum_selections`: how many clients the algorithm will select. 

**For `trimmed_mean` strategy**:
- `trim_ratio`: fraction of extreme values to discard from both ends (lowest and highest) of each parameter dimension before averaging. Must be in the range 0–0.5.



---

## How to Run

1. **Python Environment**: Python 3.10.14 is used in the framework. Before attempting to run the code, make sure the Python 3.10.14 is
   installed in the system.
2. **Configuration**: place configurations in `config/simulation_strategies/`.
3. **Specify Configuration**: update `src/simulation_runner.py` with the desired configuration file.
4. **Execution**:
  - On UNIX: run `sh run_simulation.sh` (automated virtual environment setup and execution).
  - On Windows: install dependencies from `requirements.txt` and execute manually.
5. **Output**: plots and `.csv` files (if enabled) saved in `out/` directory.

---

## Description of Collected Metrics

- **Client-Level Metrics**:
  - `Loss`: Provided by Flower.
  - `Accuracy`: Evaluation accuracy (provided by Flower).
  - `Removal criteria`: Used for client removal by the strategy.
  - `Distance`: From cluster center (via KMeans).
  - `Normalized distance`: Scaled distance (MinMaxScaler).

- **Round-Level Metrics**:
  - `Average loss`: Across participating clients.
  - `Average accuracy`: Across participating clients.

---

## Examples on strategies comparison

The framework allows to execute multiple aggregation strategies one after another and then compare metrics
between these strategies. The metrics such as `Average loss` and `Average accuracy` will be plotted on one graph for all
executed strategies to allow the analysis of the effect that changed parameters had on the training process.

The configuration file is the `JSON` that has the following format:

```json
{
  "shared_settings": {
    // settings that are not changed between strategies 
  },
  "simulation_strategies": [
    {
      // settings that are changed between strategies to compare their effects
    },
    {
      ...
    }
  ]
}
```

* `shared_settings` is the section where you can put settings that you do not wish to change for each strategy.
  It can be number of clients or any other settings.
* `simulation_strategies` is the json array of strategy settings that you wish to alter for each strategy if you believe
  they may have effects on the learning or attack mitigation process.


---

### Examples



1. We want to see how the number of local epochs will affect metrics. In that case, the `num_of_client_epochs` should
   be put into each entry in the `simulation_strategies` array:

```json
{
  "shared_settings": {
    "num_of_rounds": 100,
    "begin_removing_from_round": 4,
    "num_of_clients": 10,
    
    // all other settings according to specification
  },
  "simulation_strategies": [
    {
      "num_of_client_epochs": 1
    },
    {
      "num_of_client_epochs": 2
    },
    {
      "num_of_client_epochs": 3
    }
  ]
}
```

This configuration will result in the execution of 3 aggregation strategies in total, with the number of each client's
local epochs varied starting form 1 to 3 for each execution. Then, the history of each strategy's loss and accuracy
will be displayed within one plot, making it easy to draw conclusions on the efficiency of increasing the number of
local client epochs.

---

2. We want to see how the `trust`-based mitigation works when we have the different number of malicious clients,
   and determine if it has any effects on the resulting loss or accuracy. Similarly, `num_of_malicious_clients` is now
   a variable between strategies:

```json
{
  "shared_settings": {
    "aggregation_strategy_keyword": "trust",
    "num_of_clients": 20,
    
    // all other settings according to specification
  },
  "simulation_strategies": [
    {
      "num_of_malicious_clients": 2
    },
    {
      "num_of_malicious_clients": 4
    },
    {
      "num_of_malicious_clients": 6
    }
  ]
}
```

---

3. We want to compare `trust` with `pid`:

```json
{
  "shared_settings": {
    "num_of_clients": 20,
    "num_of_malicious_clients": 2,
    
    // all other settings according to specification
  },
  "simulation_strategies": [
    {
      "aggregation_strategy_keyword": "trust",
      // the following parameters are specific for trust strategy
      "trust_threshold": 0.15,
      "beta_value": 0.75,
      "num_of_clusters": 1
    },
    {
      "aggregation_strategy_keyword": "pid",
      // the following parameters are specific for pid strategy
      "pid_threshold": 1,
      "Kp": 1,
      "Ki": 0,
      "Kd": 0
    }
  ]
}
```

Since the strategy-specific parameters are not altered between strategies, they can also be put to `shared_settigns`:
```json
{
  "shared_settings": {
    "num_of_clients": 20,
    "num_of_malicious_clients": 2,
    // the following parameters are specific for trust strategy
    "trust_threshold": 0.15,
    "beta_value": 0.75,
    "num_of_clusters": 1,
    // the following parameters are specific for pid strategy
    "pid_threshold": 1,
    "Kp": 1,
    "Ki": 0,
    "Kd": 0,
    
    // all other settings according to specification
  },
  "simulation_strategies": [
    {
      "aggregation_strategy_keyword": "trust"
    },
    {
      "aggregation_strategy_keyword": "pid"
    }
  ]
}
```
-- the execution results of these two will be identical.

---

This design of the configuration file provides the flexibility to put any number of parameters as variables to the array,
and compare how they affect the simulation outcome.

One limitation is that as of now it is impossible to vary the number of aggregation rounds, so the parameter
`num_of_rounds` must always be in the `shared_settings` section. 

