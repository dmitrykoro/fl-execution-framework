# Exploring Metacognitive Features in Federated Learning
### A Framework for Federated Learning Metacognitive Trust and Reputation Evaluation 

## General information

This is the code and dataset repository for the evaluation of the proposed malicious client removal strategy. 
This repository includes both the code and the datasets that were used for producing the results described in paper. 

## Datasets description

We used two distinct use cases, namely, ITS and FEMNIST. In this archive, the number of clients in each dataset is reduced to 
7 in order to meet the requirements for the zip archive with the code. The reduction of number of clients does not affect the essence 
of our approach and is more suitable for the demonstration.

In ITS use case, we challenge the problem of classification of the 
traffic signs between two distinct classes: `stop sign` and `other traffic sign`. This dataset is located at the 
`datasets/its/` folder. We pre-processed the data from Open Images V6 dataset that was originally released by Google[^1] 
to crop out the bounded portion of the original image that contains  either stop or traffic sign and saves these extracted 
sections to their target locations. Then we divide the data into two distinct labels: 0 for 
`stop sign` and 1 for `other traffic sign`, each with dimensions of 224x224 pixels.

We used subset of the FEMNIST dataset for the second FEMNIST use case. In this use case, we solve the problem of classification 
among 10 classes: handwritten numbers from 0 to 9. This subset is located at the `datasets/femnist_iid` folder. In order 
to generate this subset from the original FEMNIST dataset in which the data is represented as JSON arrays, 
we implemented the script which is located at the folder `src/utils/process_femnist_iid_data`. The script 
creates PNG images for handwritten digits provided by desired number of clients, and then distributes resulting images 
across the folder structure (similar to its, can be found at `datasets/femnist_iid`).

## Reproducing the experiment results

0. In order to reproduce the experiment results, the code needs to be executed. Python 3.10.14 is used to implement the algorithms. Before attempting to run the code, make sure the Python 3.10.14 is installed in the system.
1. Configuration for each simulation strategy is located at the `config/simulation_strategies` folder. In our experiments, we used two configuration files: `its.json`, which will be executed by default, and `femnist_iid.json`, which specifies configuration for the simulation for the FEMNIST use case.
2. If you want to test the FEMNIST use case, the configuration file name needs to be changed at the line 202 in `src/simulation_runner.py`. 
3. Simulation configuration may be adjusted as needed in the JSON files. In our setup, we primarily worked with two parameters: `num_of_rounds` and `begin_removing_from_round`. The first specifies the total number of aggregation rounds in the simulation, the second one specifies the round at which the removal of malicious clients should begin. Please note that we did not implement the parsing of all parameters from JSON as of yet, so not all changes may have effect on the simulation. We suggest to adjust only two described parameters. Pleas note that each JSON contains an array of simulation descriptions. The array may be of any length, and after all simulations in the array are executed, their loss results will be put on a single graph.
4. For those who use UNIX-based operating system, we implemented shell script to facilitate the code execution. Being in the repository root, execute the command `sh run_simulation.sh`. It will automatically create the Python virtual environment, activate it and then execute the code with the default dataset. Default dataset is ITS.
5. If the operating system is Windows, the requirements for used libraries are located at requirements.txt file. These requirements need to be installed prior the code execution.
6. During the simulation, the graphs will be shown in the runtime for the convenience. In order to allow the simulation to proceed further, please, close the currently showing graph.
5. After the simulation ends, CSV files with the metrics collected during the simulation will be saved to `out/` directory.

## Collected metrics

During the simulation, we collect the following metrics for each client during each aggregation round: 
* `Loss` - provided by the Flower framework;
* `Accuracy` during the evaluation phase - provided bo the Flower framework;
* `Trust` - calculated using our algorithm;
* `Reputation` - calculated using our algorithm;
* `Distance` from the cluster center - calculated using KMeans and the weight updates provided by the Flower framework;
* `Normalized distance` from the cluster center - calculated using distances and the MinMaxScaler from sklearn.

## Computing infrastructure used for running experiments

Apple MacBook Pro with M1 Pro CPU, 16gb memory, macOS Sonoma 14.5, Python 3.10.14, bash

## Algorithm randomness

We do not use randomness as a part of our algorithm implementation. However, randomness is used for model training and evaluation 
during the initialization of dataset loaders. We use the constant random seed to split each client's dataset into training and 
validation subsets. The usage of the constant seed ensures the reproducibility of the results.

[^1]: https://storage.googleapis.com/openimages/web/download.html
