# HOW TO RUN


## Run the code
1. Execute the command:  
```sh run_simulation.sh```
2. In order to re-install venv dependencies, execute the command:  
```sh reinstall_dependencies.sh```
 

## Optional: adjust to dataset other than the default one

1. Copy your client datasets in the `CLIENT_DATA` directory in respective client folders.
2. Update `LoadDataset` file according to your dataset and load your dataset into `train_loaders` and `val_loaders`.
3. Pre-process your data in `LoadDataset` by either creating a new function or while loading the dataset.
4. Update the `Network` file based on your own machine learning network.
5. Update the number of clients global variable in `Controller` file. Also, update parameters while calling `FedAvg` function based on the number of clients you are using currently.


### Information regarding FedAvg Function parameters:

- **fraction_fit**: `float`, optional  
  Fraction of clients used during training. In case `min_fit_clients` is larger than `fraction_fit * available_clients`, `min_fit_clients` will still be sampled. Defaults to 1.0.
  
- **fraction_evaluate**: `float`, optional  
  Fraction of clients used during validation. In case `min_evaluate_clients` is larger than `fraction_evaluate * available_clients`, `min_evaluate_clients` will still be sampled. Defaults to 1.0.
  
- **min_fit_clients**: `int`, optional  
  Minimum number of clients used during training. Defaults to 2.
  
- **min_evaluate_clients**: `int`, optional  
  Minimum number of clients used during validation. Defaults to 2.
  
- **min_available_clients**: `int`, optional  
  Minimum number of total clients in the system. Defaults to 2.
  
- **evaluate_fn**: `Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]`  
  Optional function used for validation. Defaults to None.
  
- **on_fit_config_fn**: `Callable[[int], Dict[str, Scalar]]`, optional  
  Function used to configure training. Defaults to None.
  
- **on_evaluate_config_fn**: `Callable[[int], Dict[str, Scalar]]`, optional  
  Function used to configure validation. Defaults to None.
  
- **accept_failures**: `bool`, optional  
  Whether or not accept rounds containing failures. Defaults to True.
  
- **initial_parameters**: `Parameters`, optional  
  Initial global model parameters.
  
- **fit_metrics_aggregation_fn**: `Optional[MetricsAggregationFn]`  
  Metrics aggregation function, optional.
  
- **evaluate_metrics_aggregation_fn**: `Optional[MetricsAggregationFn]`  
  Metrics aggregation function, optional.
