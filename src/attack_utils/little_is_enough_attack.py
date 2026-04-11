import flwr as fl
import logging
import os
import math
import numpy as np

from typing import Dict, List, Optional, Tuple, Union
from flwr.server.strategy import Strategy
from flwr.common import EvaluateRes, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from scipy.stats import norm

from src.data_models.simulation_strategy_history import SimulationStrategyHistory
from src.output_handlers.attack_snapshot_handler import save_visual_snapshot


class LittleIsEnoughAttack(fl.server.strategy.FedAvg):

    def __init__(
            self,
            aggregation_strategy: Strategy,
            remove_clients: bool,
            begin_removing_from_round: int,
            strategy_history: SimulationStrategyHistory,
            output_dirname,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.aggregation_strategy = aggregation_strategy
        self.strategy_history = strategy_history

        # Create a logger
        self.logger = logging.getLogger(f"pid_strategy_{id(self)}")
        self.logger.setLevel(logging.INFO)  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        # Create handlers
        self.out_dir = output_dirname
        os.makedirs(self.out_dir, exist_ok=True)
        file_handler = logging.FileHandler(f"{self.out_dir}/output.log")
        console_handler = logging.StreamHandler()

        # Add the handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

    
    @staticmethod
    def check_for_attack(attack: Union[str, List[dict]]) -> bool:
        """Check if there will be a little is enough attack during the simulation.

        Args:
            attack: Attack type or Attack configuration list of dicts

        Returns:
            Boolean for whether there will be a little is enough attack
        """

        if isinstance(attack, str):
            return attack == "little_is_enough"
        elif isinstance(attack, list):
            if attack:
                return True if "little_is_enough" in [cfg.get("attack_type") or cfg.get("type", "unknown") for cfg in attack] else False
        return False
    

    def _compute_stats(self, malicious_updates) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Compute mean and standard deviation from malicious updates."""

        all_weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in malicious_updates]
        # group same layers together
        layers = list(zip(*all_weights))

        mu = []
        sigma = []
        for layer in layers:
            stacked = np.stack(layer)
            mu_layer = np.mean(stacked, axis=0)
            sigma_layer = np.std(stacked, axis=0)
            mu.append(mu_layer)
            sigma.append(sigma_layer)

        return mu, sigma
    

    def _replace_parameters(self, fit_res, mu, sigma, z_max) -> FitRes:
        """Replace a malicious client's parameters with adversarial ones."""

        original_layers = parameters_to_ndarrays(fit_res.parameters)
        new_layers = []
        for layer_idx, layer in enumerate(original_layers):
            mu_layer = mu[layer_idx]
            sigma_layer = sigma[layer_idx]
            # Construct adversarial update
            adv_layer = mu_layer + z_max * sigma_layer
            adv_layer = adv_layer.astype(layer.dtype)   # Ensure shape consistency
            new_layers.append(adv_layer)

        new_params = ndarrays_to_parameters(new_layers)
        # Create new FitRes with modified parameters
        new_fit_res = FitRes(
            parameters=new_params,
            num_examples=fit_res.num_examples,
            metrics=fit_res.metrics,
            status=fit_res.status
        )
        
        return new_fit_res


    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Handle empty results
        if not results:
            return self.aggregation_strategy.aggregate_fit(server_round, results, failures)
        
        new_results = results

        """
        Perform little is enough attack first
        """
        malicious = []
        benign = []
        for client_proxy, fit_res in results:
            client_id = int(client_proxy.cid)
            client = self.strategy_history.get_client_by_id(client_id)
            client_current_attacks = client.get_current_attack_keywords(server_round)
            if "little_is_enough" in client_current_attacks:
                malicious.append((client_proxy, fit_res))
            else:
                benign.append((client_proxy, fit_res))

        if malicious:
            # Set the number of requred workers for a majorty
            n = len(malicious) + len(benign)    # number of total clients
            m = len(malicious)       # number of malicious clients
            s = math.floor((n/2) + 1) - m

            if m < 2:
                self.logger.warning(
                    f"[Little Is Enough Attack] Number of malicious clients m must satisfy m >= 2. Continuing without attack."
                )
                return self.aggregation_strategy.aggregate_fit(server_round, results, failures)

            if s < 0:
                self.logger.warning(
                    f"[Little Is Enough Attack] Number of malicious clients m must satisfy m <= Floor(n/2 + 1). Continuing without attack."
                )
                return self.aggregation_strategy.aggregate_fit(server_round, results, failures)
            
            # Set z_max value using standard normal table
            # if s = 0, clamp p to just under 1 to allow attack to continue
            p = (n - s) / n
            p = min(max(p, 1e-6), 1 - 1e-6)
            z_max = norm.ppf(p)

            # Compute per layer μ and σ from malicious updates
            mu, sigma = self._compute_stats(malicious)

            # Create modified malicious updates
            modified_malicious = []
            for client, fit_res in malicious:
                new_fit_res = self._replace_parameters(fit_res, mu, sigma, z_max)
                modified_malicious.append((client, new_fit_res))

            # Combine malicious and benign
            new_results = benign + modified_malicious

            # Save little_is_enough visual snapshot for each malicious client
            if self.strategy_history.strategy_config.attack_schedule:
                experiment_info = {
                    'total_clients': n,
                    'malicious_clients': m,
                    'benign_clients': len(benign),
                    's': s,
                    'p': p,
                    'z_max': z_max
                }
                for client_proxy, _ in malicious:
                    client_id = int(client_proxy.cid)
                    client_instance = self.strategy_history.get_client_by_id(client_id)
                    current_round_attack_configs = client_instance.get_current_attack_configs(server_round)

                    save_visual_snapshot(
                        client_id=client_id,
                        round_num=server_round,
                        attack_config=current_round_attack_configs,
                        data_sample=np.array([]),
                        labels_sample=np.array([]),
                        original_labels_sample=np.array([]),
                        output_dir=self.out_dir,
                        experiment_info=experiment_info,
                        strategy_number=self.strategy_history.strategy_config.strategy_number,
                        specific_attack_type="little_is_enough"
                    )

        return self.aggregation_strategy.aggregate_fit(server_round, new_results, failures)


    def configure_fit(self, server_round, parameters, client_manager):
        return self.aggregation_strategy.configure_fit(server_round, parameters, client_manager)
    

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Tuple[Union[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return self.aggregation_strategy.aggregate_evaluate(server_round, results, failures)

    


