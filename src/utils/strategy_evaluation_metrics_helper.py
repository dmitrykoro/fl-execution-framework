from src.data_models.simulation_strategy_history import SimulationStrategyHistory

def store_evaluation_metrics(
        clinet_id: int,
        current_round: int,
        metrics: dict, 
        strategy_history: SimulationStrategyHistory
    ):
    """
    Extract the proper metrics from evaluation metrics dictionary and save in ClientInfo object
    """

    if "accuracy" in metrics:
        # For classification tasks
        strategy_history.insert_single_client_history_entry(
            client_id=clinet_id,
            current_round=current_round,
            accuracy=metrics['accuracy']
        )
    elif "ss_res" in metrics and "ss_tot" in metrics and "r2" in metrics:
        # For regression tasks
        strategy_history.insert_single_client_history_entry(
            client_id=clinet_id,
            current_round=current_round,
            ss_res=metrics["ss_res"],
            ss_tot=metrics["ss_tot"],
            r2=metrics["r2"]
        )
