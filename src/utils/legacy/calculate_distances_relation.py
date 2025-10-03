def calculate_distances_relation(experiment_results):
    """Calculate relation of distances."""
    round_number = 10

    distance_sum_without_bad = 0.0
    norm_distance_sum_without_bad = 0.0

    distance_sum_with_bad = 0.0
    norm_distance_sum_with_bad = 0.0

    for client_data in experiment_results[round_number]:
        if client_data["cid"] not in ("10", "11"):
            distance_sum_without_bad += client_data["distance"]
            norm_distance_sum_without_bad += client_data["normalised_distance"]

        distance_sum_with_bad += client_data["distance"]
        norm_distance_sum_with_bad += client_data["normalised_distance"]

    print(
        f"Distance relation without bad / with bad: {distance_sum_without_bad / distance_sum_with_bad}"
    )
    print(
        f"Normalized distance relation without bad / with bad: {norm_distance_sum_without_bad / norm_distance_sum_with_bad}"
    )
