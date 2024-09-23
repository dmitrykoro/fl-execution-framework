

class AdditionalDataCalculator:

    @staticmethod
    def calculate_data(data: dict) -> dict:
        """
        The primary data that is collected during the simulation is as follows (for each client at each round):

        loss,
        accuracy - achieved accuracy during the validation,
        removal_criterion - the calculated value based on which the removal is performed,
        absolute_distance (to cluster center) - the absolute distance to the center of the cluster of all models
        normalized_distance (to cluster center) - the distance to the cluster center scaled between 0 and 1

        The following derivative data is calculated here (for each round):

        average_loss - average loss of all clients that are not removed at a given round,
        average_accuracy - average accuracy of all clients that are not removed at a given round,
        average_distance - average absolute distance to the cluster center of all clients that are not removed at a given round

        """

        for round_num, round_data in data.items():

            total_round_loss = 0.0
            total_round_accuracy = 0.0
            total_round_distance = 0.0

            num_of_round_clients = 0

            for client_id, client_data in round_data['client_info'].items():
                if client_data['is_removed']:
                    continue

                total_round_loss += client_data['loss']
                total_round_accuracy += client_data['accuracy']
                total_round_distance += client_data['absolute_distance']

                num_of_round_clients += 1

            data[round_num]['round_info'] = {}
            data[round_num]['round_info']['average_loss'] = total_round_loss / num_of_round_clients
            data[round_num]['round_info']['average_accuracy'] = total_round_accuracy / num_of_round_clients
            data[round_num]['round_info']['average_distance'] = total_round_distance / num_of_round_clients

        return data
