import { apiClient } from '@api/client';

export const getSimulations = () => apiClient.get('/simulations');

export const getSimulationDetails = simulationId => apiClient.get(`/simulations/${simulationId}`);

export const createSimulation = config => apiClient.post('/simulations', config);

export const getSimulationStatus = simulationId =>
  apiClient.get(`/simulations/${simulationId}/status`);

export const getResultFile = (simulationId, filename) =>
  apiClient.get(`/simulations/${simulationId}/results/${filename}`);

export const deleteSimulation = simulationId => apiClient.delete(`/simulations/${simulationId}`);

export const deleteMultipleSimulations = simulationIds =>
  apiClient.delete('/simulations', { data: { simulation_ids: simulationIds } });

export const renameSimulation = (simulationId, displayName) =>
  apiClient.patch(`/simulations/${simulationId}/rename`, { display_name: displayName });

export const stopSimulation = simulationId => apiClient.post(`/simulations/${simulationId}/stop`);
