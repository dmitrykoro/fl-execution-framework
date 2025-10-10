import axios from 'axios';

const apiClient = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getSimulations = () => {
  return apiClient.get('/simulations');
};

export const getSimulationDetails = simulationId => {
  return apiClient.get(`/simulations/${simulationId}`);
};

export const createSimulation = config => {
  return apiClient.post('/simulations', config);
};

export const getSimulationStatus = simulationId => {
  return apiClient.get(`/simulations/${simulationId}/status`);
};

export const getResultFile = (simulationId, filename) => {
  return apiClient.get(`/simulations/${simulationId}/results/${filename}`);
};

export const deleteSimulation = simulationId => {
  return apiClient.delete(`/simulations/${simulationId}`);
};

export const deleteMultipleSimulations = simulationIds => {
  return apiClient.delete('/simulations', { data: { simulation_ids: simulationIds } });
};

export const renameSimulation = (simulationId, displayName) => {
  return apiClient.patch(`/simulations/${simulationId}/rename`, { display_name: displayName });
};

export const stopSimulation = simulationId => {
  return apiClient.post(`/simulations/${simulationId}/stop`);
};
