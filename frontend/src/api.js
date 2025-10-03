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

export const getSimulationDetails = (simulationId) => {
  return apiClient.get(`/simulations/${simulationId}`);
};

export const createSimulation = (config) => {
  return apiClient.post('/simulations', config);
};

export const getSimulationStatus = (simulationId) => {
  return apiClient.get(`/simulations/${simulationId}/status`);
};

export const getResultFile = (simulationId, filename) => {
  return apiClient.get(`/simulations/${simulationId}/results/${filename}`);
};
