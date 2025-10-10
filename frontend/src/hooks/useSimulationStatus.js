import { useState, useCallback, useEffect } from 'react';
import { getSimulationStatus } from '@api/endpoints/simulations';
import { usePolling } from './usePolling';

export function useSimulationStatus(simulationIdOrSimulations, options = {}) {
  // Handle both single simulation (string) and multiple simulations (array)
  const isMultiple = Array.isArray(simulationIdOrSimulations);

  const [status, setStatus] = useState(null);
  const [statuses, setStatuses] = useState({});
  const [error, setError] = useState(null);

  const pollStatus = useCallback(async () => {
    if (isMultiple) {
      // Poll multiple simulations
      const simulations = simulationIdOrSimulations;
      if (!simulations || simulations.length === 0) return;

      try {
        const statusPromises = simulations.map(sim =>
          getSimulationStatus(sim.simulation_id)
            .then(response => ({ id: sim.simulation_id, data: response.data }))
            .catch(err => ({ id: sim.simulation_id, error: err }))
        );

        const results = await Promise.all(statusPromises);
        const newStatuses = {};
        results.forEach(result => {
          if (result.data) {
            newStatuses[result.id] = result.data;
          }
        });
        setStatuses(newStatuses);
      } catch (err) {
        setError(err);
      }
    } else {
      // Poll single simulation
      const simulationId = simulationIdOrSimulations;
      if (!simulationId) return;

      try {
        const response = await getSimulationStatus(simulationId);
        setStatus(response.data);
      } catch (err) {
        setError(err);
      }
    }
  }, [simulationIdOrSimulations, isMultiple]);

  // Initial fetch
  useEffect(() => {
    pollStatus();
  }, [pollStatus]);

  const shouldPoll = isMultiple
    ? Object.values(statuses).some(s => s.status === 'running')
    : status?.status === 'running';

  usePolling(pollStatus, options.interval || 2000, shouldPoll);

  return isMultiple
    ? { statuses, error, refetch: pollStatus }
    : { status, error, refetch: pollStatus };
}
