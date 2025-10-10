import { useState, useEffect, useCallback } from 'react';
import { getSimulations, getSimulationStatus } from '@api/endpoints/simulations';

export function useSimulations() {
  const [simulations, setSimulations] = useState([]);
  const [statuses, setStatuses] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchSimulations = async () => {
    try {
      const response = await getSimulations();
      setSimulations(response.data);
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const fetchStatuses = useCallback(async () => {
    if (!simulations.length) return;
    const newStatuses = {};
    for (const sim of simulations) {
      try {
        const response = await getSimulationStatus(sim.simulation_id);
        newStatuses[sim.simulation_id] = response.data;
      } catch {
        newStatuses[sim.simulation_id] = { status: 'unknown' };
      }
    }
    setStatuses(newStatuses);
  }, [simulations]);

  useEffect(() => {
    fetchSimulations();
  }, []);

  useEffect(() => {
    if (!simulations.length) return;
    fetchStatuses();
    const interval = setInterval(fetchStatuses, 5000);
    return () => clearInterval(interval);
  }, [simulations, fetchStatuses]);

  return { simulations, statuses, loading, error, refetch: fetchSimulations };
}
