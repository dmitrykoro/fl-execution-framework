import { useState, useEffect } from 'react';
import { getSimulations, getSimulationStatus } from '@api';

/**
 * Hook to detect if any simulation is currently running
 * Used for queue blocking logic to prevent concurrent simulations
 */
export function useRunningSimulation() {
  const [hasRunning, setHasRunning] = useState(false);
  const [runningSimIds, setRunningSimIds] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let intervalId;

    const checkRunning = async () => {
      try {
        const response = await getSimulations();
        const simulations = response.data;

        // Check status for each simulation
        const statusPromises = simulations.map(sim =>
          getSimulationStatus(sim.simulation_id).catch(() => ({ data: { status: 'unknown' } }))
        );
        const statuses = await Promise.all(statusPromises);

        const running = simulations
          .filter((sim, i) => statuses[i].data.status === 'running')
          .map(sim => sim.simulation_id);

        setRunningSimIds(running);
        setHasRunning(running.length > 0);
        setLoading(false);
      } catch (err) {
        console.error('Failed to check running simulations:', err);
        setLoading(false);
      }
    };

    checkRunning();

    // Poll every 10 seconds
    intervalId = setInterval(checkRunning, 10000);

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, []);

  return {
    hasRunning,
    runningSimIds,
    loading,
  };
}
