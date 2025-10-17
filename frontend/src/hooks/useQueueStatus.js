import { useState, useEffect } from 'react';
import { getSimulationDetails, getSimulationStatus } from '@api';

export function useQueueStatus(simulationId) {
  const [simulation, setSimulation] = useState(null);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Calculate queue progress
  const calculateProgress = (sim, stat) => {
    if (!sim?.config?.simulation_strategies || !stat) {
      return { current: 0, total: 1, strategies: [] };
    }

    const totalStrategies = sim.config.simulation_strategies.length;
    const resultFiles = stat.result_files || [];

    // Detect completed strategies by looking for strategy_N/csv files
    const completedStrategies = [];
    for (let i = 0; i < totalStrategies; i++) {
      const hasResults = resultFiles.some(f => f.includes(`strategy_${i}/csv`));
      if (hasResults) {
        completedStrategies.push(i);
      }
    }

    const currentStrategy = completedStrategies.length;
    const isComplete = currentStrategy >= totalStrategies;

    // Build strategy status list
    const strategies = sim.config.simulation_strategies.map((strat, index) => {
      let stratStatus = 'queued';
      if (completedStrategies.includes(index)) {
        stratStatus = 'completed';
      } else if (index === currentStrategy && stat.status === 'running') {
        stratStatus = 'running';
      } else if (stat.status === 'failed') {
        stratStatus = 'failed';
      }

      return {
        index,
        config: strat,
        status: stratStatus,
      };
    });

    return {
      current: currentStrategy,
      total: totalStrategies,
      strategies,
      isComplete: isComplete && stat.status === 'completed',
    };
  };

  useEffect(() => {
    let intervalId;

    const fetchData = async () => {
      try {
        const [simResponse, statusResponse] = await Promise.all([
          getSimulationDetails(simulationId),
          getSimulationStatus(simulationId),
        ]);

        setSimulation(simResponse.data);
        setStatus(statusResponse.data);
        setLoading(false);
      } catch (err) {
        console.error('Failed to fetch queue status:', err);
        setError(err);
        setLoading(false);
      }
    };

    fetchData();

    // Poll every 5 seconds
    intervalId = setInterval(fetchData, 5000);

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [simulationId]);

  const progress = simulation && status ? calculateProgress(simulation, status) : null;

  return {
    simulation,
    status,
    progress,
    loading,
    error,
  };
}
