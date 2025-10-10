import { useState, useEffect, useCallback } from 'react';
import { getSimulationDetails, getSimulationStatus } from '@api/endpoints/simulations';

export function useSimulationDetails(simulationId) {
  const [details, setDetails] = useState(null);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchDetails = useCallback(async () => {
    try {
      const response = await getSimulationDetails(simulationId);
      setDetails(response.data);
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  }, [simulationId]);

  const pollStatus = useCallback(async () => {
    try {
      const response = await getSimulationStatus(simulationId);
      setStatus(response.data.status);
      if (response.data.status === 'completed' && details?.status !== 'completed') {
        fetchDetails();
      }
    } catch (err) {
      console.error('Failed to poll status:', err);
    }
  }, [simulationId, details, fetchDetails]);

  useEffect(() => {
    fetchDetails();
  }, [fetchDetails]);

  useEffect(() => {
    if (!details || details.status !== 'running') return;
    pollStatus();
    const interval = setInterval(pollStatus, 2000);
    return () => clearInterval(interval);
  }, [details, pollStatus]);

  return { details, status, loading, error, refetch: fetchDetails };
}
