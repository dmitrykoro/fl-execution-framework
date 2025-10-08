import { useState, useEffect } from 'react';
import axios from 'axios';

/**
 * Real-time HuggingFace dataset validation hook
 *
 * Debounces validation requests to avoid API spam
 * Returns validation status with loading, valid, compatible, info, error fields
 *
 * @param {string} datasetName - HuggingFace dataset identifier
 * @returns {Object} { loading, valid, compatible, info, error }
 */
export function useDatasetValidation(datasetName) {
  const [status, setStatus] = useState({
    loading: false,
    valid: null,
    compatible: null,
    info: null,
    error: null
  });

  useEffect(() => {
    // Don't validate empty or very short strings
    if (!datasetName || datasetName.length < 3) {
      setStatus({ loading: false, valid: null, compatible: null, info: null, error: null });
      return;
    }

    // Debounce validation by 500ms
    const timeoutId = setTimeout(async () => {
      setStatus(prev => ({ ...prev, loading: true }));

      try {
        const response = await axios.get(`/api/datasets/validate?name=${encodeURIComponent(datasetName)}`);
        setStatus({
          loading: false,
          valid: response.data.valid,
          compatible: response.data.compatible,
          info: response.data.info,
          error: response.data.error
        });
      } catch (err) {
        setStatus({
          loading: false,
          valid: false,
          compatible: false,
          info: null,
          error: 'Validation request failed'
        });
      }
    }, 500);

    return () => clearTimeout(timeoutId);
  }, [datasetName]);

  return status;
}
