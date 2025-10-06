import { useMemo } from 'react';
import { validateConfig } from '../utils/configValidation';

/**
 * Real-time configuration validation hook
 *
 * Automatically validates config whenever it changes using useMemo for performance
 * Returns validation results with errors, warnings, infos, and isValid flag
 *
 * @param {Object} config - Simulation configuration object
 * @returns {Object} { errors, warnings, infos, isValid }
 */
export function useConfigValidation(config) {
  return useMemo(() => {
    const result = validateConfig(config);
    return {
      ...result,
      isValid: result.errors.length === 0
    };
  }, [config]);
}
