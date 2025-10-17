import { useState } from 'react';
import { createSimulation } from '@api';

export function useMultiSimulation() {
  const [isSubmitting, setIsSubmitting] = useState(false);

  const createMultiSimulation = async multiSimConfig => {
    setIsSubmitting(true);
    try {
      const response = await createSimulation(multiSimConfig);
      return response;
    } finally {
      setIsSubmitting(false);
    }
  };

  return {
    createMultiSimulation,
    isSubmitting,
  };
}
