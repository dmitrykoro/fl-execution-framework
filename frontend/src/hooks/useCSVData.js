import { useState, useEffect } from 'react';
import { getResultFile } from '@api/endpoints/simulations';

export function useCSVData(simulationId, resultFiles) {
  const [csvData, setCsvData] = useState({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!resultFiles) return;

    const csvFiles = resultFiles.filter(file => file.endsWith('.csv'));
    if (csvFiles.length === 0) return;

    setLoading(true);

    Promise.all(
      csvFiles.map(async file => {
        try {
          const response = await getResultFile(simulationId, file);
          return { file, data: response.data };
        } catch (err) {
          console.error(`Failed to load CSV ${file}:`, err);
          return { file, data: null };
        }
      })
    ).then(results => {
      const newData = results.reduce(
        (acc, { file, data }) => (data ? { ...acc, [file]: data } : acc),
        {}
      );
      setCsvData(newData);
      setLoading(false);
    });
  }, [simulationId, resultFiles]);

  return { csvData, loading };
}
