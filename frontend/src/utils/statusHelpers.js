// Status badge and error message utilities

export const parseErrorMessage = errorText => {
  if (!errorText) return 'Unknown error occurred';

  // Check for validation errors (missing required fields)
  if (errorText.includes('is a required property')) {
    const match = errorText.match(/'([^']+)' is a required property/);
    const field = match ? match[1] : 'field';
    return `Configuration Error: Missing required field '${field}'`;
  }

  // Check for config loading errors
  if (errorText.includes('Error while loading config')) {
    return 'Configuration Error: Invalid or incomplete configuration';
  }

  // Check for enum validation errors
  if (errorText.includes('is not one of')) {
    const match = errorText.match(/'([^']+)' is not one of/);
    const value = match ? match[1] : 'value';
    return `Configuration Error: Invalid value '${value}'`;
  }

  // Default: take first line of error if multiline
  const lines = errorText.split('\n');
  const firstMeaningfulLine = lines.find(
    line => line.includes('ERROR') || line.includes('Error') || line.trim().length > 0
  );

  if (firstMeaningfulLine) {
    // Clean up the line (remove ERROR: prefix, trim)
    return firstMeaningfulLine
      .replace(/^ERROR:root:/i, '')
      .replace(/^ERROR:/i, '')
      .trim();
  }

  return 'An error occurred during simulation';
};

export const getStatusVariant = status => {
  if (status === 'completed') return 'success';
  if (status === 'failed') return 'danger';
  if (status === 'running') return 'primary';
  if (status === 'stopped') return 'warning';
  return 'secondary';
};
