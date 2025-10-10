// Date and time formatting utilities

export const getRelativeTime = timestamp => {
  if (!timestamp) return '';
  const now = new Date();
  const then = new Date(timestamp);
  const diffMs = now - then;
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffSecs < 60) return 'just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return then.toLocaleDateString();
};

// Error message parsing and formatting
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
