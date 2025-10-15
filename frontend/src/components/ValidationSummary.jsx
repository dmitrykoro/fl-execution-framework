import { Alert } from 'react-bootstrap';

/**
 * Validation Summary Component
 *
 * Displays aggregated validation status above the submit button
 * Shows errors, warnings, and info messages in organized alerts
 *
 * @param {Array} errors - Array of error objects {field, message}
 * @param {Array} warnings - Array of warning objects {field, message}
 * @param {Array} infos - Array of info objects {field, message}
 */
function ValidationSummary({ errors = [], warnings = [], infos = [] }) {
  // If no validation messages, show success
  if (errors.length === 0 && warnings.length === 0 && infos.length === 0) {
    return (
      <Alert variant="success" className="mb-3 py-2 validation-success-alert">
        ✓ Configuration valid - ready to launch
      </Alert>
    );
  }

  return (
    <div className="validation-summary mb-3">
      {/* Errors */}
      {errors.length > 0 && (
        <Alert variant="danger" className="mb-2">
          <strong>
            ❌ {errors.length} Error{errors.length > 1 ? 's' : ''}
          </strong>
          <ul className="mb-0 mt-2">
            {errors.map((e, i) => (
              <li key={i}>
                <code>{e.field}</code>: {e.message}
              </li>
            ))}
          </ul>
        </Alert>
      )}

      {/* Warnings */}
      {warnings.length > 0 && (
        <Alert variant="warning" className="mb-2">
          <strong>
            ⚠️ {warnings.length} Warning{warnings.length > 1 ? 's' : ''}
          </strong>
          <ul className="mb-0 mt-2">
            {warnings.map((w, i) => (
              <li key={i}>
                <code>{w.field}</code>: {w.message}
              </li>
            ))}
          </ul>
        </Alert>
      )}

      {/* Infos */}
      {infos.length > 0 && (
        <Alert variant="info" className="mb-2">
          <strong>ℹ️ Information</strong>
          <ul className="mb-0 mt-2">
            {infos.map((info, i) => (
              <li key={i}>
                <code>{info.field}</code>: {info.message}
              </li>
            ))}
          </ul>
        </Alert>
      )}
    </div>
  );
}

export default ValidationSummary;
