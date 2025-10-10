import { OverlayTrigger, Tooltip } from 'react-bootstrap';

export function StatusBadge({ status, errorMessage, className = '' }) {
  const badge = (
    <span className={`status-badge status-${status || 'pending'} ${className}`}>
      <span className="status-dot"></span>
      {status || 'pending'}
    </span>
  );

  if (status === 'failed' && errorMessage) {
    return (
      <OverlayTrigger
        placement="left"
        overlay={
          <Tooltip id="tooltip-error">
            <div style={{ textAlign: 'left', whiteSpace: 'pre-wrap', maxWidth: '400px' }}>
              {errorMessage}
            </div>
          </Tooltip>
        }
      >
        <span style={{ cursor: 'pointer' }}>{badge}</span>
      </OverlayTrigger>
    );
  }

  return badge;
}
