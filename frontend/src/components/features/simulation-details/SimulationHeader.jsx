import { Badge } from 'react-bootstrap';
import OutlineButton from '@components/common/Button/OutlineButton';
import { MaterialIcon } from '@components/common/Icon/MaterialIcon';

export function SimulationHeader({ simulation, onRunAgain, isCloning, onStop, isStopping }) {
  const { config, status, id } = simulation;
  const cfg = config.shared_settings || config;
  const displayName = cfg.display_name;
  const isRunning = status === 'running';

  const statusVariant =
    status === 'completed'
      ? 'success'
      : status === 'failed'
        ? 'danger'
        : status === 'running'
          ? 'primary'
          : 'secondary';

  return (
    <div className="d-flex flex-column gap-2 mt-3 mb-3">
      <div className="d-flex flex-column flex-md-row align-items-stretch align-items-md-center gap-2 gap-md-3">
        <div className="d-flex align-items-center gap-2 flex-wrap flex-grow-1">
          <h4 className="mb-0">{displayName || id}</h4>
          <Badge bg={statusVariant}>{status}</Badge>
        </div>
        <div className="d-flex gap-2 flex-shrink-0">
          {isRunning && (
            <OutlineButton
              onClick={onStop}
              disabled={isStopping}
              variant="outline-warning"
              className="flex-shrink-0"
            >
              <div className="d-flex align-items-center gap-1">
                <MaterialIcon name="stop" size={24} />
                <span>{isStopping ? 'Stopping...' : 'Stop'}</span>
              </div>
            </OutlineButton>
          )}
          <OutlineButton onClick={onRunAgain} disabled={isCloning} className="flex-shrink-0">
            <div className="d-flex align-items-center gap-1">
              <MaterialIcon name="replay" size={18} />
              <span>{isCloning ? 'Starting...' : 'Run Again'}</span>
            </div>
          </OutlineButton>
        </div>
      </div>
      {displayName && (
        <div className="text-muted small">
          ID: <code>{id}</code>
        </div>
      )}
    </div>
  );
}
