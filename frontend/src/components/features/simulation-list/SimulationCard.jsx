import { Card, Alert } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import { MaterialIcon } from '@components/common/Icon/MaterialIcon';
import { StatusBadge } from '@components/common/Badge/StatusBadge';
import EditableSimName from '@components/EditableSimName';
import { getRelativeTime, parseErrorMessage } from '@utils/formatters';

export function SimulationCard({
  simulation,
  statusData,
  isSelected,
  onCardClick,
  onDelete,
  onRename,
  onStop,
  deleting,
  stopping,
}) {
  const { simulation_id, display_name, created_at, num_of_rounds, num_of_clients } = simulation;
  const isFailed = statusData?.status === 'failed';
  const isRunning = statusData?.status === 'running';

  return (
    <Card
      onClick={e => onCardClick(simulation_id, e)}
      className={`simulation-card ${isFailed ? 'border-danger' : isSelected ? 'selected' : ''}`}
      style={{ cursor: 'pointer', position: 'relative' }}
    >
      {isRunning && (
        <button
          className="stop-btn"
          onClick={e => {
            e.stopPropagation();
            onStop(simulation_id);
          }}
          disabled={stopping}
          title="Stop simulation"
          aria-label="Stop simulation"
        >
          <MaterialIcon name="stop_circle" size={16} />
        </button>
      )}
      <button
        className="delete-btn"
        onClick={e => {
          e.stopPropagation();
          onDelete(simulation_id);
        }}
        disabled={deleting || statusData?.status === 'running'}
        title="Delete simulation"
        aria-label="Delete simulation"
      >
        <MaterialIcon name="delete" size={16} />
      </button>
      <Card.Body>
        <div
          className="d-flex justify-content-between align-items-start mb-2"
          style={{ minWidth: 0 }}
        >
          <div style={{ minWidth: 0, flex: '1 1 auto' }}>
            <Card.Title className="mb-0">
              <div className="editable-sim-name">
                <EditableSimName
                  simulationId={simulation_id}
                  displayName={display_name}
                  onRename={onRename}
                />
              </div>
            </Card.Title>
          </div>
          <div className="flex-shrink-0 ms-2">
            <StatusBadge status={statusData?.status} error={statusData?.error} />
          </div>
        </div>
        <Card.Subtitle className="mb-2 text-muted">
          {display_name && (
            <span className="small">
              ID: <code>{simulation_id}</code>
              <span className="mx-2">•</span>
            </span>
          )}
          {!display_name && <code>{simulation_id}</code>}
          {created_at && (
            <span className="ms-2 small">
              {display_name && ''}
              {getRelativeTime(created_at)}
            </span>
          )}
        </Card.Subtitle>
        {isFailed && statusData.error && (
          <Alert variant="danger" className="mb-2 py-2">
            {parseErrorMessage(statusData.error)}
          </Alert>
        )}
        <Card.Text>
          Rounds: {num_of_rounds} | Clients: {num_of_clients}
        </Card.Text>
        <Link to={`/simulations/${simulation_id}`}>View Details</Link>
      </Card.Body>
    </Card>
  );
}
