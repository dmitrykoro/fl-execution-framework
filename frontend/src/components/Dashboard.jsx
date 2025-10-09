import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Card,
  Row,
  Col,
  Spinner,
  Alert,
  Button,
  Badge,
  Tooltip,
  OverlayTrigger,
} from 'react-bootstrap';
import { Trash2 } from 'lucide-react';
import useApi from '../hooks/useApi';
import {
  getSimulations,
  getSimulationStatus,
  deleteSimulation,
  deleteMultipleSimulations,
} from '../api';
import EditableSimName from './EditableSimName';
import ConfirmModal from './ConfirmModal';
import { useToast } from '../contexts/ToastContext';

function Dashboard() {
  const { data: simulations, loading, error, refetch } = useApi(getSimulations);
  const [statuses, setStatuses] = useState({});
  const [selectedSims, setSelectedSims] = useState([]);
  const [deleting, setDeleting] = useState(false);
  const navigate = useNavigate();
  const { showSuccess, showError, showWarning } = useToast();

  // Modal states
  const [confirmModal, setConfirmModal] = useState({
    show: false,
    title: '',
    message: '',
    onConfirm: () => {},
    variant: 'danger',
  });

  const getRelativeTime = timestamp => {
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

  const parseErrorMessage = errorText => {
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

  useEffect(() => {
    if (!simulations) return;

    const fetchStatuses = async () => {
      const newStatuses = {};
      for (const sim of simulations) {
        try {
          const response = await getSimulationStatus(sim.simulation_id);
          newStatuses[sim.simulation_id] = response.data;
        } catch {
          newStatuses[sim.simulation_id] = { status: 'unknown' };
        }
      }
      setStatuses(newStatuses);
    };

    fetchStatuses();
    const interval = setInterval(fetchStatuses, 5000);
    return () => clearInterval(interval);
  }, [simulations]);

  const handleCardClick = (simId, e) => {
    // Don't toggle selection if clicking on interactive elements
    if (
      e.target.closest('a') ||
      e.target.closest('button') ||
      e.target.closest('input') ||
      e.target.closest('.editable-sim-name')
    ) {
      return;
    }
    setSelectedSims(prev =>
      prev.includes(simId) ? prev.filter(id => id !== simId) : [...prev, simId]
    );
  };

  const handleCompare = () => {
    if (selectedSims.length < 2) {
      showWarning('Please select at least 2 simulations to compare');
      return;
    }
    navigate(`/compare?ids=${selectedSims.join(',')}`);
  };

  const handleDeleteOne = async simId => {
    const statusData = statuses[simId];
    if (statusData?.status === 'running') {
      showError('Cannot delete a running simulation');
      return;
    }

    setConfirmModal({
      show: true,
      title: '🗑️ Delete Simulation',
      message: `Delete simulation "${simId}"?`,
      variant: 'danger',
      onConfirm: async () => {
        setConfirmModal({ ...confirmModal, show: false });
        setDeleting(true);
        try {
          await deleteSimulation(simId);
          setSelectedSims(prev => prev.filter(id => id !== simId));
          await refetch();
          showSuccess('Simulation deleted successfully');
        } catch (err) {
          showError(`Failed to delete: ${err.response?.data?.detail || err.message}`);
        } finally {
          setDeleting(false);
        }
      },
    });
  };

  const handleDeleteSelected = async () => {
    if (selectedSims.length === 0) {
      showWarning('No simulations selected');
      return;
    }

    const runningSimulations = selectedSims.filter(simId => statuses[simId]?.status === 'running');
    if (runningSimulations.length > 0) {
      showError(`Cannot delete running simulations: ${runningSimulations.join(', ')}`);
      return;
    }

    setConfirmModal({
      show: true,
      title: '🗑️ Delete Simulations',
      message: `Delete ${selectedSims.length} simulation(s)?`,
      variant: 'danger',
      onConfirm: async () => {
        setConfirmModal({ ...confirmModal, show: false });
        setDeleting(true);
        try {
          const response = await deleteMultipleSimulations(selectedSims);
          const { deleted, failed } = response.data;

          if (failed.length > 0) {
            const failedList = failed.map(f => `${f.simulation_id}: ${f.error}`).join('\n');
            showWarning(`Some deletions failed:\n${failedList}`);
          }

          if (deleted.length > 0) {
            setSelectedSims([]);
            await refetch();
            showSuccess(`Successfully deleted ${deleted.length} simulation(s)`);
          }
        } catch (err) {
          showError(`Failed to delete: ${err.response?.data?.detail || err.message}`);
        } finally {
          setDeleting(false);
        }
      },
    });
  };

  const handleClearAll = async () => {
    if (!simulations || simulations.length === 0) {
      showWarning('No simulations to clear');
      return;
    }

    const runningSimulations = simulations.filter(
      sim => statuses[sim.simulation_id]?.status === 'running'
    );
    if (runningSimulations.length > 0) {
      showError(
        `Cannot clear all: ${runningSimulations.length} simulation(s) still running. Please wait for them to complete or fail first.`
      );
      return;
    }

    setConfirmModal({
      show: true,
      title: '🗑️ Clear All Simulations',
      message: `Clear ALL ${simulations.length} simulation(s)?\n\nThis will permanently delete all simulation data and cannot be undone.`,
      variant: 'danger',
      onConfirm: async () => {
        setConfirmModal({ ...confirmModal, show: false });
        setDeleting(true);
        try {
          const allSimIds = simulations.map(sim => sim.simulation_id);
          const response = await deleteMultipleSimulations(allSimIds);
          const { deleted, failed } = response.data;

          if (failed.length > 0) {
            const failedList = failed.map(f => `${f.simulation_id}: ${f.error}`).join('\n');
            showWarning(`Some deletions failed:\n${failedList}`);
          }

          if (deleted.length > 0) {
            setSelectedSims([]);
            await refetch();
            showSuccess(`Successfully cleared ${deleted.length} simulation(s)`);
          }
        } catch (err) {
          showError(`Failed to clear simulations: ${err.response?.data?.detail || err.message}`);
        } finally {
          setDeleting(false);
        }
      },
    });
  };

  const getStatusBadge = statusData => {
    if (!statusData) {
      return (
        <span className="status-badge status-pending">
          <span className="status-dot"></span>
          pending
        </span>
      );
    }

    const { status, error: errorMsg } = statusData;

    const badge = (
      <span className={`status-badge status-${status || 'pending'}`}>
        <span className="status-dot"></span>
        {status || 'pending'}
      </span>
    );

    if (status === 'failed' && errorMsg) {
      return (
        <OverlayTrigger
          placement="left"
          overlay={
            <Tooltip id={`tooltip-error`}>
              <div style={{ textAlign: 'left', whiteSpace: 'pre-wrap', maxWidth: '400px' }}>
                {errorMsg}
              </div>
            </Tooltip>
          }
        >
          <span style={{ cursor: 'pointer' }}>{badge}</span>
        </OverlayTrigger>
      );
    }

    return badge;
  };

  if (loading) {
    return (
      <div>
        <h1>Simulation Dashboard</h1>
        <Spinner animation="border" role="status">
          <span className="visually-hidden">Loading...</span>
        </Spinner>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <h1>Simulation Dashboard</h1>
        <Alert variant="danger">{error}</Alert>
      </div>
    );
  }

  return (
    <div>
      <div className="d-flex flex-column flex-md-row justify-content-between align-items-start align-items-md-center mb-4 gap-3">
        <h1 className="mb-0">Simulation Dashboard</h1>
        <div className="d-flex flex-wrap gap-2 w-100 w-md-auto">
          {selectedSims.length > 0 && (
            <>
              <Button
                variant="danger"
                size="sm"
                onClick={handleDeleteSelected}
                disabled={deleting}
                className="flex-grow-1 flex-md-grow-0"
              >
                🗑️ Delete ({selectedSims.length})
              </Button>
              <Button
                variant="info"
                size="sm"
                onClick={handleCompare}
                className="flex-grow-1 flex-md-grow-0"
              >
                📊 Compare ({selectedSims.length})
              </Button>
            </>
          )}
          {simulations && simulations.length > 0 && (
            <Button
              variant="outline-danger"
              size="sm"
              onClick={handleClearAll}
              disabled={deleting}
              className="flex-grow-1 flex-md-grow-0"
            >
              🗑️ Clear All
            </Button>
          )}
        </div>
      </div>
      <Row xs={1} md={2} lg={3} className="g-4">
        {simulations && simulations.length > 0 ? (
          simulations.map(sim => {
            const statusData = statuses[sim.simulation_id];
            const isFailed = statusData?.status === 'failed';

            return (
              <Col key={sim.simulation_id}>
                <Card
                  onClick={e => handleCardClick(sim.simulation_id, e)}
                  className={`simulation-card ${
                    isFailed
                      ? 'border-danger'
                      : selectedSims.includes(sim.simulation_id)
                        ? 'selected'
                        : ''
                  }`}
                  style={{ cursor: 'pointer', position: 'relative' }}
                >
                  <button
                    className="delete-btn"
                    onClick={e => {
                      e.stopPropagation();
                      handleDeleteOne(sim.simulation_id);
                    }}
                    disabled={deleting || statusData?.status === 'running'}
                    title="Delete simulation"
                    aria-label="Delete simulation"
                  >
                    <Trash2 size={16} />
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
                              simulationId={sim.simulation_id}
                              displayName={sim.display_name}
                              onRename={() => refetch()}
                            />
                          </div>
                        </Card.Title>
                      </div>
                      <div className="flex-shrink-0 ms-2">{getStatusBadge(statusData)}</div>
                    </div>
                    <Card.Subtitle className="mb-2 text-muted">
                      {sim.display_name && (
                        <span className="small">
                          ID: <code>{sim.simulation_id}</code>
                          <span className="mx-2">•</span>
                        </span>
                      )}
                      {!sim.display_name && <code>{sim.simulation_id}</code>}
                      {sim.created_at && (
                        <span className="ms-2 small">
                          {sim.display_name && ''}
                          {getRelativeTime(sim.created_at)}
                        </span>
                      )}
                    </Card.Subtitle>
                    {isFailed && statusData.error && (
                      <Alert variant="danger" className="mb-2 py-2">
                        {parseErrorMessage(statusData.error)}
                      </Alert>
                    )}
                    <Card.Text>
                      Rounds: {sim.num_of_rounds} | Clients: {sim.num_of_clients}
                    </Card.Text>
                    <Link to={`/simulations/${sim.simulation_id}`}>View Details</Link>
                  </Card.Body>
                </Card>
              </Col>
            );
          })
        ) : (
          <Col>
            <p>No simulations found.</p>
          </Col>
        )}
      </Row>

      <ConfirmModal
        show={confirmModal.show}
        title={confirmModal.title}
        message={confirmModal.message}
        variant={confirmModal.variant}
        onConfirm={confirmModal.onConfirm}
        onCancel={() => setConfirmModal({ ...confirmModal, show: false })}
      />
    </div>
  );
}

export default Dashboard;
