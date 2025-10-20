import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Alert, Button } from 'react-bootstrap';
import { PageContainer } from '@components/layout/PageContainer';
import { PageHeader } from '@components/layout/PageHeader';
import { LoadingPage } from '@components/common/Loading/LoadingPage';
import { SimulationList } from '@components/features/simulation-list/SimulationList';
import { BulkActions } from '@components/features/simulation-list/BulkActions';
import { ConfirmModal } from '@components/common/Modal/ConfirmModal';
import { useSimulations } from '@hooks/useSimulations';
import { useSimulationStatus } from '@hooks/useSimulationStatus';
import { useRunningSimulation } from '@hooks/useRunningSimulation';
import { deleteSimulation, deleteMultipleSimulations, stopSimulation } from '@api';
import { toast } from 'sonner';

export function Dashboard() {
  const { simulations, loading, error, refetch } = useSimulations();
  const { statuses } = useSimulationStatus(simulations);
  const { hasRunning, runningSimIds } = useRunningSimulation();
  const [selectedSims, setSelectedSims] = useState([]);
  const [deleting, setDeleting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const navigate = useNavigate();

  const [confirmModal, setConfirmModal] = useState({
    show: false,
    title: '',
    message: '',
    onConfirm: () => {},
    variant: 'danger',
  });

  const handleCardClick = (simId, e) => {
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
      toast.warning('Please select at least 2 simulations to compare');
      return;
    }
    navigate(`/compare?ids=${selectedSims.join(',')}`);
  };

  const handleStop = async simId => {
    setConfirmModal({
      show: true,
      title: 'Stop Simulation',
      message: `Stop running simulation "${simId}"? This action cannot be undone.`,
      variant: 'warning',
      onConfirm: async () => {
        setConfirmModal({ ...confirmModal, show: false });
        setStopping(true);
        try {
          await stopSimulation(simId);
          await refetch();
          toast.success('Simulation stopped successfully');
        } catch (err) {
          toast.error(`Failed to stop: ${err.response?.data?.detail || err.message}`);
        } finally {
          setStopping(false);
        }
      },
    });
  };

  const handleDeleteOne = async simId => {
    const statusData = statuses[simId];
    if (statusData?.status === 'running') {
      toast.error('Cannot delete a running simulation');
      return;
    }

    setConfirmModal({
      show: true,
      title: 'Delete Simulation',
      message: `Delete simulation "${simId}"?`,
      variant: 'danger',
      onConfirm: async () => {
        setConfirmModal({ ...confirmModal, show: false });
        setDeleting(true);
        try {
          await deleteSimulation(simId);
          setSelectedSims(prev => prev.filter(id => id !== simId));
          await refetch();
          toast.success('Simulation deleted successfully');
        } catch (err) {
          toast.error(`Failed to delete: ${err.response?.data?.detail || err.message}`);
        } finally {
          setDeleting(false);
        }
      },
    });
  };

  const handleDeleteSelected = async () => {
    if (selectedSims.length === 0) {
      toast.warning('No simulations selected');
      return;
    }

    const runningSimulations = selectedSims.filter(simId => statuses[simId]?.status === 'running');
    if (runningSimulations.length > 0) {
      toast.error(`Cannot delete running simulations: ${runningSimulations.join(', ')}`);
      return;
    }

    setConfirmModal({
      show: true,
      title: 'Delete Simulations',
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
            toast.warning(`Some deletions failed:\n${failedList}`);
          }

          if (deleted.length > 0) {
            setSelectedSims([]);
            await refetch();
            toast.success(`Successfully deleted ${deleted.length} simulation(s)`);
          }
        } catch (err) {
          toast.error(`Failed to delete: ${err.response?.data?.detail || err.message}`);
        } finally {
          setDeleting(false);
        }
      },
    });
  };

  const handleClearAll = async () => {
    if (!simulations || simulations.length === 0) {
      toast.warning('No simulations to clear');
      return;
    }

    const runningSimulations = simulations.filter(
      sim => statuses[sim.simulation_id]?.status === 'running'
    );
    if (runningSimulations.length > 0) {
      toast.error(
        `Cannot clear all: ${runningSimulations.length} simulation(s) still running. Please wait for them to complete or fail first.`
      );
      return;
    }

    setConfirmModal({
      show: true,
      title: 'Clear All Simulations',
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
            toast.warning(`Some deletions failed:\n${failedList}`);
          }

          if (deleted.length > 0) {
            setSelectedSims([]);
            await refetch();
            toast.success(`Successfully cleared ${deleted.length} simulation(s)`);
          }
        } catch (err) {
          toast.error(`Failed to clear simulations: ${err.response?.data?.detail || err.message}`);
        } finally {
          setDeleting(false);
        }
      },
    });
  };

  if (loading) {
    return <LoadingPage title="Simulation Dashboard" />;
  }

  if (error) {
    return (
      <PageContainer>
        <PageHeader title="Simulation Dashboard" />
        <div className="alert alert-danger">{error}</div>
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      <PageHeader title="Simulation Dashboard">
        <BulkActions
          selectedCount={selectedSims.length}
          totalCount={simulations?.length || 0}
          onDeleteSelected={handleDeleteSelected}
          onCompare={handleCompare}
          onClearAll={handleClearAll}
          deleting={deleting}
        />
      </PageHeader>

      {hasRunning && (
        <Alert variant="warning" className="mb-4">
          <div className="d-flex justify-content-between align-items-center">
            <div>
              <i className="bi bi-exclamation-triangle me-2"></i>
              <strong>Simulation in progress</strong> - New simulations will queue automatically
            </div>
            <Button
              as={Link}
              to={`/queue/${runningSimIds[0]}`}
              className="btn-warning-action"
              size="sm"
            >
              View Queue Status
            </Button>
          </div>
        </Alert>
      )}

      <SimulationList
        simulations={simulations}
        statuses={statuses}
        selectedSims={selectedSims}
        onCardClick={handleCardClick}
        onDelete={handleDeleteOne}
        onRename={refetch}
        onStop={handleStop}
        deleting={deleting}
        stopping={stopping}
      />

      <ConfirmModal
        show={confirmModal.show}
        title={confirmModal.title}
        message={confirmModal.message}
        variant={confirmModal.variant}
        onConfirm={confirmModal.onConfirm}
        onCancel={() => setConfirmModal({ ...confirmModal, show: false })}
      />
    </PageContainer>
  );
}
