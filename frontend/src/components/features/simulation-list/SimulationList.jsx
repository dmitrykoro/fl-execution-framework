import { Row, Col } from 'react-bootstrap';
import { SimulationCard } from './SimulationCard';

export function SimulationList({
  simulations,
  statuses,
  selectedSims,
  onCardClick,
  onDelete,
  onRename,
  onStop,
  deleting,
  stopping,
}) {
  if (!simulations || simulations.length === 0) {
    return (
      <Row xs={1} md={2} lg={3} className="g-4">
        <Col>
          <p>No simulations found.</p>
        </Col>
      </Row>
    );
  }

  return (
    <Row xs={1} md={2} lg={3} className="g-4">
      {simulations.map(sim => (
        <Col key={sim.simulation_id}>
          <SimulationCard
            simulation={sim}
            statusData={statuses[sim.simulation_id]}
            isSelected={selectedSims.includes(sim.simulation_id)}
            onCardClick={onCardClick}
            onDelete={onDelete}
            onRename={onRename}
            onStop={onStop}
            deleting={deleting}
            stopping={stopping}
          />
        </Col>
      ))}
    </Row>
  );
}
