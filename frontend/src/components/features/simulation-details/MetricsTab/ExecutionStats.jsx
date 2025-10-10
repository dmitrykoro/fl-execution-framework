import { Card } from 'react-bootstrap';

export function ExecutionStats({ data, config }) {
  if (!data || data.length === 0) {
    return null;
  }

  const stats = data[0];
  const meanAccuracy = parseFloat(stats.mean_average_accuracy_history || 0);

  return (
    <Card className="mb-4">
      <Card.Header>
        <div className="d-flex justify-content-between align-items-center">
          <h5 className="mb-0">Execution Statistics</h5>
          <small className="text-muted">Performance timing breakdown</small>
        </div>
      </Card.Header>
      <Card.Body>
        <div className="row g-3">
          <div className="col-12 col-sm-6 col-md-4">
            <div className="p-3 border rounded">
              <div className="small text-muted">Final Accuracy</div>
              <div className="h4 mb-0 text-dark">{(meanAccuracy * 100).toFixed(1)}%</div>
            </div>
          </div>
          <div className="col-12 col-sm-6 col-md-4">
            <div className="p-3 border rounded">
              <div className="small text-muted">Total Rounds</div>
              <div className="h4 mb-0 text-dark">{config.num_of_rounds}</div>
            </div>
          </div>
          <div className="col-12 col-sm-6 col-md-4">
            <div className="p-3 border rounded">
              <div className="small text-muted">Total Clients</div>
              <div className="h4 mb-0 text-dark">{config.num_of_clients}</div>
            </div>
          </div>
        </div>
        <div className="mt-3 small text-muted">
          <strong>Tip:</strong> Compare these metrics across different aggregation strategies to see
          which defends best against attacks!
        </div>
      </Card.Body>
    </Card>
  );
}
