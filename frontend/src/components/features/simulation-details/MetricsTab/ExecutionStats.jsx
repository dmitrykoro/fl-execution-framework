import { Card, Badge } from 'react-bootstrap';
import { InfoTooltip } from '@components/common/Tooltip/InfoTooltip';

export function ExecutionStats({ data, config }) {
  if (!data || data.length === 0) {
    return null;
  }

  const stats = data[0];

  const finalAccuracy = parseFloat(stats.final_accuracy || 0) * 100;
  const finalLoss = parseFloat(stats.final_loss || 0);
  const avgScoreTime = parseFloat(stats.avg_score_calc_time_ms || 0);

  const hasDefenseMetrics = stats.mean_average_accuracy_history !== undefined;
  const defenseAccuracy = hasDefenseMetrics ? stats.mean_average_accuracy_history : null;
  const defensePrecision = hasDefenseMetrics ? stats.mean_removal_precision_history : null;
  const defenseRecall = hasDefenseMetrics ? stats.mean_removal_recall_history : null;

  return (
    <Card className="mb-4">
      <Card.Header>
        <div className="d-flex justify-content-between align-items-center">
          <h5 className="mb-0">Execution Summary</h5>
          <small className="text-muted">Key performance metrics</small>
        </div>
      </Card.Header>
      <Card.Body>
        <div className="row g-3">
          <div className="col-12 col-sm-6 col-lg-3">
            <div className="p-3 border rounded">
              <div className="small text-muted">
                <InfoTooltip
                  text="Final model accuracy after all training rounds completed"
                  placement="top"
                >
                  Final Accuracy
                </InfoTooltip>
              </div>
              <div className="h4 mb-0 text-dark">{finalAccuracy.toFixed(2)}%</div>
            </div>
          </div>
          <div className="col-12 col-sm-6 col-lg-3">
            <div className="p-3 border rounded">
              <div className="small text-muted">
                <InfoTooltip text="Final aggregated loss value across all clients" placement="top">
                  Final Loss
                </InfoTooltip>
              </div>
              <div className="h4 mb-0 text-dark">{finalLoss.toFixed(4)}</div>
            </div>
          </div>
          <div className="col-12 col-sm-6 col-lg-3">
            <div className="p-3 border rounded">
              <div className="small text-muted">
                <InfoTooltip
                  text="Average time to compute aggregation scores per round"
                  placement="top"
                >
                  Avg Score Time
                </InfoTooltip>
              </div>
              <div className="h4 mb-0 text-dark">{avgScoreTime.toFixed(2)}ms</div>
            </div>
          </div>
          <div className="col-12 col-sm-6 col-lg-3">
            <div className="p-3 border rounded">
              <div className="small text-muted">Configuration</div>
              <div className="d-flex flex-column gap-1">
                <div className="small">
                  <Badge bg="secondary">{config.num_of_rounds} rounds</Badge>
                </div>
                <div className="small">
                  <Badge bg="secondary">{config.num_of_clients} clients</Badge>
                </div>
              </div>
            </div>
          </div>
        </div>

        {hasDefenseMetrics && (
          <>
            <hr className="my-4" />
            <div className="mb-2">
              <h6 className="text-muted mb-3">Defense Performance Metrics</h6>
            </div>
            <div className="row g-3">
              <div className="col-12 col-sm-6 col-lg-4">
                <div className="p-3 border rounded bg-light">
                  <div className="small text-muted">
                    <InfoTooltip
                      text="Mean accuracy across rounds after client removal began"
                      placement="top"
                    >
                      Avg Accuracy
                    </InfoTooltip>
                  </div>
                  <div className="h5 mb-0 text-dark">{defenseAccuracy}</div>
                </div>
              </div>
              <div className="col-12 col-sm-6 col-lg-4">
                <div className="p-3 border rounded bg-light">
                  <div className="small text-muted">
                    <InfoTooltip
                      text="Precision of identifying malicious clients (TP / TP+FP)"
                      placement="top"
                    >
                      Defense Precision
                    </InfoTooltip>
                  </div>
                  <div className="h5 mb-0 text-dark">{defensePrecision}</div>
                </div>
              </div>
              <div className="col-12 col-sm-6 col-lg-4">
                <div className="p-3 border rounded bg-light">
                  <div className="small text-muted">
                    <InfoTooltip
                      text="Recall of identifying malicious clients (TP / TP+FN)"
                      placement="top"
                    >
                      Defense Recall
                    </InfoTooltip>
                  </div>
                  <div className="h5 mb-0 text-dark">{defenseRecall}</div>
                </div>
              </div>
            </div>
          </>
        )}

        <div className="mt-3 small text-muted">
          <strong>Tip:</strong> Compare these metrics across different aggregation strategies to see
          which performs best for your use case!
        </div>
      </Card.Body>
    </Card>
  );
}
