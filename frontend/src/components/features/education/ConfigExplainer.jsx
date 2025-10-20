import { Card, Badge } from 'react-bootstrap';

const STRATEGY_EXPLANATIONS = {
  'multi-krum': {
    name: 'Multi-Krum',
    icon: '🛡️',
    description:
      'Byzantine-resilient aggregation that selects the k most similar client updates based on pairwise distance metrics.',
    parameters: {
      num_krum_selections:
        'Number of clients (k) to select for aggregation. Lower k provides stronger defense but may reduce model diversity.',
      num_of_malicious_clients:
        'Number of malicious clients in the simulation. Multi-Krum can tolerate up to (n-k-2)/2 Byzantine clients.',
    },
    strength: 'Strong defense against coordinated Byzantine attacks',
    weakness: 'May exclude honest but statistically different clients',
  },
  krum: {
    name: 'Krum',
    icon: '🎯',
    description:
      'Selects a single most trustworthy client update based on distance to other clients.',
    parameters: {
      num_of_malicious_clients:
        'Number of malicious clients. Krum can tolerate up to (n-f-2) Byzantine clients where f is malicious count.',
    },
    strength: 'Simple and effective for moderate Byzantine threats',
    weakness: 'Uses only one client update, potentially slowing convergence',
  },
  trust: {
    name: 'TRUST',
    icon: '🔐',
    description:
      'Trust-based client scoring and optional removal system that tracks client reliability over time.',
    parameters: {
      trust_threshold: 'Minimum trust score required to participate in aggregation (0-1 scale)',
      remove_clients: 'Whether to permanently remove low-trust clients or just down-weight them',
      begin_removing_from_round:
        'Round number to start evaluating clients for removal (allows warm-up period)',
    },
    strength: 'Adaptive defense that learns client behavior patterns',
    weakness: 'Requires multiple rounds to build accurate trust scores',
  },
  fedavg: {
    name: 'FedAvg',
    icon: '📊',
    description:
      'Standard federated averaging - simple weighted average of all client updates with no defense mechanisms.',
    parameters: {
      num_of_malicious_clients:
        'Number of malicious clients. FedAvg has NO defense against attacks.',
    },
    strength: 'Fast convergence in honest settings, baseline for comparison',
    weakness: 'Vulnerable to all poisoning attacks',
  },
  trimmed_mean: {
    name: 'Trimmed Mean',
    icon: '✂️',
    description:
      'Removes extreme values (top and bottom percentiles) before averaging to reduce impact of outliers.',
    parameters: {
      trim_ratio: 'Percentage of extreme values to trim from each end (e.g., 0.1 = trim 10%)',
    },
    strength: 'Robust against outlier poisoning attacks',
    weakness: 'May trim honest clients with legitimate statistical differences',
  },
};

export function ConfigExplainer({ strategy, config, variant = 'detailed' }) {
  const strategyKey = strategy?.toLowerCase() || 'fedavg';
  const explanation = STRATEGY_EXPLANATIONS[strategyKey];

  if (!explanation) {
    return null;
  }

  if (variant === 'compact') {
    return (
      <div className="d-flex align-items-center gap-2 mb-2">
        <span style={{ fontSize: '1.2rem' }}>{explanation.icon}</span>
        <span className="text-muted small">{explanation.description}</span>
      </div>
    );
  }

  return (
    <Card className="mb-3">
      <Card.Body>
        <div className="d-flex align-items-center gap-2 mb-3">
          <span style={{ fontSize: '1.5rem' }}>{explanation.icon}</span>
          <h5 className="mb-0">{explanation.name}</h5>
        </div>

        <p className="text-muted">{explanation.description}</p>

        {config && (
          <div className="mb-3">
            <h6 className="mb-2">📋 Configuration:</h6>
            {Object.entries(explanation.parameters).map(([param, desc]) => {
              const value = config[param];
              if (value === undefined) return null;

              return (
                <div key={param} className="mb-2 ps-3">
                  <div className="d-flex align-items-start gap-2">
                    <Badge bg="secondary">{value}</Badge>
                    <div className="flex-grow-1">
                      <strong className="d-block">{param}</strong>
                      <small className="text-muted">{desc}</small>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        <div className="d-flex gap-3">
          <div className="flex-fill">
            <h6 className="text-success mb-1">✅ Strength</h6>
            <small className="text-muted">{explanation.strength}</small>
          </div>
          <div className="flex-fill">
            <h6 className="text-warning mb-1">⚠️ Limitation</h6>
            <small className="text-muted">{explanation.weakness}</small>
          </div>
        </div>
      </Card.Body>
    </Card>
  );
}
