import { Table, Button, Form, Dropdown, Badge } from 'react-bootstrap';
import { QUICK_PATTERNS, AGGREGATION_STRATEGIES } from '@constants/strategyVariations';
import { ATTACKS } from '@constants/attacks';
import { InfoTooltip } from '@components/common/Tooltip/InfoTooltip';
import { MaterialIcon } from '@components/common/Icon/MaterialIcon';

export function StrategyVariationList({ variations, onChange, numOfClients }) {
  const handleAddRow = () => {
    const nextNumber = variations.length + 1;

    const newVariation = {
      id: Date.now(),
      name: `strategy_${nextNumber}`,
      aggregation_strategy_keyword: 'fedavg',
      num_of_malicious_clients: 0,
      remove_clients: 'false',
    };
    onChange([...variations, newVariation]);
  };

  const handleDeleteRow = id => {
    onChange(variations.filter(v => v.id !== id));
  };

  const handleFieldChange = (id, field, value) => {
    onChange(
      variations.map(v => {
        if (v.id === id) {
          const updated = { ...v, [field]: value };

          if (field === 'aggregation_strategy_keyword' || field === 'num_of_malicious_clients') {
            const strategy =
              field === 'aggregation_strategy_keyword' ? value : v.aggregation_strategy_keyword;
            const malCount =
              field === 'num_of_malicious_clients' ? value : v.num_of_malicious_clients;
            updated.name = `${strategy}_mal${malCount}`;
          }

          return updated;
        }
        return v;
      })
    );
  };

  const handleQuickPattern = patternId => {
    const pattern = QUICK_PATTERNS[patternId];
    if (pattern) {
      const baseStrategy =
        variations.length > 0 ? variations[0].aggregation_strategy_keyword : 'fedavg';
      const generated = pattern.generate(baseStrategy);
      const withIds = generated.map((v, i) => ({
        ...v,
        id: Date.now() + i,
      }));
      onChange(withIds);
    }
  };

  return (
    <div className="strategy-variation-list">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <div>
          <h5 className="mb-1">Strategy Variations</h5>
          <p className="text-muted small mb-0">
            Define parameter variations to compare. At least 2 strategies required.
          </p>
        </div>
        <div className="d-flex gap-2">
          <Dropdown align="end">
            <Dropdown.Toggle variant="outline-secondary" size="sm">
              Quick Patterns
            </Dropdown.Toggle>
            <Dropdown.Menu>
              {Object.values(QUICK_PATTERNS).map(pattern => (
                <Dropdown.Item key={pattern.id} onClick={() => handleQuickPattern(pattern.id)}>
                  <div>
                    <strong>{pattern.name}</strong>
                    <div className="small text-muted">{pattern.description}</div>
                  </div>
                </Dropdown.Item>
              ))}
            </Dropdown.Menu>
          </Dropdown>
          <Button variant="primary" size="sm" onClick={handleAddRow}>
            Add Strategy
          </Button>
        </div>
      </div>

      {variations.length === 0 ? (
        <div className="text-center py-5 border rounded">
          <i className="bi bi-table empty-state-icon"></i>
          <p className="text-muted mt-3">
            No strategies yet. Click "Add Strategy" or choose a Quick Pattern to get started.
          </p>
        </div>
      ) : (
        <>
          <div className="table-responsive">
            <Table hover className="variation-table">
              <thead>
                <tr>
                  <th style={{ width: '8%' }}>#</th>
                  <th style={{ width: '20%' }}>
                    <InfoTooltip text="Descriptive name for this strategy variation">
                      Name
                    </InfoTooltip>
                  </th>
                  <th style={{ width: '20%' }}>
                    <InfoTooltip text="Federated learning aggregation method (FedAvg, Krum, Multi-Krum, Trust, PID)">
                      Aggregation Strategy
                    </InfoTooltip>
                  </th>
                  <th style={{ width: '17%' }}>
                    <InfoTooltip text="Number of Byzantine/malicious clients to simulate poisoning attacks">
                      Malicious Clients
                    </InfoTooltip>
                  </th>
                  <th style={{ width: '17%' }}>
                    <InfoTooltip text="Number of closest clients to aggregate. Lower = more Byzantine robustness but less data diversity.">
                      Krum Selections
                    </InfoTooltip>
                  </th>
                  <th style={{ width: '15%' }}>
                    <InfoTooltip text="Enable client removal based on trust scores">
                      Remove Clients
                    </InfoTooltip>
                  </th>
                  <th style={{ width: '15%' }}>
                    <InfoTooltip text="Type of Byzantine attack (Gaussian noise or Label flipping)">
                      Attack Type
                    </InfoTooltip>
                  </th>
                  <th style={{ width: '15%' }}>
                    <InfoTooltip text="Standard deviation for Gaussian noise attack">
                      Noise Std Dev
                    </InfoTooltip>
                  </th>
                  <th style={{ width: '15%' }}>
                    <InfoTooltip text="Trust score threshold below which clients are flagged (Trust strategy)">
                      Trust Threshold
                    </InfoTooltip>
                  </th>
                  <th style={{ width: '12%' }}>
                    <InfoTooltip text="Beta decay parameter for trust score calculation (Trust strategy)">
                      Beta Value
                    </InfoTooltip>
                  </th>
                  <th style={{ width: '15%' }}>
                    <InfoTooltip text="Round number to begin removing untrusted clients">
                      Remove From Round
                    </InfoTooltip>
                  </th>
                  <th style={{ width: '10%' }}>
                    <InfoTooltip text="Proportional gain for PID controller">Kp</InfoTooltip>
                  </th>
                  <th style={{ width: '10%' }}>
                    <InfoTooltip text="Integral gain for PID controller">Ki</InfoTooltip>
                  </th>
                  <th style={{ width: '10%' }}>
                    <InfoTooltip text="Derivative gain for PID controller">Kd</InfoTooltip>
                  </th>
                </tr>
              </thead>
              <tbody>
                {variations.map((variation, index) => (
                  <tr key={variation.id}>
                    <td>
                      <div className="d-flex align-items-center gap-2">
                        <Badge bg="secondary">{index + 1}</Badge>
                        <button
                          onClick={() => handleDeleteRow(variation.id)}
                          title="Delete strategy"
                          aria-label="Delete strategy"
                          className="strategy-delete-btn"
                        >
                          <MaterialIcon name="delete" size={18} />
                        </button>
                      </div>
                    </td>
                    <td>
                      <Form.Control
                        type="text"
                        size="sm"
                        value={variation.name}
                        onChange={e => handleFieldChange(variation.id, 'name', e.target.value)}
                      />
                    </td>
                    <td>
                      <Form.Select
                        size="sm"
                        value={variation.aggregation_strategy_keyword}
                        onChange={e =>
                          handleFieldChange(
                            variation.id,
                            'aggregation_strategy_keyword',
                            e.target.value
                          )
                        }
                      >
                        {AGGREGATION_STRATEGIES.map(s => (
                          <option key={s.value} value={s.value}>
                            {s.label}
                          </option>
                        ))}
                      </Form.Select>
                    </td>
                    <td>
                      <Form.Control
                        type="number"
                        size="sm"
                        min="0"
                        max={numOfClients}
                        value={variation.num_of_malicious_clients}
                        onChange={e =>
                          handleFieldChange(
                            variation.id,
                            'num_of_malicious_clients',
                            parseInt(e.target.value)
                          )
                        }
                        isInvalid={variation.num_of_malicious_clients > numOfClients}
                      />
                      {variation.num_of_malicious_clients > numOfClients && (
                        <Form.Control.Feedback type="invalid">
                          Cannot exceed {numOfClients} clients
                        </Form.Control.Feedback>
                      )}
                    </td>
                    <td>
                      {variation.aggregation_strategy_keyword === 'multi-krum' ? (
                        <>
                          <Form.Control
                            type="number"
                            size="sm"
                            min="1"
                            max={numOfClients - variation.num_of_malicious_clients}
                            value={variation.num_krum_selections || 18}
                            onChange={e =>
                              handleFieldChange(
                                variation.id,
                                'num_krum_selections',
                                parseInt(e.target.value)
                              )
                            }
                            isInvalid={
                              (variation.num_krum_selections || 18) >
                              numOfClients - variation.num_of_malicious_clients
                            }
                          />
                          {(variation.num_krum_selections || 18) >
                            numOfClients - variation.num_of_malicious_clients && (
                            <Form.Control.Feedback type="invalid">
                              Cannot exceed {numOfClients - variation.num_of_malicious_clients}{' '}
                              (total clients minus malicious clients)
                            </Form.Control.Feedback>
                          )}
                        </>
                      ) : (
                        <span className="text-muted small">-</span>
                      )}
                    </td>
                    <td>
                      <Form.Check
                        type="switch"
                        checked={variation.remove_clients === 'true'}
                        onChange={e =>
                          handleFieldChange(
                            variation.id,
                            'remove_clients',
                            e.target.checked ? 'true' : 'false'
                          )
                        }
                      />
                    </td>
                    <td>
                      {variation.num_of_malicious_clients > 0 ? (
                        <Form.Select
                          size="sm"
                          value={variation.attack_type || 'gaussian_noise'}
                          onChange={e =>
                            handleFieldChange(variation.id, 'attack_type', e.target.value)
                          }
                        >
                          {ATTACKS.map(attack => (
                            <option key={attack} value={attack}>
                              {attack === 'gaussian_noise' ? 'Gaussian Noise' : 'Label Flipping'}
                            </option>
                          ))}
                        </Form.Select>
                      ) : (
                        <span className="text-muted small">-</span>
                      )}
                    </td>
                    <td>
                      {variation.num_of_malicious_clients > 0 &&
                      (variation.attack_type || 'gaussian_noise') === 'gaussian_noise' ? (
                        <Form.Control
                          type="number"
                          size="sm"
                          min="0"
                          value={variation.gaussian_noise_std || 75}
                          onChange={e =>
                            handleFieldChange(
                              variation.id,
                              'gaussian_noise_std',
                              parseInt(e.target.value)
                            )
                          }
                        />
                      ) : (
                        <span className="text-muted small">-</span>
                      )}
                    </td>
                    <td>
                      {variation.aggregation_strategy_keyword === 'trust' ? (
                        <Form.Control
                          type="number"
                          size="sm"
                          step="0.01"
                          min="0"
                          max="1"
                          value={variation.trust_threshold || 0.15}
                          onChange={e =>
                            handleFieldChange(
                              variation.id,
                              'trust_threshold',
                              parseFloat(e.target.value)
                            )
                          }
                        />
                      ) : (
                        <span className="text-muted small">-</span>
                      )}
                    </td>
                    <td>
                      {variation.aggregation_strategy_keyword === 'trust' ? (
                        <Form.Control
                          type="number"
                          size="sm"
                          step="0.01"
                          min="0"
                          max="1"
                          value={variation.beta_value || 0.75}
                          onChange={e =>
                            handleFieldChange(
                              variation.id,
                              'beta_value',
                              parseFloat(e.target.value)
                            )
                          }
                        />
                      ) : (
                        <span className="text-muted small">-</span>
                      )}
                    </td>
                    <td>
                      {(variation.aggregation_strategy_keyword === 'trust' ||
                        variation.aggregation_strategy_keyword === 'pid') &&
                      variation.remove_clients === 'true' ? (
                        <Form.Control
                          type="number"
                          size="sm"
                          min="1"
                          value={variation.begin_removing_from_round || 2}
                          onChange={e =>
                            handleFieldChange(
                              variation.id,
                              'begin_removing_from_round',
                              parseInt(e.target.value)
                            )
                          }
                        />
                      ) : (
                        <span className="text-muted small">-</span>
                      )}
                    </td>
                    <td>
                      {variation.aggregation_strategy_keyword === 'pid' ? (
                        <Form.Control
                          type="number"
                          size="sm"
                          step="0.01"
                          min="0"
                          value={variation.Kp || 1}
                          onChange={e =>
                            handleFieldChange(variation.id, 'Kp', parseFloat(e.target.value))
                          }
                        />
                      ) : (
                        <span className="text-muted small">-</span>
                      )}
                    </td>
                    <td>
                      {variation.aggregation_strategy_keyword === 'pid' ? (
                        <Form.Control
                          type="number"
                          size="sm"
                          step="0.01"
                          min="0"
                          value={variation.Ki || 0.05}
                          onChange={e =>
                            handleFieldChange(variation.id, 'Ki', parseFloat(e.target.value))
                          }
                        />
                      ) : (
                        <span className="text-muted small">-</span>
                      )}
                    </td>
                    <td>
                      {variation.aggregation_strategy_keyword === 'pid' ? (
                        <Form.Control
                          type="number"
                          size="sm"
                          step="0.01"
                          min="0"
                          value={variation.Kd || 0.05}
                          onChange={e =>
                            handleFieldChange(variation.id, 'Kd', parseFloat(e.target.value))
                          }
                        />
                      ) : (
                        <span className="text-muted small">-</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </div>

          {variations.length < 2 && (
            <div className="alert alert-warning mt-3">
              <i className="bi bi-exclamation-triangle me-2"></i>
              At least 2 strategies are required to create an experiment queue.
            </div>
          )}
        </>
      )}
    </div>
  );
}
