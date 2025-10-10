import { useState } from 'react';
import { Accordion, Table, OverlayTrigger, Tooltip } from 'react-bootstrap';
import OutlineButton from '@components/common/Button/OutlineButton';
import { formatStrategyName, formatDatasetName, formatAttackName } from '@constants/strategyLabels';

export function ConfigTab({ config }) {
  const [showRawJSON, setShowRawJSON] = useState(false);
  const cfg = config.shared_settings || config;

  const ConfigRow = ({ label, value, tooltip }) => (
    <tr>
      <td className="fw-semibold" style={{ width: '40%' }}>
        {label}
        {tooltip && (
          <OverlayTrigger placement="right" overlay={<Tooltip>{tooltip}</Tooltip>}>
            <span style={{ cursor: 'help', marginLeft: '4px', fontSize: '0.9rem' }}>ℹ️</span>
          </OverlayTrigger>
        )}
      </td>
      <td>{value}</td>
    </tr>
  );

  if (showRawJSON) {
    return (
      <>
        <div className="d-flex justify-content-between align-items-center mb-3">
          <h5 className="mb-0">Raw Configuration JSON</h5>
          <OutlineButton onClick={() => setShowRawJSON(false)}>View Human-Readable</OutlineButton>
        </div>
        <pre
          style={{
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            fontFamily: 'monospace',
            fontSize: '0.875rem',
            color: '#E6E1E5', // Light text for dark mode
          }}
        >
          {JSON.stringify(config, null, 2)}
        </pre>
      </>
    );
  }

  return (
    <>
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h5 className="mb-0">Configuration</h5>
        <OutlineButton onClick={() => setShowRawJSON(true)}>View Raw JSON</OutlineButton>
      </div>

      <Accordion defaultActiveKey="0">
        <Accordion.Item eventKey="0">
          <Accordion.Header>Common Settings</Accordion.Header>
          <Accordion.Body>
            <Table size="sm" className="mb-0">
              <tbody>
                <ConfigRow
                  label="Aggregation Strategy"
                  value={formatStrategyName(cfg.aggregation_strategy_keyword)}
                  tooltip="Aggregation algorithm for combining client updates. fedavg is simplest (average), others provide Byzantine robustness."
                />
                <ConfigRow
                  label="Dataset"
                  value={formatDatasetName(cfg.dataset_keyword)}
                  tooltip="Training dataset for federated learning. Medical datasets (pneumoniamnist, bloodmnist) are common for healthcare FL."
                />
                <ConfigRow
                  label="Model Type"
                  value={cfg.model_type === 'cnn' ? 'CNN' : cfg.model_type || 'CNN'}
                />
                <ConfigRow label="Use LLM" value={cfg.use_llm === 'true' ? 'Yes' : 'No'} />
                <ConfigRow
                  label="Number of Rounds"
                  value={cfg.num_of_rounds}
                  tooltip="Communication rounds between server and clients. Start with 2-5 for quick tests, use 10+ for real experiments."
                />
                <ConfigRow
                  label="Number of Clients"
                  value={cfg.num_of_clients}
                  tooltip="Total participating devices/clients. More clients = more realistic but slower simulation."
                />
                <ConfigRow
                  label="Batch Size"
                  value={cfg.batch_size}
                  tooltip="Number of samples per training batch. Larger = faster but more memory. 32 is standard."
                />
                <ConfigRow
                  label="Client Epochs"
                  value={cfg.num_of_client_epochs}
                  tooltip="Training passes each client performs locally before sending updates. 1 epoch is fastest."
                />
              </tbody>
            </Table>
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="1">
          <Accordion.Header>Attack Configuration</Accordion.Header>
          <Accordion.Body>
            <Table size="sm" className="mb-0">
              <tbody>
                <ConfigRow label="Malicious Clients" value={cfg.num_of_malicious_clients || 0} />
                <ConfigRow label="Attack Type" value={formatAttackName(cfg.attack_type)} />
                {cfg.attack_type === 'gaussian_noise' && (
                  <>
                    <ConfigRow label="Gaussian Noise Mean" value={cfg.gaussian_noise_mean ?? 0} />
                    <ConfigRow label="Gaussian Noise Std" value={cfg.gaussian_noise_std ?? 1} />
                    <ConfigRow label="Attack Ratio" value={cfg.attack_ratio ?? 0.5} />
                  </>
                )}
              </tbody>
            </Table>
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="2">
          <Accordion.Header>Strategy-Specific Parameters</Accordion.Header>
          <Accordion.Body>
            <Table size="sm" className="mb-0">
              <tbody>
                {cfg.aggregation_strategy_keyword === 'trust' && (
                  <>
                    <ConfigRow
                      label="Begin Removing From Round"
                      value={cfg.begin_removing_from_round || 1}
                      tooltip="Round number when trust-based client filtering starts. Earlier = more aggressive filtering."
                    />
                    <ConfigRow
                      label="Trust Threshold"
                      value={cfg.trust_threshold || 0.5}
                      tooltip="Minimum trust score (0-1) for client inclusion. Lower = more permissive, higher = stricter filtering."
                    />
                    <ConfigRow
                      label="Beta Value"
                      value={cfg.beta_value || 0.9}
                      tooltip="Exponential decay factor for trust score updates. Higher (closer to 1) = trust changes slowly."
                    />
                    <ConfigRow
                      label="Number of Clusters"
                      value={cfg.num_of_clusters || 1}
                      tooltip="Number of client clusters for trust grouping. More clusters = finer-grained trust analysis."
                    />
                  </>
                )}
                {['pid', 'pid_scaled', 'pid_standardized'].includes(
                  cfg.aggregation_strategy_keyword
                ) && (
                  <>
                    <ConfigRow
                      label="Number of Std Deviations"
                      value={cfg.num_std_dev || 2.0}
                      tooltip="Threshold for outlier detection. Updates beyond this many standard deviations are filtered."
                    />
                    <ConfigRow
                      label="Kp (Proportional Gain)"
                      value={cfg.Kp || 1.0}
                      tooltip="PID controller proportional term. Controls reaction to current error. Higher = more aggressive correction."
                    />
                    <ConfigRow
                      label="Ki (Integral Gain)"
                      value={cfg.Ki || 0.1}
                      tooltip="PID controller integral term. Eliminates steady-state error by accumulating past errors."
                    />
                    <ConfigRow
                      label="Kd (Derivative Gain)"
                      value={cfg.Kd || 0.01}
                      tooltip="PID controller derivative term. Predicts future error based on rate of change. Reduces overshoot."
                    />
                  </>
                )}
                {['multi-krum', 'krum', 'multi-krum-based'].includes(
                  cfg.aggregation_strategy_keyword
                ) && (
                  <ConfigRow
                    label="Krum Selections"
                    value={cfg.num_krum_selections || 5}
                    tooltip="Number of closest clients to aggregate. Lower = more Byzantine robustness but less data diversity."
                  />
                )}
                {cfg.aggregation_strategy_keyword === 'trimmed_mean' && (
                  <ConfigRow
                    label="Trim Ratio"
                    value={cfg.trim_ratio || 0.1}
                    tooltip="Fraction of extreme values to remove from both ends (0-0.5). Higher = more aggressive outlier filtering."
                  />
                )}
                {![
                  'trust',
                  'pid',
                  'pid_scaled',
                  'pid_standardized',
                  'multi-krum',
                  'krum',
                  'multi-krum-based',
                  'trimmed_mean',
                ].includes(cfg.aggregation_strategy_keyword) && (
                  <tr>
                    <td colSpan="2" className="text-muted fst-italic">
                      No strategy-specific parameters for{' '}
                      {formatStrategyName(cfg.aggregation_strategy_keyword)}
                    </td>
                  </tr>
                )}
              </tbody>
            </Table>
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="3">
          <Accordion.Header>Resource & Output Settings</Accordion.Header>
          <Accordion.Body>
            <Table size="sm" className="mb-0">
              <tbody>
                <ConfigRow
                  label="Training Device"
                  value={cfg.training_device?.toUpperCase() || 'CPU'}
                />
                <ConfigRow label="CPUs per Client" value={cfg.cpus_per_client || 1} />
                <ConfigRow label="GPUs per Client" value={cfg.gpus_per_client || 0.0} />
                <ConfigRow
                  label="Training Subset Fraction"
                  value={cfg.training_subset_fraction || 0.9}
                />
                <ConfigRow label="Show Plots" value={cfg.show_plots === 'true' ? 'Yes' : 'No'} />
                <ConfigRow label="Save Plots" value={cfg.save_plots === 'true' ? 'Yes' : 'No'} />
                <ConfigRow label="Save CSV" value={cfg.save_csv === 'true' ? 'Yes' : 'No'} />
                <ConfigRow
                  label="Preserve Dataset"
                  value={cfg.preserve_dataset === 'true' ? 'Yes' : 'No'}
                />
                <ConfigRow
                  label="Remove Clients"
                  value={cfg.remove_clients === 'true' ? 'Yes' : 'No'}
                />
                <ConfigRow label="Strict Mode" value={cfg.strict_mode === 'true' ? 'Yes' : 'No'} />
              </tbody>
            </Table>
          </Accordion.Body>
        </Accordion.Item>

        <Accordion.Item eventKey="4">
          <Accordion.Header>Flower Framework Settings</Accordion.Header>
          <Accordion.Body>
            <Table size="sm" className="mb-0">
              <tbody>
                <ConfigRow label="Min Fit Clients" value={cfg.min_fit_clients || 5} />
                <ConfigRow label="Min Evaluate Clients" value={cfg.min_evaluate_clients || 5} />
                <ConfigRow label="Min Available Clients" value={cfg.min_available_clients || 5} />
                <ConfigRow
                  label="Evaluate Metrics Aggregation"
                  value={cfg.evaluate_metrics_aggregation_fn || 'weighted_average'}
                />
              </tbody>
            </Table>
          </Accordion.Body>
        </Accordion.Item>

        {cfg.use_llm === 'true' && (
          <Accordion.Item eventKey="5">
            <Accordion.Header>LLM Settings</Accordion.Header>
            <Accordion.Body>
              <Table size="sm" className="mb-0">
                <tbody>
                  <ConfigRow label="LLM Model" value={cfg.llm_model || 'BiomedBERT'} />
                  <ConfigRow
                    label="Fine-tuning Method"
                    value={cfg.llm_finetuning === 'lora' ? 'LoRA' : 'Full'}
                  />
                  <ConfigRow
                    label="Task"
                    value={cfg.llm_task === 'mlm' ? 'MLM (Masked Language Modeling)' : cfg.llm_task}
                  />
                  <ConfigRow label="Chunk Size" value={cfg.llm_chunk_size || 256} />
                  {cfg.llm_task === 'mlm' && (
                    <ConfigRow label="MLM Probability" value={cfg.mlm_probability || 0.15} />
                  )}
                  {cfg.llm_finetuning === 'lora' && (
                    <>
                      <ConfigRow label="LoRA Rank" value={cfg.lora_rank || 16} />
                      <ConfigRow label="LoRA Alpha" value={cfg.lora_alpha || 32} />
                      <ConfigRow label="LoRA Dropout" value={cfg.lora_dropout || 0.1} />
                      <ConfigRow
                        label="LoRA Target Modules"
                        value={
                          Array.isArray(cfg.lora_target_modules)
                            ? cfg.lora_target_modules.join(', ')
                            : 'query, value'
                        }
                      />
                    </>
                  )}
                </tbody>
              </Table>
            </Accordion.Body>
          </Accordion.Item>
        )}
      </Accordion>
    </>
  );
}
