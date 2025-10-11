// Tooltip definitions for metrics and column names in CSV data

export const METRIC_TOOLTIPS = {
  // Confusion Matrix Components
  tp: 'True Positives (TP): Cases correctly identified as positive class',
  tn: 'True Negatives (TN): Cases correctly identified as negative class',
  fp: 'False Positives (FP): Negative cases incorrectly classified as positive (Type I error)',
  fn: 'False Negatives (FN): Positive cases incorrectly classified as negative (Type II error)',
  mean_tp: 'Average True Positives across all federated learning rounds',
  mean_tn: 'Average True Negatives across all federated learning rounds',
  mean_fp: 'Average False Positives across all federated learning rounds',
  mean_fn: 'Average False Negatives across all federated learning rounds',

  // Performance Metrics
  accuracy:
    'Accuracy: Ratio of correct predictions to total predictions. Formula: (TP + TN) / (TP + TN + FP + FN)',
  precision:
    'Precision: Ratio of true positives to all positive predictions. Formula: TP / (TP + FP). Measures exactness.',
  recall:
    'Recall (Sensitivity, TPR): Ratio of true positives to all actual positives. Formula: TP / (TP + FN). Measures completeness. Also called Sensitivity or True Positive Rate.',
  f1: 'F1 Score: Harmonic mean of precision and recall. Formula: 2 × (Precision × Recall) / (Precision + Recall). Balanced performance metric.',
  f1_score:
    'F1 Score: Harmonic mean of precision and recall. Formula: 2 × (Precision × Recall) / (Precision + Recall). Balanced performance metric.',
  mean_accuracy: 'Mean accuracy across all federated learning rounds',
  mean_precision: 'Mean precision across all federated learning rounds',
  mean_recall: 'Mean recall across all federated learning rounds',
  mean_f1: 'Mean F1 score across all federated learning rounds',

  // Loss Metrics
  loss: 'Loss function value quantifying model error. Lower values indicate better model fit.',
  train_loss: 'Loss computed on training data during local client training',
  test_loss: 'Loss computed on validation/test data to measure generalization',
  mean_loss: 'Average loss across all federated learning rounds',
  rmse: 'Root Mean Squared Error: Square root of mean squared differences between predicted and actual values. Penalizes larger errors more heavily.',
  mae: 'Mean Absolute Error: Average of absolute differences between predicted and actual values. Less sensitive to outliers than RMSE.',

  // Round and Timing
  round:
    'Communication round number in federated learning. Each round involves: client training → update aggregation → global model distribution.',
  num_rounds: 'Total number of communication rounds in this FL simulation',
  timestamp: 'Timestamp when this metric was recorded during execution',
  duration: 'Elapsed time for this operation in seconds',

  // Client Information
  client_id: 'Unique identifier for a federated learning client (simulated device/participant)',
  num_clients: 'Total number of clients participating in the federated learning simulation',
  num_samples: "Number of data samples in this client's local dataset",
  sample_size: 'Size of the dataset sample used for training or evaluation',

  // Strategy and Configuration
  strategy:
    'Federated aggregation algorithm (e.g., FedAvg, Krum, Trimmed Mean) used to combine client model updates',
  attack_type: 'Type of Byzantine attack being simulated (e.g., Gaussian noise, label flipping)',
  defense_type: 'Defense mechanism applied to mitigate Byzantine attacks',
  poisoned_clients:
    'Number of malicious/Byzantine clients injecting corrupted data or model updates',

  // Dataset Information
  dataset: 'Dataset name used for federated learning training and evaluation',
  train_size: 'Number of samples in the training dataset partition',
  test_size: 'Number of samples in the test/validation dataset partition',

  // Model Information
  model: 'Neural network architecture used for learning (e.g., CNN, ResNet, MLP)',
  learning_rate:
    'Learning rate (α) controlling the step size in gradient descent optimization. Common ranges: 0.001-0.1 for SGD, 0.0001-0.01 for Adam.',
  batch_size:
    'Number of training samples processed in one forward/backward pass before updating model weights. Often powers of 2 (32, 64, 128) for GPU efficiency.',
  epochs: 'Number of complete passes through the entire local training dataset',

  // Advanced Metrics (Removal/Detection)
  removal_threshold:
    'Threshold value for Byzantine client removal. Clients with scores exceeding this are excluded from aggregation.',
  removal_accuracy:
    'Accuracy of Byzantine client detection mechanism in identifying malicious vs benign clients',
  removal_precision:
    'Precision of Byzantine detection: ratio of correctly identified malicious clients to all clients marked for removal',
  removal_recall:
    'Recall of Byzantine detection: ratio of correctly identified malicious clients to all actual malicious clients',
  removal_f1: 'F1 score of Byzantine detection mechanism, balancing precision and recall',
  removal_criterion:
    'Score quantifying client trustworthiness or deviation. Used to determine removal eligibility.',
  absolute_distance:
    'Distance metric measuring deviation of client update from the global model or expected behavior',
  aggregation_participation:
    "Binary indicator of whether this client's update was included in aggregation (1=included, 0=excluded)",
  aggregated_loss: 'Loss value of the global model after aggregating client updates',
  score_calculation_time_nanos:
    'Computation time in nanoseconds for calculating client removal scores',
  average_accuracy: 'Mean accuracy computed across all participating clients in this round',

  // Special Columns
  'round #': 'Federated learning round number',
};

/**
 * Get tooltip text for a column name
 * @param {string} columnName - The column name to get tooltip for
 * @returns {string|null} Tooltip text or null if not found
 */
export function getMetricTooltip(columnName) {
  if (!columnName) return null;

  // Normalize column name: lowercase, remove _history suffix, clean up
  let normalized = columnName
    .toLowerCase()
    .replace(/_history$/i, '') // Remove _history suffix
    .replace(/\s+/g, '_') // Replace spaces with underscores
    .trim();

  // Handle client-specific metrics (e.g., client_0_loss_history -> loss)
  const clientMetricMatch = normalized.match(/^client_\d+_(.+)$/);
  if (clientMetricMatch) {
    normalized = clientMetricMatch[1];
  }

  return METRIC_TOOLTIPS[normalized] || null;
}
