import { Form, OverlayTrigger, Tooltip } from 'react-bootstrap';
import { POPULAR_DATASETS } from '@constants/datasets';
import { useDatasetValidation } from '@hooks/useDatasetValidation';

export function DatasetSelector({ config, onChange }) {
  const datasetValidation = useDatasetValidation(config);

  return (
    <>
      <Form.Group className="mb-3">
        <Form.Label>
          Dataset <span className="text-danger">*</span>{' '}
          <OverlayTrigger
            placement="right"
            overlay={
              <Tooltip>
                HuggingFace dataset name (e.g., "cifar10", "mnist"). Framework validates dataset
                exists and checks Flower Datasets compatibility. Popular choices in dropdown.
              </Tooltip>
            }
          >
            <span style={{ cursor: 'help' }}>ℹ️</span>
          </OverlayTrigger>
        </Form.Label>
        <Form.Control
          type="text"
          name="dataset_keyword"
          value={config.dataset_keyword}
          onChange={onChange}
          list="popular-datasets"
          required
          placeholder="Select from suggestions or type dataset name..."
        />
        <datalist id="popular-datasets">
          {POPULAR_DATASETS.map(d => (
            <option key={d.value} value={d.value}>
              {d.label}
            </option>
          ))}
        </datalist>
        {datasetValidation.loading && (
          <Form.Text className="text-muted d-block mt-1">⏳ Checking dataset...</Form.Text>
        )}
        {datasetValidation.valid === false && (
          <Form.Text className="text-danger d-block mt-1">
            ❌ Dataset not found: {datasetValidation.error}
          </Form.Text>
        )}
        {datasetValidation.valid === true && !datasetValidation.compatible && (
          <Form.Text className="text-warning d-block mt-1">
            ⚠️ Dataset found but may not be compatible with Flower Datasets
          </Form.Text>
        )}
        {datasetValidation.valid === true && datasetValidation.compatible && (
          <Form.Text className="text-success d-block mt-1">
            ✅ Valid dataset ({datasetValidation.info?.num_examples?.toLocaleString()} examples,
            splits: {datasetValidation.info?.splits?.join(', ')})
            {datasetValidation.info?.key_features?.length > 0 && (
              <span> • Fields: {datasetValidation.info.key_features.join(', ')}</span>
            )}
          </Form.Text>
        )}
      </Form.Group>

      <Form.Group className="mb-3">
        <Form.Label>
          Partitioning Strategy <span className="text-danger">*</span>{' '}
          <OverlayTrigger
            placement="right"
            overlay={
              <Tooltip>
                How to distribute data across clients. IID = balanced/uniform. Dirichlet = realistic
                heterogeneous distribution (tune α: lower = more heterogeneous). Pathological =
                extreme non-IID (each client gets limited label classes).
              </Tooltip>
            }
          >
            <span style={{ cursor: 'help' }}>ℹ️</span>
          </OverlayTrigger>
        </Form.Label>
        <Form.Select
          name="partitioning_strategy"
          value={config.partitioning_strategy}
          onChange={onChange}
        >
          <option value="iid">IID (Balanced)</option>
          <option value="dirichlet">Dirichlet (Heterogeneous)</option>
          <option value="pathological">Pathological (Extreme Non-IID)</option>
        </Form.Select>
        {datasetValidation.valid &&
          datasetValidation.info?.has_label === false &&
          (config.partitioning_strategy === 'dirichlet' ||
            config.partitioning_strategy === 'pathological') && (
            <Form.Text className="d-block mt-1" style={{ color: '#d97706', fontWeight: '500' }}>
              ⚠️ Warning: This dataset may not work with {config.partitioning_strategy} partitioning
              (no "label" field detected). Consider using IID partitioning or switching to a
              classification dataset.
            </Form.Text>
          )}
        {datasetValidation.valid &&
          datasetValidation.info?.has_label === false &&
          config.partitioning_strategy === 'iid' && (
            <Form.Text className="d-block mt-1" style={{ color: '#0ea5e9', fontWeight: '500' }}>
              ℹ️ This dataset works best with IID partitioning since no label field was detected.
              Good choice!
            </Form.Text>
          )}
      </Form.Group>

      {config.partitioning_strategy === 'dirichlet' && (
        <Form.Group className="mb-3">
          <Form.Label>
            Dirichlet Alpha (α){' '}
            <OverlayTrigger
              placement="right"
              overlay={
                <Tooltip>
                  Controls data heterogeneity. Lower α = more heterogeneous (realistic). α=0.1 =
                  very heterogeneous, α=0.5 = moderate, α=10.0 = nearly IID. Typical research
                  values: 0.1-1.0.
                </Tooltip>
              }
            >
              <span style={{ cursor: 'help' }}>ℹ️</span>
            </OverlayTrigger>
          </Form.Label>
          <Form.Control
            type="number"
            name="dirichlet_alpha"
            value={config.dirichlet_alpha}
            onChange={onChange}
            step="0.1"
            min="0.01"
          />
        </Form.Group>
      )}

      {config.partitioning_strategy === 'pathological' && (
        <Form.Group className="mb-3">
          <Form.Label>
            Num Classes Per Client{' '}
            <OverlayTrigger
              placement="right"
              overlay={
                <Tooltip>
                  Number of unique label classes each client will have access to. Lower = more
                  extreme non-IID (e.g., 1-2 classes per client).
                </Tooltip>
              }
            >
              <span style={{ cursor: 'help' }}>ℹ️</span>
            </OverlayTrigger>
          </Form.Label>
          <Form.Control
            type="number"
            name="num_classes_per_client"
            value={config.num_classes_per_client}
            onChange={onChange}
            min="1"
          />
        </Form.Group>
      )}
    </>
  );
}
