import { Form } from 'react-bootstrap';
import { SelectField } from '../FormFields/SelectField';
import { NumberField } from '../FormFields/NumberField';
import { STRATEGIES } from '@constants/strategies';
import { DATASETS } from '@constants/datasets';

export function CommonSettings({ config, onChange }) {
  const datasetSource = config.dataset_source || 'local';
  const isHuggingFace = datasetSource === 'huggingface';

  return (
    <>
      <SelectField
        name="aggregation_strategy_keyword"
        label="Aggregation Strategy"
        value={config.aggregation_strategy_keyword}
        onChange={onChange}
        options={STRATEGIES}
        tooltip="Algorithm for aggregating client model updates. FedAvg uses weighted averaging; robust strategies (Krum, Trimmed Mean, etc.) provide Byzantine fault tolerance against malicious clients."
        required
      />

      <Form.Group className="mb-3">
        <Form.Label>Dataset Source</Form.Label>
        <div>
          <Form.Check
            inline
            type="radio"
            name="dataset_source"
            id="dataset-source-local"
            label="Local"
            value="local"
            checked={datasetSource === 'local'}
            onChange={onChange}
          />
          <Form.Check
            inline
            type="radio"
            name="dataset_source"
            id="dataset-source-hf"
            label="HuggingFace"
            value="huggingface"
            checked={datasetSource === 'huggingface'}
            onChange={onChange}
          />
        </div>
        <Form.Text className="text-muted">
          Choose between local datasets or HuggingFace Hub datasets
        </Form.Text>
      </Form.Group>

      {!isHuggingFace ? (
        <SelectField
          name="dataset_keyword"
          label="Dataset (Local)"
          value={config.dataset_keyword}
          onChange={onChange}
          options={DATASETS}
          tooltip="Dataset for training and evaluation. Medical imaging datasets (PneumoniaMNIST, BloodMNIST) are commonly used in healthcare FL research. MNIST/CIFAR are standard benchmarks."
          required
        />
      ) : (
        <>
          <Form.Group className="mb-3">
            <Form.Label>HuggingFace Dataset</Form.Label>
            <Form.Control
              type="text"
              name="hf_dataset_name"
              value={config.hf_dataset_name || ''}
              onChange={onChange}
              placeholder="e.g., stanfordnlp/sst2"
              required
            />
            <Form.Text className="text-muted">
              HuggingFace dataset identifier (e.g., stanfordnlp/sst2, ag_news, glue/mnli)
            </Form.Text>
          </Form.Group>

          <SelectField
            name="model_type"
            label="Model Type"
            value={config.model_type || 'cnn'}
            onChange={onChange}
            options={[
              { value: 'cnn', label: 'CNN (Image Classification)' },
              { value: 'transformer', label: 'Transformer (Text Classification)' },
            ]}
            tooltip="CNN for image datasets, Transformer for text datasets"
            required
          />

          <SelectField
            name="partitioning_strategy"
            label="Data Partitioning"
            value={config.partitioning_strategy || 'iid'}
            onChange={onChange}
            options={[
              { value: 'iid', label: 'IID (Independent and Identically Distributed)' },
              { value: 'dirichlet', label: 'Dirichlet (Non-IID with α parameter)' },
              { value: 'pathological', label: 'Pathological (K classes per client)' },
            ]}
            tooltip="Distribution of data across clients. IID = balanced, Dirichlet = realistic non-IID, Pathological = extreme heterogeneity"
          />

          {config.partitioning_strategy === 'dirichlet' && (
            <NumberField
              name="partitioning_params.alpha"
              label="Dirichlet Alpha (α)"
              value={config.partitioning_params?.alpha || 0.5}
              onChange={e => {
                const alpha = parseFloat(e.target.value);
                onChange({
                  target: {
                    name: 'partitioning_params',
                    value: { ...config.partitioning_params, alpha },
                  },
                });
              }}
              min={0.01}
              max={10}
              step={0.01}
              tooltip="Controls heterogeneity: lower α = more non-IID (0.1 = high, 0.5 = moderate, 1.0 = mild)"
            />
          )}

          {config.partitioning_strategy === 'pathological' && (
            <NumberField
              name="partitioning_params.num_classes_per_partition"
              label="Classes per Client"
              value={config.partitioning_params?.num_classes_per_partition || 2}
              onChange={e => {
                const num_classes_per_partition = parseInt(e.target.value);
                onChange({
                  target: {
                    name: 'partitioning_params',
                    value: { ...config.partitioning_params, num_classes_per_partition },
                  },
                });
              }}
              min={1}
              max={10}
              tooltip="Number of classes each client receives (lower = more extreme non-IID)"
            />
          )}
        </>
      )}

      <NumberField
        name="num_of_rounds"
        label="Number of Rounds"
        value={config.num_of_rounds}
        onChange={onChange}
        min={1}
        tooltip="Communication rounds between server and clients. Start with 2-5 for quick tests, use 10+ for real experiments."
        required
      />

      <NumberField
        name="num_of_clients"
        label="Number of Clients"
        value={config.num_of_clients}
        onChange={onChange}
        min={1}
        tooltip="Total participating devices/clients. More clients = more realistic but slower simulation."
        required
      />
    </>
  );
}
