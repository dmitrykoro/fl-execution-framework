import { SelectField } from '../FormFields/SelectField';
import { NumberField } from '../FormFields/NumberField';
import { STRATEGIES } from '@constants/strategies';
import { DATASETS } from '@constants/datasets';

export function CommonSettings({ config, onChange }) {
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

      <SelectField
        name="dataset_keyword"
        label="Dataset (Local)"
        value={config.dataset_keyword}
        onChange={onChange}
        options={DATASETS}
        tooltip="Dataset for training and evaluation. Medical imaging datasets (PneumoniaMNIST, BloodMNIST) are commonly used in healthcare FL research. MNIST/CIFAR are standard benchmarks."
        required
      />

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
