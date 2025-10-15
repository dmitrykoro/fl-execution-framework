# Dataset Preparation Scripts

This directory contains scripts for downloading and preparing datasets for the FL execution framework.

## Available Datasets

### FEMNIST Non-IID

Downloads the FEMNIST dataset with natural non-IID partitioning by writer.

**Required for:** `convergence` preset (Non-IID Defense)

**To use this dataset, you must run:**

```bash
python scripts/download_femnist_niid.py
```

This will download and organize the dataset into `datasets/femnist_niid/`. Without running this script first, the `convergence` preset will fail with a missing dataset error.

**Options:**

- `--output-dir`: Target directory (default: `datasets/femnist_niid`)
- `--num-clients`: Number of clients to create (default: 100)
- `--min-samples`: Minimum samples per client (default: 50)

**Example:**

```bash
python scripts/download_femnist_niid.py --num-clients 50 --min-samples 100
```

## Creating New Dataset Scripts

Use `download_dataset_template.py` as a starting point for adding new datasets:

1. Copy the template: `cp scripts/download_dataset_template.py scripts/download_mydata.py`
2. Modify the download/processing logic for your dataset
3. Update the output directory structure to match: `datasets/mydataset/client_X/label_Y/`
4. Test with a small number of clients first
5. Document it in this README

**Requirements:**

- Install vision dependencies: `pip install flwr-datasets[vision]`
- Or install from project requirements: `./reinstall_requirements.sh`
