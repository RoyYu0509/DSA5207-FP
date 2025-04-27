

## 1. Install Required Packages

Make sure you have Python installed.
 Then install all required dependencies with a single command:

```bash
pip install -r requirements.txt
```

> `requirements.txt` already contains all necessary packages (e.g., `scipy`, `numpy`, `openai`, `torch`, `transformers`, etc.).

------

## 2. Navigate to the Project Directory

Make sure you are in the correct project directory, e.g.:

```bash
cd DSA5207-FP/prompt_generate_task
```

------

## 3. Run Dataset Generation Scripts

### ➡️ 3.1 Generate Spatial Reverse Dataset (One-Way)

Run the following command to generate a **single-direction** spatial relation dataset:

```bash
python -m scripts.reverse_experiments.generate_spatial_reverse_one_by_one \
--num_examples_per_group 30 \
--num_train_examples 60 \
--num_test_examples 30 \
--dataset_name spatial_generate_dataset_
```

- **Purpose:** Generate *A → B* (A2B) spatial relation training and testing data.
- **Dataset name prefix:** `spatial_generate_dataset_`

------

### ➡️ 3.2 Generate Spatial Reverse 1.5-Way Dataset

Run the following command to generate a **1.5-way** dataset (partial understanding of B→A):

```bash
scripts.reverse_experiments.generate_spatial_reverse_1_5_dataset \
--num_examples_per_group 20 \
--num_train_examples 40 \
--num_test_examples 20 \
--dataset_name spatial_generate_1_5_dataset_
```

- **Purpose:** Generate datasets where both A→B and B→A spatial relations are partially included.
- **Dataset name prefix:** `spatial_generate_1_5_dataset_`