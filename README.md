# PhysicsDeform

**PhysicsDeform** is a framework for deformation-aware prediction and reconstruction in endoscopic environments.  
This repository includes implementations for both **temporally extended deformation forecasting** and **cross-scene adaptation** tasks.  
Demonstrations are provided using sequences from the **Hamlyn Silicon Phantom** dataset (Trajectories 11 and 12).

---

## 🔧 Environment Setup

Please refer to the [`setup_instructions.md`](./setup_instructions.md) file for full setup instructions.

Dependencies are listed in [`requirements.txt`](./requirements.txt). Additional modules — including **CoTracker** and **PythonCDT** — are installed via the same instructions.

---

## 📂 Datasets

You can download the required dataset anonymously from:

> [https://anonymous.4open.science/r/datasets-PhysiXDeform/](https://anonymous.4open.science/r/datasets-PhysiXDeform/)

After downloading, place the `stereo_hamlyn/` folder inside `./stereo_datasets/` so that the final structure looks like:

```
./stereo_datasets/
└── stereo_hamlyn/
    └── rectified11000/
    └── ...
```

This includes:

- Hamlyn Silicon Phantom — Trajectory 11
- Hamlyn Silicon Phantom — Trajectory 12

---

## 🛠️ Dataset Generation

To generate data in the correct format for training and evaluation, run:

```bash
python hamlyn_dataset.py
```

This script will generate mesh sequences and associated metadata under `hamlyn_data/`.

---

## 🚀 Training

Training is handled by `auto_run.py`. No external configuration files are required.

### ➔ Base Training (Temporally Extended Forecasting)

Run:

```bash
python auto_run.py
```

Set the training mode in the script:

```python
mode = "base"
```

Checkpoints are saved automatically. Training can be safely interrupted and resumed.

---

### ➔ Fine-Tuning (Cross-Scene Adaptation)

To perform fine-tuning:

1. Create the output directory:

```bash
mkdir -p ./result/hamlyn/trajectory_12/patched_0.005_0.01_loss_700k_steps/
```

2. Copy the base model checkpoint:

```bash
cp -r ./result/hamlyn/trajectory_11/patched_0.005_0.01_loss_400k_steps/patched_0.005_0.01_loss_400k_ckpts \
      ./result/hamlyn/trajectory_12/patched_0.005_0.01_loss_700k_steps/patched_0.005_0.01_loss_700k_ckpts
```

3. Run training with:

```bash
python auto_run.py
```

And set:

```python
mode = "finetune"
```

---

## 📊 Evaluation

Evaluation is performed with `auto_eval.py`. Run:

```bash
python auto_eval.py
```

Inside the script, specify the evaluation type:

```python
evaluation_mode = "base"      # for evaluating base models
# or
evaluation_mode = "finetune"  # for fine-tuned models
```

This script computes Chamfer Distance and threshold metrics and saves predicted meshes and logs in the output directory.

---
