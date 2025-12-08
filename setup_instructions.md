# PhysicsDeform - Environment Setup

This document provides step-by-step instructions to set up the environment required to run the PhysicsDeform pipeline, including dependencies for CoTracker and PythonCDT.

**Tested Configuration**: Ubuntu 18.04 · CUDA 10.0 · Python 3.7 · GPU: 2080 Ti / 2070 / V100 / Titan V / Titan XP

---

## 1 · Create a Conda environment

We recommend Python 3.7 to ensure compatibility with TensorFlow 1.15.0

```bash
conda create -n physics_deform python=3.7 -y
conda activate physics_deform
```

---

## 2 · Install core Python dependencies

Install all required packages using `pip`:

```bash
pip install -r requirements.txt
```

---

## 3 · Install CoTracker (editable mode)

> ⚠️ **Important:** To avoid path conflicts, clone **outside** the main `PhysiXDeform_demo/` folder. However, the model file `cotracker2.pth` must still be placed in `PhysiXDeform_demo/cotracker/models/`.

```bash
cd ..
git clone https://github.com/facebookresearch/co-tracker
cd co-tracker
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard
cd ..
```

> 📥 **Note:** Please download `cotracker2.pth` from the following link as stated in the CoTracker repository:  
> https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth  
> and store it in:  
> `PhysiXDeform_demo/cotracker/models/cotracker2.pth`

---

## 4 · Install PythonCDT

This provides mesh processing utilities required for graph-based prediction.

> ⚠️ **Important:** Clone this repository **outside** the main `PhysiXDeform_demo/` folder to avoid naming or path conflicts.

```bash
cd ..
git clone --recurse-submodules https://github.com/artem-ogre/PythonCDT.git
cd PythonCDT
pip install .
pytest ./cdt_bindings_test.py  # Optional test
cd ..
```

---

## ✅ Done!

You can now run the training, evaluation, and visualisation code in the `PhysiXDeform_demo/` repository as described in the main README.
