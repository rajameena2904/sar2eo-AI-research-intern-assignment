# Multi-Sensor Satellite Image Translation: Sentinel-1 SAR to Sentinel-2 EO

This repository contains the end-to-end applied research pipeline and production deployment assets developed for the GalaxEye Satellite AI Research Intern Technical Assignment. The project implements a conditional Generative Adversarial Network (Pix2Pix) framework engineered to map single-channel 8-bit grayscale Synthetic Aperture Radar (SAR) backscatter matrices into high-fidelity 3-channel Electro-Optical (EO) RGB spatial maps.

The task addresses a fundamentally ill-posed inverse problem: translating cloud-penetrating, speckled microwave backscatter into human-interpretable optical profiles while overcoming information asymmetry between sensor modalities.

---

##  Repository Architecture

```text
├── README.md                      # Technical execution and configuration manual
├── infer.py                       # Production CLI inference script (Strict I/O contract)
├── sar2eo-training.ipynb          # Documented training notebook with metric tracking
├── sar2eo Report.pdf              # Finalized Technical Research Report & Time Log
└── artifacts/                     # Evaluation visual plots & tracking assets
    ├── loss_curve.png             # Optimization curves (Gen vs Disc loss)
    ├── loss_metrics.csv           # Tabular training loss matrix over 10 epochs
    └── sar_optical_analysis_pair.png # Qualitative SAR-to-EO translation sample

```

---

##  Environmental Setup & Dependencies

### Requirements

* Python `>= 3.10`

* Pinned primary package dependencies:


* `torch==2.1.2`
* `torchvision==0.16.2`
* `numpy==1.26.2`
* `opencv-python==4.8.1.78`
* `matplotlib==3.8.2`
* `pyyaml==6.0.1`



### Step-by-Step Environment Setup

Execute the following venv initialization framework to establish your runtime workspace:

```bash
# Create a localized virtual environment
python -m venv galaxeye_env

# Activate the virtual environment environment
# On Windows:
galaxeye_env\Scripts\activate
# On Linux/MacOS:
source galaxeye_env/bin/activate

# Upgrade pip and install pinned package matrices
pip install --upgrade pip
pip install torch torchvision numpy opencv-python matplotlib pyyaml

```

---

##  Dataset Structure

This project utilizes a scene-isolated configuration derived from the **SEN1-2 dataset** (Sentinel-1 SAR VV + Sentinel-2 RGB patches). The localized data loader pipeline expects your directories to follow the standardized layout structure printed below:

```text
data/
├── train/
│   ├── sar/         # 256x256 single-channel Sentinel-1 patches
│   └── eo/          # 256x256 corresponding 3-channel RGB patches
└── val/
    ├── sar/
    └── eo/

```

---

##  Pipeline Execution Instructions

### 1. Training From Scratch

To execute the baseline conditional optimization pipeline loop from scratch utilizing your hyperparameter configuration profiles, run:

```bash
# Executing training notebook or training module hook
python -c "import notebook; notebook.notebookapp.main()" 
# If running your script module equivalent:
# python train.py --config config.yaml

```

Note: Training progression logs raw loss variations per-epoch directly to `artifacts/loss_metrics.csv`.

### 2. Running Inference (`infer.py` Command Contract)

The inference component complies strictly with the requested technical input/output infrastructure contract. It scales an arbitrary target directory of single-channel SAR images and writes matching 3-channel generated optical outputs.

```bash
python infer.py --input_dir /path/to/test_sar_folder --output_dir /path/to/output_eo_folder --weights /path/to/pix2pix_generator.pth

```

* Parameter Interfaces:


* `--input_dir`: Folder containing test SAR patches (supported: 8-bit, dB-scaled, min-max normalized PNG).


* `--output_dir`: Location where translated optical EO images preserve original filenames.


* `--weights`: Local target file path pointing to your downloaded model weights tensor file.





### 3. Metric Evaluation

To run quantitative metric scoring scripts comparing your model's translated EO output maps directly against matching Ground-Truth scenes, execute:

```bash
python eval.py --pred_dir /path/to/output_eo_folder --gt_dir /path/to/val/eo

```

---

##  Verified Model Weights Checkpoint

To keep this repository lightweight and performant as required by the assignment constraints, the trained model parameters are hosted externally.

* **Model Checkpoint Link (`pix2pix_generator.pth`):** [Access Public Google Drive Weights Here](https://drive.google.com/file/d/1Fx_2A_qIwF9SBKiwbuoPwrQPM5iSvMQB/view?usp=drive_link)

* Note: Sharing permissions are explicitly opened to "Anyone with the link can view/download" to guarantee friction-free evaluation.



---

##  Quantitative Optimization Tracking

The model was optimized over 10 epochs under strict computing constraints ($\le$16 GB VRAM).

### Reported Metrics Summary Matrix

Below is the tabulated summary tracking performance metrics for validation set:

| Partition Split | LPIPS ↓ | FID ↓ | SSIM ↑ | PSNR ↑ |
| --- | --- | --- | --- | --- |
| **Validation Set** | `0.2413` | `42.18` | `0.0855` | `11.43 dB` |


The training equilibrium established between the Generator's structural reconstruction and the Discriminator's spatial critique is visualized below:

Figure 1: Tabulated loss progression highlighting steady reconstruction stabilization over 10 epochs.

The complete tabular data documenting every step of the optimization run is saved locally at [`artifacts/loss_metrics.csv`](https://drive.google.com/file/d/1CRgGP96WnsF8W_ylnLpI2edop52TTEiU/view?usp=sharing).

---

##  Citation & References

* Schmitt, M., Hughes, L. H., & Zhu, X. X. (2018). "SEN1-2 — A Dataset for Deep Learning in SAR-Optical Data Fusion". arXiv:1807.01569.
* Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). "Image-to-Image Translation with Conditional Adversarial Networks". CVPR.

---
