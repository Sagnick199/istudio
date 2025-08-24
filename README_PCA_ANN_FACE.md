
# PCA + ANN Face Recognition (Step-by-step)

This project implements **Eigenfaces (PCA)** for feature extraction and a simple **ANN (backprop from scratch)** for classification + **open-set imposter detection**.

> Requirements aligned with your assignment brief: Numpy/Scipy for linear algebra, OpenCV for images. (Matplotlib for plots, ReportLab to render the final PDF report.)

---

## 1) Setup

```bash
# Create venv (optional)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install numpy scipy opencv-python matplotlib reportlab
```

> Internet is required once to download the dataset automatically. Alternatively, download it manually from the assignment link and place `dataset.zip` inside `dataset/`.

- Dataset link (as per assignment):
  - https://github.com/robaita/introduction_to_machine_learning/blob/main/dataset.zip

---

## 2) Run End-to-End

```bash
python pca_ann_face_recognition.py --dataset_root dataset --output_dir outputs --img_h 64 --img_w 64 --epochs 60 --lr 0.01
```

What this does:
- Downloads and extracts the dataset (if missing)
- Loads and preprocesses face images
- Computes PCA eigenfaces using surrogate covariance
- Trains an ANN on eigenface signatures
- Varies `k` (number of eigenfaces) and produces **Accuracy vs k** plot
- Holds out a subset of subjects as **imposters** (open-set) and estimates detection rate
- Exports `outputs/Face_Recognition_PCA_ANN_Report.pdf` with results + plots

---

## 3) Parameters

- `--k_values 10,20,30,40,50`  (optional; default tries a reasonable range)
- `--train_ratio 0.6`          (as required: 60% training, 40% testing for enrolled subjects)
- `--imposter_subject_frac 0.2` (fraction of subjects held out entirely as imposters)
- `--epochs 60` `--lr 0.01`    (ANN training settings)

---

## 4) Outputs

Inside `outputs/` you will get:
- `accuracy_vs_k.png`
- `eigenfaces_k{k}.png` (montage of top eigenfaces)
- `results.json` (all metrics)
- `Face_Recognition_PCA_ANN_Report.pdf` (final report with metrics & plots)

---

## 5) Notes

- ANN is implemented **from scratch** (NumPy) to adhere closely to the "allowed libraries" spirit.
- Imposter detection uses a **softmax confidence threshold** calibrated on a validation split.
- You can change image size via `--img_h/--img_w`.
- Code keeps randomness controlled via `--seed`.

