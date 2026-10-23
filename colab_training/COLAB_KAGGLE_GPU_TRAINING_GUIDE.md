# Google Colab & Kaggle GPU Training Guide for Padel AI Coach

This guide explains **when**, **why**, and **how** to use Free Cloud GPUs (Google Colab or Kaggle) to train deep vision models for your Padel AI Coach without putting any strain on your laptop.

---

## 1. When Do You Need Colab / Kaggle?

| Component | Where to Run? | Why? |
| :--- | :--- | :--- |
| **Shot Classifier & Biomechanics AI** | **Local Laptop (Already Built!)** | Trained in ~20 seconds using XGBoost on extracted kinematics. Zero laptop strain. |
| **Padel AI Coach Rule & Drill Engine** | **Local Laptop (Already Built!)** | Evaluates elbow angles, contact point timing, knee flexion, and tactical positioning in milliseconds. |
| **Deep Learning Vision Detector (Ball & Pose on Unseen Videos)** | **Google Colab / Kaggle (Free T4 GPU)** | Fine-tuning YOLO or TrackNet on thousands of 1080p video frames requires CUDA tensor cores. Free on Colab (15–30 mins), but would take 48+ hours and overheat a laptop CPU. |

---

## 2. Quick-Start: Running The Colab Notebook

We have generated a complete, turnkey notebook for you:
📁 [`c:\Padel\colab_training\Padel_Vision_Colab_Trainer.ipynb`](file:///c:/Padel/colab_training/Padel_Vision_Colab_Trainer.ipynb)

### Step 1: Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and upload `Padel_Vision_Colab_Trainer.ipynb`.
3. In the top menu, select: **Runtime** -> **Change runtime type** -> **T4 GPU** (100% Free).

### Step 2: Run The Cells
The notebook automatically:
1. Verifies the T4 GPU (`nvidia-smi`).
2. Installs `ultralytics`, `opencv-python`, and `torch`.
3. Sets up the PadelTracker100 dataset.
4. Trains **YOLOv8-Padel-Ball** & **YOLOv8-Padel-Pose** with GPU acceleration.
5. Exports the trained models as **Quantized ONNX** weights (which run 4x faster on your laptop's Intel CPU!).
6. Automatically downloads `padel_ball_best.onnx` and `padel_pose_best.onnx`.

### Step 3: Put Weights in Your Local Workspace
Move the downloaded model files into your local project folder:
```
c:\Padel\models\padel_ball_best.onnx
c:\Padel\models\padel_pose_best.onnx
```

Once placed there, your local Padel AI Coach can process **any brand new unannotated padel match video** fully automatically!

---

## 3. Alternative: Running on Kaggle (30 Hours Free GPU / Week)

If you prefer Kaggle:
1. Go to [Kaggle](https://www.kaggle.com/) -> **New Notebook**.
2. Click **File** -> **Import Notebook** and select `Padel_Vision_Colab_Trainer.ipynb`.
3. In the right panel under **Notebook options** -> **Accelerator**, choose **GPU T4 x2** or **GPU P100**.
4. Click **Run All**.
5. Download the model artifacts from the **Output** tab.
