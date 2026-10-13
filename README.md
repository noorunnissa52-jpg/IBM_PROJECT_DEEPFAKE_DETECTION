# Deepfake (Real vs Fake) Image Detector

A PyTorch project that classifies an image as **REAL** or **FAKE** using transfer learning
(EfficientNet-B0 by default; ResNet18/50 also supported). Includes training, evaluation
(accuracy, precision/recall/F1, ROC-AUC, confusion matrix), single-image / folder
prediction with confidence, Grad-CAM heatmaps, and an optional Gradio web demo.

## 1. Setup
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
A GPU is optional but makes training much faster (it is used automatically if available).

## 2. Get data
Folder layout expected by the code:
```
data/
  train/{real,fake}/*.jpg
  val/{real,fake}/*.jpg
  test/{real,fake}/*.jpg       (optional, needed for evaluate.py)
```
Good public datasets:
| Dataset | Notes |
|---|---|
| **140k Real and Fake Faces** (Kaggle) | GAN faces vs real faces; already split into train/valid/test (rename `valid` -> `val`) |
| **CIFAKE** (Kaggle) | 32x32 real vs AI-generated images, easy to start with |
| **FaceForensics++ / DFDC / Celeb-DF** | Video deepfakes; extract face frames first |

If you only have two flat folders, run:
```bash
python split_dataset.py --real_dir path/to/real --fake_dir path/to/fake
```

**No dataset yet?** Verify the pipeline runs with a tiny synthetic one:
```bash
python make_demo_data.py
```
(This is only a smoke test; the images are synthetic and results mean nothing.)

## 3. Train
```bash
python train.py --epochs 10 --arch efficientnet_b0
# options: --arch resnet18|resnet50  --batch_size 64  --img_size 224  --lr 3e-4  --no_pretrained
```
The best model (by validation accuracy) is saved to `checkpoints/best.pt`.

## 4. Evaluate
```bash
python evaluate.py --ckpt checkpoints/best.pt
```
Prints a classification report + ROC-AUC and saves `outputs/confusion_matrix.png`, `outputs/roc_curve.png`.

## 5. Predict (real or fake?)
```bash
python predict.py --input path/to/image.jpg
python predict.py --input path/to/folder --gradcam     # also saves heatmaps to outputs/
```
Example output:
```
face1.jpg   -> FAKE ( 97.3%)  [fake=0.973, real=0.027]
```

## 6. Web demo (optional)
```bash
python app.py
```

## Project structure
```
config.py          paths + hyperparameters
dataset.py         ImageFolder loaders + transforms
model.py           backbone + classifier head, checkpoint helpers
train.py           training loop (AdamW + cosine LR)
evaluate.py        metrics and plots
predict.py         inference + Grad-CAM
app.py             Gradio demo
make_demo_data.py  synthetic smoke-test data
split_dataset.py   split flat folders into train/val/test
```

## Tips for better results / your report
- Split by identity/video, not randomly by frame, to avoid data leakage.
- Test on a *different* dataset/generator than you trained on; detectors often fail to generalize
  and showing this honestly is a strong point in a project report.
- Social-media compression, resizing and blur reduce accuracy; consider JPEG/blur augmentation if you
  need robustness.
- Try `--arch resnet50`, more epochs, or fine-tuning with a lower LR for extra accuracy.
- For video deepfakes: detect faces (e.g. MTCNN), classify each frame, then average the scores.

## Limitations
No detector is 100% reliable. Treat output as a probability, not proof.
