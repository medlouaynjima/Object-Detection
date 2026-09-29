# Object Detection Project

This project implements an object detection pipeline based on a custom dataset and a Faster R-CNN model from `torchvision`.

## Project goal

The goal is to detect objects in images by learning bounding boxes and class labels from annotated data.

## Project structure

- `Projet_Deep_learning_3_1_1_2_1.ipynb` : experimental notebook used for preprocessing, augmentation and training.
- `train.py` : cleaner training script.
- `requirements.txt` : Python dependencies.
- `.gitignore` : excludes local data and generated files.

## Dataset structure

The training code expects the following structure:

```text
project/
├── data/
│   ├── images/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   └── annotations/
│       ├── image_001.txt
│       ├── image_002.txt
│       └── ...
└── train.py
```

Each annotation file should contain one bounding box per line in this format:

```text
xmin,ymin,xmax,ymax,class_id
```

or

```text
(xmin,ymin),(xmax,ymax),class_id
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
python train.py --data-dir data --epochs 10 --batch-size 4 --lr 0.005
```

## Notes

- This project was developed as a deep learning object detection prototype.
- The notebook contains exploration and augmentation experiments, while the Python script provides a cleaner training entry point.
- For large datasets, it is recommended to store the data outside the Git repository and connect it through a local path.

## Recommended next steps

- clean the annotation format consistently;
- split data into train/validation/test;
- add mean average precision (mAP) evaluation;
- add prediction visualization after training;
- save checkpoints and best model weights.

## License

This project is intended for educational and personal use.
