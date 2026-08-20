# Buttervly Computer Vision Assignment

## Overview
The assignment is intended to check basic Computer Vision (CV) skills. The assignment is consisted of two tasks:
 - Task 1: basic operations using OpenCV library,
 - Task 2: finding the pair of original and one copied and modified image in dataset containing 1000 images of butterflies. 

## Project Structure
The repository containe 5 Python scripts as proposed solutions for two given tasks. The specific task for which is scrit intended is indicated in the name of file. Please note that for the Task 1, there is only one avaliable scripts, as the soluction is straight forwarad, as for the Task 2 there are foud different implementations, where specific implemented method is inicated as *task_2_xxxx.py* where *xxxx* indicate the name of method.
The repo contains following scripts:
- [`task_1.py`](src/task_1.py)
- [`task_2_correlation.py`](src/task_2_correlation.py)
- [`task_2_mse.py`](src/task_2_mse.py)
- [`task_2_cosine.py`](src/task_2_cosine.py)
- [`task_2_resnet.py`](src/task_2_resnet.py)

The two distictive pathways could be indentified in proposed solutions for Task 2. The first caclulated similarites on pixel level using standard mathods from statistics and patter recognition (*e.g.,* zero-lag cross-correlation, mean square error, cosine similarity of two flattened vectors), while second pathway utilized the feature extraction embedded in convolutional layers of pretrained ResNet18, where extracted 512 features are compared between images using cosine similarity.

## Installation

Clone the repository:

```bash
git clone <repo-url>
cd butterfly-vision
````

Install the project dependencies:

```bash
uv sync
```

This will create the virtual environment and install the dependencies defined in `pyproject.toml` using the versions from `uv.lock`.

Run a script with:

```bash
uv run python src/task_1.py
```

For example:

```bash
uv run python src/task_2_cosine.py
uv run python src/task_2_mse.py
uv run python src/task_2_correlation.py
uv run python src/task_2_resnet.py
```

If `uv` is not installed, install it first by following the official uv installation instructions.

The `.venv` directory is not included in the repository. The environment can be recreated from:

* `pyproject.toml`
* `uv.lock`

## Results

The statitical and patter recognition methods unanimously indicated files (image_153.jpg, image_489.jpg) as original and copied modified pair, while method using pretrained ResNet18 in combination with cosine similarity indicate (image_78.jpg, image_704.jpg) as original and copied modified pair. Using majority voting mechanisms, the final answer is (image_153.jpg, image_489.jpg).  
