# Butterfly Computer Vision Assignment

## Overview

This assignment is intended to demonstrate basic Computer Vision (CV) skills. It consists of two tasks:

- **Task 1:** basic image processing operations using the OpenCV library.
- **Task 2:** finding the pair consisting of an original image and its modified duplicate in a dataset containing 1000 butterfly images.

## Project Structure

The repository contains five Python scripts with proposed solutions for the two tasks. The task corresponding to each script is indicated by its filename.

For **Task 1**, only one implementation is provided because the required image processing operations are straightforward.

For **Task 2**, four different implementations are provided. The method used by each script is indicated by the filename `task_2_xxxx.py`, where `xxxx` represents the applied method.

The repository contains the following scripts:

- [`task_1.py`](src/task_1.py)
- [`task_2_correlation.py`](src/task_2_correlation.py)
- [`task_2_mse.py`](src/task_2_mse.py)
- [`task_2_cosine.py`](src/task_2_cosine.py)
- [`task_2_resnet.py`](src/task_2_resnet.py)

Two different approaches were explored for Task 2.

The first approach compares images at the pixel level using standard statistical and pattern recognition methods:

- zero-lag correlation,
- mean squared error (MSE),
- cosine similarity between flattened image vectors.

The second approach uses a pretrained **ResNet18** as a feature extractor. The final fully connected classification layer is removed, resulting in a 512-dimensional feature representation for each image. Cosine similarity is then calculated between the extracted feature vectors. 
**Note:** This approach was included as an exploratory feature-based alternative to the pixel-level methods. Since the extracted embeddings represent higher-level visual characteristics rather than exact pixel correspondence, it was treated as a complementary analysis rather than the primary duplicate-detection method.

## Installation

Clone the repository:

```bash
git clone <repo-url>
cd butterfly-vision
```

Install the project dependencies:

```bash
uv sync
```

This will create the virtual environment and install the dependencies defined in `pyproject.toml` using the versions specified in `uv.lock`.

The `.venv` directory is not included in the repository. The environment can be recreated using:

- `pyproject.toml`
- `uv.lock`

If `uv` is not installed, it should be installed before running the commands above.

## Dataset

Extract the provided dataset and place the `CV_dataset` directory in the project root:

```text
butterfly-vision/
├── CV_dataset/
├── src/
│   ├── task_1.py
│   ├── task_2_correlation.py
│   ├── task_2_cosine.py
│   ├── task_2_mse.py
│   └── task_2_resnet.py
├── pyproject.toml
├── README.md
└── uv.lock
```

## Running the Scripts

Task 1 can be run with:

```bash
uv run python src/task_1.py
```

The different Task 2 approaches can be run independently:

```bash
uv run python src/task_2_cosine.py
uv run python src/task_2_mse.py
uv run python src/task_2_correlation.py
uv run python src/task_2_resnet.py
```

## Results

All three pixel-based methods identified the same pair as the most likely original and modified duplicate:

```text
image_153.jpg
image_489.jpg
```

The pretrained ResNet18 feature-based approach identified a different pair:

```text
image_78.jpg
image_704.jpg
```

Since the cosine similarity, MSE, and zero-lag correlation methods independently identified the same pair, the final selected result is:

**`image_153.jpg` and `image_489.jpg`**