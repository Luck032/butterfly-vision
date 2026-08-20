import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def cos_similarity(imageA, imageB):
    dot = np.dot(imageA, imageB)
    norm1 = np.linalg.norm(imageA)
    norm2 = np.linalg.norm(imageB)
    cosS = dot / (norm1 * norm2)
    return cosS


def prepareImage(image):
    v_tensor = torch.from_numpy(image).float() / 255.0
    v_tensor = v_tensor.permute(2, 0, 1)
    v_tensor = v_tensor.unsqueeze(0)
    return v_tensor


def main():
    work_dir = "./CV_dataset"

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()

    files = [f for f in os.listdir(work_dir) if f.lower().endswith((".jpg", ".jpeg"))]

    max_CS = -1
    ind_i = -1
    ind_j = -1

    features = []

    with torch.no_grad():
        for f in files:
            img = cv2.imread(os.path.join(work_dir, f), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            prep = prepareImage(img)
            feat = model(prep).detach().numpy()[0]
            features.append(feat)

    n = len(files)

    for i in range(n):
        for j in range(i + 1, n):
            temp_CS = cos_similarity(features[i], features[j])

            if temp_CS > max_CS:
                max_CS = temp_CS
                ind_i = i
                ind_j = j

    print(files[ind_i])
    print(files[ind_j])

    v1 = cv2.imread(os.path.join(work_dir, files[ind_i]), cv2.IMREAD_COLOR)
    v2 = cv2.imread(os.path.join(work_dir, files[ind_j]), cv2.IMREAD_COLOR)

    plt.figure(figsize=(8, 16))
    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(v1, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(v2, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    main()
