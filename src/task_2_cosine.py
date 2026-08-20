import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def cos_similarity(imageA, imageB):
    dot = np.dot(imageA, imageB)
    norm1 = np.linalg.norm(imageA)
    norm2 = np.linalg.norm(imageB)
    cosS = dot / (norm1 * norm2)
    return cosS


def main():
    work_dir = "./CV_dataset"
    out_dir = "./downsampled_images"

    os.makedirs(out_dir, exist_ok=True)

    for filename in os.listdir(work_dir):
        if filename.lower().endswith((".jpg", ".jpeg")):
            full_path = os.path.join(work_dir, filename)
            img = cv2.imread(full_path, cv2.IMREAD_COLOR)
            blured_img = cv2.GaussianBlur(img, (5, 5), 0)
            downsampled_image = cv2.resize(blured_img, (128, 128))
            cv2.imwrite(os.path.join(out_dir, filename), downsampled_image)

    files = [f for f in os.listdir(out_dir) if f.lower().endswith((".jpg", ".jpeg"))]
    N = len(files)

    vectors = []
    for f in files:
        img = cv2.imread(os.path.join(out_dir, f), cv2.IMREAD_GRAYSCALE)
        vec = img.flatten().astype("float32") / 255.0
        vectors.append(vec)

    max_cos = -1
    ind_i = -1
    ind_j = -1

    for i in range(N):
        for j in range(i + 1, N):
            v1 = vectors[i]
            v2 = vectors[j]

            temp_cos_similarity = cos_similarity(v1, v2)

            if temp_cos_similarity > max_cos:
                max_cos = temp_cos_similarity
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
