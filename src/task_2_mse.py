import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


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

    min_mse = np.inf
    ind_i = -1
    ind_j = -1

    for i, f1 in enumerate(files):
        v1 = cv2.imread(os.path.join(out_dir, f1), cv2.IMREAD_GRAYSCALE)

        for j, f2 in enumerate(files):
            if j <= i:
                continue

            v2 = cv2.imread(os.path.join(out_dir, f2), cv2.IMREAD_GRAYSCALE)

            curr_mse = mse(v1, v2)

            if curr_mse < min_mse:
                min_mse = curr_mse
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
