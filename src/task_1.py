import cv2
import os
import matplotlib.pyplot as plt


def main():
    work_dir = "./CV_dataset"
    img = cv2.imread(os.path.join(work_dir, "image_0.jpg"), cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_image, (5, 5), 0)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 1, 2)
    plt.imshow(blurred_img, cmap = 'gray')
    plt.show()

if __name__ == "__main__":
    main()