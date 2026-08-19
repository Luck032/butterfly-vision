import cv2
import os

def main():
    work_dir = "./CV_dataset"
    img = cv2.imread(os.path.join(work_dir, "image_0.jpg"), cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_image, (5, 5), 0)

    cv2.imshow("Original", img)
    cv2.imshow("Grayscale", gray_image)
    cv2.imshow("Gaussian blur", blurred_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()