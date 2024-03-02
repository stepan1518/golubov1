import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(img, im_name='-'):
    cv2.imshow(im_name, np.uint8(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_hist(img):
    plt.hist(img.ravel(), bins=256, range=(0, 256))
    plt.title('Гистограмма распределения света на изображении')
    plt.show()

def salt_pepper(img, proc):
    salt = np.random.random(img.shape) <= proc / 2
    pepper = np.random.random(img.shape) <= proc / 2
    img_cpy = np.copy(img)
    img_cpy[salt] = 255
    img_cpy[pepper] = 0
    return img_cpy

img = cv2.imread('photo.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show_image(img, 'Image')

img = (img - img.min()) / (img.max() - img.min()) * 255
img = np.uint8(img)

show_hist(img)

s_p = salt_pepper(img, 0.01)
show_image(s_p, 'Salt&Pepper')

blur = cv2.blur(s_p, (3, 3))
show_image(blur, 'Blur filter')

sharp_matr = np.array([
    [-1, -2, -1],
    [-2, 22, -2],
    [-1, -2, -1]
])
sharp = cv2.filter2D(blur, -1, sharp_matr / np.sum(sharp_matr))
show_image(sharp, 'Sharp filter')

median = cv2.medianBlur(np.uint8(s_p), 3)
show_image(median, 'Median filter')