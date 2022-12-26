import matplotlib.pyplot as plt
import numpy as np
from random import gauss


def gaussNoiseImg(img):
    image = img.astype("float")
    noise_img = image + np.random.normal(0, 10/100, img.shape)
    plt.imshow(noise_img)
    plt.title("Additive noise")
    plt.show()
    return noise_img


def multiNoiseImg(img):
    image = img.astype("float")
    noise_img = image * np.random.normal(1, 10/100, img.shape)
    plt.imshow(noise_img)
    plt.title("Multiplicative noise")
    plt.show()
    return noise_img


def saltPepper(img):
    noise_img = np.copy(img)
    noise_img = forSaltPaper(img, noise_img, 0)
    noise_img = forSaltPaper(img, noise_img, 1)

    plt.imshow(noise_img)
    plt.title("Salt & Pepper")
    plt.show()
    return noise_img


def forSaltPaper(img, noise_img, i):
    d = 0.1 + (10 - 14) / 100
    num_salt = np.ceil(d * img.size * 0.5)
    x_y = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noise_img[x_y] = i
    return noise_img


def mean_filter(img):
    n = 3
    clean_img = mahotas.mean_filter(img, n)
    plt.imshow(clean_img)
    plt.title(f"Rezult, Average {n}x{n}")
    plt.show()


if __name__ == "__main__":
    gaussImg = gaussNoiseImg(plt.imread('img.png'))
    multiImg = multiNoiseImg(gaussImg)
    saltPepperImg = saltPepper(multiImg)
    mean_filter(saltPepper)
