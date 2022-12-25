import matplotlib.pyplot as plt
import numpy as np


def part1(img):
    image = np.dot(img[..., :3], [0.3, 0.59, 0.11])
    ImS = 55 + (220 - 55) * ((image-np.amin(image))/(np.amax(image)-np.amin(image)))**0.4
    neg = np.amax(ImS) - ImS

    plt.imshow(img)
    # plt.show()

    plt.imshow(image, cmap='gray')
    plt.title(f"Original, K={round((np.amax(image) - np.amin(image)) / 255, 4)}")
    # plt.show()

    plt.imshow(ImS, cmap='gray')
    plt.title(f"Result1, Lmax={np.amax(ImS)}, Lmin={np.amin(ImS)}, K={round((np.amax(ImS) - np.amin(ImS)) / 255, 4)}")
    # plt.show()

    plt.imshow(neg, cmap='gray')
    plt.title(f"Negative, Lmax={np.amax(neg)}, Lmin={np.amin(neg)}, K={round((np.amax(neg) - np.amin(neg)) / 255, 4)}")
    # plt.show()
    return ImS


def part2(img):
    count1, bins1 = np.histogram(img, 256)
    count, bins = np.histogram(img)
    cs = np.cumsum(count)
    print(f"Cumsum = {cs}")
    print(count)
    fig, ax = plt.subplots(3, 1)
    ax[0].hist(count, 10)
    ax[0].set_title('Histogram')
    ax[1].plot(cs, 'o')
    ax[1].plot(cs, color='red')
    ax[1].set_title('Cumulative func')
    ax[2].plot(count1)
    plt.show()


if __name__ == "__main__":
    # result1 = part1(plt.imread('img.png'))
    # part2(result1)
    part2(plt.imread('img.png'))
