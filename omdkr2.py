import matplotlib.pyplot as plt
import numpy as np


def part2(img):
    histAndCumulative(img)
    Distension(img)
    Equalization(img)


def histAndCumulative(img):
    count, bins = np.histogram(img, 10)
    Cumsum = count.cumsum()
    c = 255 * Cumsum / Cumsum[-1]
    cumulative = c * max(count) / c.max()

    plt.subplot(2, 1, 1)
    plt.bar(bins[:-1], count)
    plt.title('Histogram')

    plt.subplot(2, 1, 2)
    plt.plot(cumulative / cumulative.max())
    plt.title('Cumulative Function')
    plt.show()


def Distension(img):
    distension = np.round((255.0 / (200 - 25 + 2)) * (img - 25 + 1)).astype(img.dtype)
    distension[img < 25] = 0
    distension[img > 200] = 255
    Lmax, Lmin = np.amax(distension), np.amin(distension)

    plt.imshow(distension, cmap='gray')
    plt.title(f"Distension, Lmax={Lmax}, Lmin={Lmin} K={round((Lmax - Lmin) / 255, 6)}")
    plt.show()


def Equalization(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    Cumsum = hist.cumsum()
    e = np.ma.masked_equal(Cumsum, 0)
    equalization = (e-e.min())*255/(e.max()-e.min())
    Cumsum = np.ma.filled(equalization, 0).astype('uint8')
    img2 = Cumsum[img]

    Lmax, Lmin = np.amax(img2), np.amin(img2)
    plt.imshow(img2, cmap='gray')
    plt.title(f"Equalization, Lmax={Lmax}, Lmin={Lmin} K={round((Lmax - Lmin) / 255, 4)}")
    plt.show()


if __name__ == "__main__":
    part2(plt.imread('img.jpg'))
