import Imstats as img
import matplotlib.pyplot as plt

im = img.Imstat('../imgs/copenhagen-painting-thumbnail.jpg')

im.plot_hsv()
plt.show()

im.plot_rgb()
plt.show()