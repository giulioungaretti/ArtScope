from PIL import Image, ImageChops, ImageDraw
import palette as pl
import operator as op
import colorsys
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np

class Imstat:

    def __init__(self,filename,ifFull=False):
        self.image = Image.open(filename)
        self.full = ifFull

    def get_pallette(self):
        return pl.extract_colors(self.image,hist=self.full)

    def get_rgb_stats(self):
        cnorm = tuple((255.0, 255.0, 255.0))
        colors = [tuple(map(op.div, i.value, cnorm)) for i in self.get_pallette()]
        sizes = [i.prominence * 100.0 for i in self.get_pallette()]

        return colors,sizes

    def get_rgb_hist(self):
        [re, gr, bl] = pl.hist(self.image)
        rgb_array = np.array((re, gr, bl))
        return rgb_array

    def get_hsv_hist(self):
        return pl.hist_hsv(self.image)

    def get_gs_hist(self):
        gs = self.image.convert('L')
        return pl.hist(gs)

    def plot_rgb(self):

        fig, ax = plt.subplots(3, 1, figsize=(10, 5))

        cl = ['r', 'g', 'b']

        idx_array = np.arange(0, 256, 1)

        for i, t in enumerate(cl):

            ax[i].fill_between(idx_array, 0, self.get_rgb_hist()[i, :], color=cl[i], label=t)
            ax[i].set_xlim(0, 255)
            ax[i].legend()

        return fig

    def plot_hsv(self):
        fig, ax = plt.subplots(4, 1, figsize=(10, 5))

        types = ['hue', 'saturation', 'value']
        cl = ['r', 'g', 'b']

        V, H = np.mgrid[0.45:0.55:10j, 0:1:300j]
        S = np.ones_like(V)
        HSV = np.dstack((H, S, V))
        RGB = hsv_to_rgb(HSV)

        ax[0].imshow(RGB, origin="lower", extent=[0, 360, 0, 1], aspect=10)
        ax[0].xaxis.set_major_locator(plt.NullLocator())
        ax[0].yaxis.set_major_locator(plt.NullLocator())

        idx_array = np.arange(0, 256, 1)

        for i, t in enumerate(types):
            ax[i+1].fill_between(idx_array, 0, self.get_hsv_hist()[i,:], color=cl[i], label=t)
            ax[i+1].set_xlim(0, 255)
            ax[i+1].legend()

        return fig 
