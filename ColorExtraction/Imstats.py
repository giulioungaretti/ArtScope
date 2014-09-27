from PIL import Image, ImageChops, ImageDraw
import palette as pl
import operator as op
import colorsys
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv,hsv_to_rgb
import numpy as np
from scipy import fftpack


class Imstat:

    def __init__(self,filename,ifFull=False):
        self.image = Image.open(filename)
        self.full = ifFull

    def get_pallette(self):
        return pl.extract_colors(self.image,hist=self.full)

    def get_rgb_stats(self):
        cnorm = tuple((255.0, 255.0, 255.0))
        colors = [tuple(map(op.div, i.value, cnorm)) for i in self.get_pallette().colors]
        sizes = [i.prominence * 100.0 for i in self.get_pallette().colors]

        return colors,sizes

    def get_hsv_values(self):
        array = np.asarray(self.image)
        arr = (array.astype(float))/255.0
        img_hsv = rgb_to_hsv(arr[...,:3])

        return img_hsv

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

    def plot_power_spectrum(self):

        arr = np.asarray(self.image.convert('L'))

        #ft = np.fft.fft2(arr)
        #ps = np.abs(ft)**2

        F1 = fftpack.fft2(arr)

        F2 = fftpack.fftshift(F1)

        #FreqCompRows = np.fft.fftfreq(F2.shape[0],d=2)
        #FreqCompCols = np.fft.fftfreq(F2.shape[1],d=2)

        #print FreqCompCols
        #print FreqCompRows

        # Calculate a 2D power spectrum
        psd2D = np.abs(F2)**2

        # Calculate the azimuthally averaged 1D power spectrum
        psd1D = azimuthalAverage(psd2D)

        # Now plot up both

        #plt.imshow( np.log10( arr ), cmap=plt.cm.Greys)
        #plt.show()

        #plt.imshow( np.log10( psd2D ))
        #plt.show()

        #plt.semilogy( psd1D )
        #plt.xlabel('Spatial Frequency')
        #plt.ylabel('Power Spectrum')

        return psd1D

    def plot_two2_hist(self):

        hsv = self.get_hsv_values()
        hist, xbins, ybins = np.histogram2d(hsv[0,:,:].ravel(),hsv[2,:,:].ravel(),[100,100],[[0,1.0],[0,1.0]])

        maxes = np.sort(hist.flatten())

        #plt.imshow(hist,interpolation = 'nearest',vmin = 0, vmax = hist.mean()*100)
        #plt.contourf(hist,levels=[1,3,5,10,50,100,200],origin='upper')

        H, V = np.mgrid[0:1:300j, 0:1:300j]
        S = np.ones_like(V)
        HSV = np.dstack((H, S, V))
        RGB_hsv = hsv_to_rgb(HSV)

        plt.imshow(RGB_hsv,extent=[0,1,0,1])

        #plt.contourf(hist,levels=[0,1],origin='upper',extent=[0,1,0,1],colors='white',alpha=1.0)
        plt.contourf(hist,levels=[1,1000],origin='upper',extent=[0,1,0,1],colors='white',alpha=0.5)
        plt.show()

    def plot_blocks(self):

        "Save palette as a PNG with labeled, colored blocks"
        output_filename = 'palette.png' 

        sizes = [i.prominence*2000.0 for i in self.get_pallette().colors]

        x_size = np.sum(sizes)
        y_size = np.max(sizes)

        size = (int(x_size), int(y_size))
        im = Image.new('RGBA', size)
        draw = ImageDraw.Draw(im)

        x_pos = 0.0
        for i, c in enumerate(self.get_pallette().colors):
            v = colorsys.rgb_to_hsv(*norm_color(c.value))[2]
            (x1, y1) = (x_pos+11, y_size)
            (x2, y2) = (x_pos+sizes[i], y_size-sizes[i])
            x_pos = x_pos + sizes[i]
            draw.rectangle([(x1, y1), (x2, y2)], fill=c.value)

        im.save(output_filename, "PNG")


def norm_color(c):
    r, g, b = c
    return r / 255.0, g / 255.0, b / 255.0

def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof
