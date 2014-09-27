from PIL import Image, ImageChops, ImageDraw
import palette as pl
import operator as op
import colorsys

class Imstats:

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
        return pl.hist_hsv(self.hsv)

    def get_gs_hist(self):
        gs = self.image.convert('L')
        return pl.hist(gs)

    def get_hsv_stats(self):

        rgb = get

        return colorsys.rgb_to_hsv(rgb[0] * 255.0, rgb[1] * 255.0, rgb[2] * 255.0)
        

    

    # Normalization for matplotlib convention (RGB scaled between 0-1)
    

    # Normalization for matplotlib convention (Scaled between 0-100)
    sizes = [i.prominence * 100.0 for i in Pallette.colors]

    return colors, sizes