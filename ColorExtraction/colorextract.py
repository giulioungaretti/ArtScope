import numpy as np
import palette as pl
import matplotlib.pyplot as plt
import operator as op
import colorsys

def load_img(filename,hist=False):

	if hist == False:
		res1 = pl.extract_colors(filename,hist=hist)

		c,s = get_colors_sizes(res1)

		res = c,s

	elif hist == True:
		res1,res2 = pl.extract_colors(filename,hist=hist)

		c,s = get_colors_sizes(res1)
		hc,hs = get_colors_sizes(res2)

		res = c,s,hc,hs

	return res

def get_colors_sizes(Pallette):

	cnorm = tuple((255.0,255.0,255.0))

	#Normalization for matplotlib convention (RGB scaled between 0-1)
	colors = [tuple(map(op.div,i.value,cnorm)) for i in Pallette.colors]

	#Normalization for matplotlib convention (Scaled between 0-100)
	sizes = [i.prominence*100.0 for i in Pallette.colors]

	return colors,sizes

colors,sizes,hist_colors,hist_size = load_img('../imgs/copenhagen-painting-thumbnail.jpg',hist=True)

#pl.save_size_palette_as_image("pallette.png",palt)

plt.pie(sizes, colors=colors,autopct='%1.1f%%', startangle=90)
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.show()

def get_hsv_h(fulldat):
    rgb = fulldat[0]
    #Enusure that RGB tuple is a float (required for conversion)
    return colorsys.rgb_to_hsv(rgb[0]*255.0,rgb[1]*255.0,rgb[2]*255.0)[0]

def get_hsv_v(fulldat):
    rgb = fulldat[0]
    #Enusure that RGB tuple is a float (required for conversion)
    return colorsys.rgb_to_hsv(rgb[0]*255.0,rgb[1]*255.0,rgb[2]*255.0)[2]

#sort colors and sizes together
hist_colorss, hist_sizes = zip(*sorted(zip(hist_colors, hist_size),
  key=get_hsv_v, reverse=True))

def hexencode(rgb):
    print rgb
    r=rgb[0]*255.0
    g=rgb[1]*255.0
    b=rgb[2]*255.0
    return '#%02x%02x%02x' % (r,g,b)

for idx, c in enumerate(hist_colorss):
    print 'hi'
    print c
    plt.bar(idx, hist_sizes[idx], color=hexencode(c),edgecolor=hexencode(c))

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator

x = [1,2,3,1,2,3]
y = [1,1,1,0,0,0]

def boxpie(W,c,x,y,ax=None):

    print len(W)

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())

    for i,w in enumerate(W):
    	print i
        color = c[i]
        size = w/12.0
        rect = Rectangle([x[i] - size / 2, y[i] - size / 2], size, size,
            facecolor=color, edgecolor=color)
        ax.add_patch(rect)
    ax.autoscale_view()

    # Reverse the yaxis limits
    ax.set_ylim(*ax.get_ylim()[::-1])

boxpie(sizes,colors,x,y)

plt.show()

[re,gr,bl] = pl.hist('../imgs/copenhagen-painting-thumbnail.jpg')
ir = pl.hist('../imgs/copenhagen-infrared-thumbnail.jpg')
[uva,uvb,uvc] = pl.hist('../imgs/copenhagen-uv-thumbnail.jpg')

for i in [re,gr,bl,ir,uva,uva,uvb,uvc]:
	plt.plot(i)
	plt.ylim(0,5000)
plt.show()