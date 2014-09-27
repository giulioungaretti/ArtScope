import numpy as np
import palette as pl
import operator as op
import colorsys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator

from matplotlib.colors import hsv_to_rgb

def load_img(filename,hist=False):

	#Simple return for 6 dominant colors from an image
	#Should be fast enough for database applications
	if hist == False:
		res1 = pl.extract_colors(filename,hist=hist)
		
		c,s = get_colors_sizes(res1)

		res = c,s

	#(Full) data return including output of above along with a wider color histogram
	elif hist == True:
		res1,res2 = pl.extract_colors(filename,hist=hist)

		c,s = get_colors_sizes(res1)
		hc,hs = get_colors_sizes(res2)

		res = c,s,hc,hs

	return res

#Wrapper to get colors/prominences out of Pallettes
def get_colors_sizes(Pallette):

	cnorm = tuple((255.0,255.0,255.0))

	#Normalization for matplotlib convention (RGB scaled between 0-1)
	colors = [tuple(map(op.div,i.value,cnorm)) for i in Pallette.colors]

	#Normalization for matplotlib convention (Scaled between 0-100)
	sizes = [i.prominence*100.0 for i in Pallette.colors]

	return colors,sizes

#Hue sorted
def get_hsv_h(fulldat):
    rgb = fulldat[0]
    #Enusure that RGB tuple is a float (required for conversion)
    return colorsys.rgb_to_hsv(rgb[0]*255.0,rgb[1]*255.0,rgb[2]*255.0)[0]

#Lightness sorted
def get_hsv_v(fulldat):
    rgb = fulldat[0]
    #Enusure that RGB tuple is a float (required for conversion)
    return colorsys.rgb_to_hsv(rgb[0]*255.0,rgb[1]*255.0,rgb[2]*255.0)[2]

def hexencode(rgb):
    print rgb
    r=rgb[0]*255.0
    g=rgb[1]*255.0
    b=rgb[2]*255.0
    return '#%02x%02x%02x' % (r,g,b)

def boxpie(c,W,ax=None):
    
    x = [1,2,3,1,2,3]
    y = [1,1,1,0,0,0]

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
    plt.show()

def pie(data,weight,ax=None):

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    ax.pie(weight,colors=data,autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.show()

def hist(data,weight,ax=None):

    fig = plt.figure()

    for idx, c in enumerate(data):
        plt.bar(idx, weight[idx], color=hexencode(c),edgecolor=hexencode(c))

    plt.show()

colors,sizes,hist_c,hist_s = load_img('../imgs/copenhagen-painting-thumbnail.jpg',hist=True)

#Sort histogram by hue (Sorting both prominence and colors)
hist_h_c, hist_h_s = zip(*sorted(zip(hist_c, hist_s),key=get_hsv_h, reverse=True))

pie(colors,sizes)
hist(hist_h_c,hist_h_s)
boxpie(colors,sizes)


# Visible light
# http://stackoverflow.com/questions/3407942/rgb-values-of-visible-spectrum
# 390 to 750 nm maps linearly to hue [0, 1]

# See more http://en.wikipedia.org/wiki/Dominant_wavelength
# http://www.cs.sun.ac.za/~lvzijl/courses/rw778/grafika/OpenGLtuts/Big/graphicsnotes006.html

# Wavelength to RGB/HSV
# http://www.efg2.com/Lab/ScienceAndEngineering/Spectra.htm
# http://www.cs.utah.edu/~bes/papers/color/
# http://markkness.net/colorpy/ColorPy.html
# http://www.noah.org/wiki/Wavelength_to_RGB_in_Python

# IR
# 1000nm to 2500nm
# http://www.webexhibits.org/feast/analysis/xrayinfrared.html

# UV
# http://conservationblog.hearstmuseum.dreamhosters.com/?p=674
# The UV section of the electromagnetic spectrum can be divided into three subcategories:
# long-wave UV between 320 and 400 nm (also known as UV-A), medium-wave UV between 280 and 320 nm (UV-B) 
# and short-wave UV between 180 and 280 nm (UV-C).

[re,gr,bl] = pl.hist('../imgs/copenhagen-painting-thumbnail.jpg')
ir = pl.hist('../imgs/copenhagen-infrared-thumbnail.jpg')
[uva,uvb,uvc] = pl.hist('../imgs/copenhagen-uv-thumbnail.jpg')

for i in [re,gr,bl,ir,uva,uva,uvb,uvc]:
	plt.plot(i)
	plt.ylim(0,5000)
plt.show()


rgb_array = np.array((re,gr,bl))

print rgb_array.shape
print rgb_array

fig,ax = plt.subplots(3,1,figsize=(10,5))

cl = ['r','g','b']

idx_array = np.arange(0,256,1)

for i,t in enumerate(cl):
    #ax[i].hist(rgb_array[i,:],bins=256,range=(0.0,255.0),histtype='stepfilled', color=cl[i], label=t)
    ax[i].fill_between(idx_array, 0, rgb_array[i,:],color=cl[i],label=t)
    ax[i].set_xlim(0,255)
    ax[i].legend()

plt.show()







hsv = pl.hist_hsv('../imgs/copenhagen-painting-thumbnail.jpg')

fig,ax = plt.subplots(4,1,figsize=(10,5))

types = ['hue','saturation','value']
cl = ['r','g','b']

V, H = np.mgrid[0.45:0.55:10j, 0:1:300j]
S = np.ones_like(V)
HSV = np.dstack((H,S,V))
RGB = hsv_to_rgb(HSV)

ax[0].imshow(RGB, origin="lower", extent=[0, 360, 0, 1], aspect=10)
ax[0].xaxis.set_major_locator(plt.NullLocator())
ax[0].yaxis.set_major_locator(plt.NullLocator())

for i,t in enumerate(types):
    ax[i+1].hist(hsv[...,i].flatten()*255,bins=256,range=(0.0,255.0),histtype='stepfilled', color=cl[i], label=t)
    ax[i+1].set_xlim(0,255)
    ax[i+1].legend()

plt.show()
