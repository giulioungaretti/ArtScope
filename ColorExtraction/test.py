import Imstats as img
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

import colorsys


im = img.Imstat('../imgs/copenhagen-painting-thumbnail.jpg')

a = im.get_rgb_stats()

#im = img.Imstat('../imgs/test.jpg')

#im = img.Imstat('../imgs/copenhagen-painting-thumbnail.jpg')
#im2 = img.Imstat('../imgs/copenhagen-infrared.jpg')
#im3 = img.Imstat('../imgs/copenhagen-xray.jpg')

im.plot_power_spectrum()
#im2.plot_power_spectrum()
#im3.plot_power_spectrum()
plt.show()

im.plot_hsv()
plt.show()

im.plot_rgb()
plt.show()

im.plot_blocks()

im.get_hsv_values()
im.plot_two2_hist()

H, V = np.mgrid[0:1:300j, 0:1:300j]
S = np.ones_like(V)
HSV = np.dstack((H, S, V))
RGB_hsv = hsv_to_rgb(HSV)

plt.imshow(RGB_hsv)
plt.show()

def get_hls(dat):
    hls_to_rgb = np.vectorize(colorsys.hls_to_rgb)

    h, l, s = np.rollaxis(dat, axis=-1)
    r, g, b = hls_to_rgb(h, l, s)
    arr = np.dstack((r, g, b))

    #c = colorsys.hls_to_rgb(dat)
    #c = np.array(c)
    #c = c.swapaxes(0,2)
    return arr


HSV_l = np.dstack((H, V, S))
print HSV_l.shape
RGB_hls = get_hls(HSV_l)

print RGB_hls.shape

plt.imshow(RGB_hls)
plt.show()
#Pretty plotting
# ----------------------------------------------------

def adjust_spines(ax,spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',10)) # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def define_aspect(extent,target_aspect):
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    return (x_range/y_range)*target_aspect
        
# ----------------------------------------------------

vis = img.Imstat('../imgs/copenhagen-painting-thumbnail.jpg')
uv = img.Imstat('../imgs/copenhagen-uv-thumbnail.jpg')
ir = img.Imstat('../imgs/copenhagen-infrared-thumbnail.jpg')

vis_hsv = vis.get_hsv_hist()
uv_hsv = uv.get_hsv_hist()
ir_gs = ir.get_gs_hist()

#Rough physical wavelength (in nm)
#Flipped as HSV goes from warm to cold
vis_range = np.linspace(390.0, 750.0, 256)[::-1]
#ir_range = np.linspace(1000.0, 1200.0, 256)[::-1]
ir_range = np.linspace(750.0, 1000.0, 256)[::-1]
uv_range = np.linspace(300.0, 390.0, 256)[::-1]


V, H = np.mgrid[0.75:0.55:10j, 0:1:300j]
S = np.ones_like(V)
HSV = np.dstack((H, S, V))
RGB = hsv_to_rgb(HSV)

a = np.outer(np.arange(0,1,0.01),np.ones(10)).T

fig,ax = plt.subplots(3, 1, figsize=(10, 4),sharex=True,sharey=False)

ax[0].imshow(a, origin="lower", extent=[300, 1000, 0, 1], aspect=10,cmap='spectral')
ax[0].xaxis.set_major_locator(plt.NullLocator())
ax[0].yaxis.set_major_locator(plt.NullLocator())
adjust_spines(ax[0],[])
ax[0].set_ylim(0,1)
ax[0].set_axis_off()

ax[1].imshow(np.fliplr(RGB), origin="lower", extent=[390, 750, 0, 1], aspect=10,cmap='spectral')

ax[1].set_ylim(0,1)
ax[1].set_axis_off()

ax[2].fill_between(vis_range, 0, vis_hsv[0,:],lw=0,color='white',edgecolor='white',alpha=0.8)
ax[2].fill_between(ir_range, 0, ir_gs,lw=0,color='white',edgecolor='white',alpha=0.8)
ax[2].fill_between(uv_range, 0, uv_hsv[0,:],lw=0,color='white',edgecolor='white',alpha=0.8)

# ax[2].fill_between(vis_range, 0, vis_hsv[2,:],lw=0.0,color='gray',alpha=0.5)
# ax[2].fill_between(ir_range, 0, ir_gs,lw=0,color='gray')
# ax[2].fill_between(uv_range, 0, uv_hsv[2,:],lw=0.0,color='gray',alpha=0.5)

ax[2].axes.get_yaxis().set_visible(False)
ax[2].axes.get_xaxis().set_visible(False)
adjust_spines(ax[2],[])

ax[2].set_xlim(300,1000)
#plt.ylim(0,5000)

fig.subplots_adjust(wspace=0,hspace=-0.5)

plt.savefig('demo.pdf', transparent=True, bbox_inches='tight',dpi=600)
plt.show()

