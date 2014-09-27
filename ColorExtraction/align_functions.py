'''

HaloStack v0.1 by Panu Lahtinen (April 2012)

For more information, see http://www.iki.fi/~pnuu/halostack/

'''

import numpy as np
import math
import time

def align(a, b, area=None):

    ay,ax = a.shape
    by,bx = b.shape
    by,bx = by/2,bx/2

    x1,y1 = 0,0
    y2,x2 = a.shape

    if area is not None:
        x1,y1,x2,y2 = (area)

    # Check area validity
    if x1 < bx:
        x1 = bx
    if y1 < by:
        y1 = by
    if x2 >= ax-bx:
        x2 = ax-bx-1
    if y2 >= ay-by:
        y2 = ay-by-1

    best_x,best_y = None,None

    best_sqdif = 2**64

    for x in range(x1,x2):
        xr = range(x-bx,x+bx+1)
        for y in range(y1,y2):
            sqdif = ((a[y-by:y+by+1,xr]-b)**2).sum()
            if sqdif < best_sqdif:
                best_sqdif,best_x,best_y = sqdif,x,y

    # Calculate correlation coeff for the best fit
    best_fit_data = a[best_y-by:best_y+by+1,best_x-bx:best_x+bx+1].flatten()
    b_flat = b.flatten()
    best_corr = np.corrcoef(best_fit_data,b_flat)**2

    return (best_corr[0,1],best_x,best_y)


def calc_align_area(tx,ty,w,h):
    n_w = w-int(math.fabs(tx)) # width of the portion to be moved
    n_h = h-int(math.fabs(ty)) # height of the portion to be moved
    
    # Calculate the corner indices of the area to be moved
    if tx < 0:
        n_x1,n_x2 = 0,n_w
        o_x1,o_x2 = -1*tx,-1*tx+n_w
    else:
        n_x1,n_x2 = tx,tx+n_w
        o_x1,o_x2 = 0,n_w
    if ty < 0:
        n_y1,n_y2 = 0,n_h
        o_y1,o_y2 = -1*ty,-1*ty+n_h
    else:
        n_y1,n_y2 = ty,ty+n_h
        o_y1,o_y2 = 0,n_h

    n_x = range(n_x1,n_x2)
    n_y = range(n_y1,n_y2)
    o_x = range(o_x1,o_x2)
    o_y = range(o_y1,o_y2)

    return (n_x1,n_x2,n_y1,n_y2,o_x1,o_x2,o_y1,o_y2)
