from PIL import Image, ImageChops, ImageDraw
from collections import Counter, namedtuple
import matplotlib.pyplot as plt

im = Image.open('../imgs/copenhagen-painting-thumbnail.jpg')
#data = im.getdata()
#dist = Counter(data)

w, h = im.size  
colors = im.getcolors(w*h) 

def hexencode(rgb):
    r=rgb[0]
    g=rgb[1]
    b=rgb[2]
    return '#%02x%02x%02x' % (r,g,b)

for idx, c in enumerate(colors):
    plt.bar(idx, c[0], color=hexencode(c[1]),edgecolor=hexencode(c[1]))

plt.show()