from PIL import Image, ImageChops, ImageDraw

im1 = Image.open('../imgs/Output/copenhagen-painting.jpg').convert('LA') #Convert to grayscale
im2 = Image.open('../imgs/Output/copenhagen-infrared-ex.jpg').convert('LA')

out = ImageChops.difference(im1,im2) 

out.show()