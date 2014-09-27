import glob
import numpy as np
import matplotlib.pyplot as plt
import Imstats as img

psds = []

path = "../TestImgs/*.jpg"
for fname in glob.glob(path):
	psds.append(img.Imstat(fname).plot_power_spectrum())

print len(psds)

for i in range(0,len(psds)):
	#plt.plot(psds[i])
	plt.semilogy(psds[i])
plt.show()
    