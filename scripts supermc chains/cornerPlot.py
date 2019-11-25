import sys
import numpy as np
import corner

if (len(sys.argv)<4):
	print "Usage: python namechain.txt ndims aditional-burning"
	sys.exit(1)


chainfile, ndim, skip =sys.argv[1:4]
skip, ndim = int(skip), int(ndim)

npchain = np.loadtxt(chainfile)
samples = npchain[skip:,2:ndim+2]
weights = npchain[skip:,0]

rootname = chainfile.strip("_1.txt").strip(".txt")
print parsfile

parsfile =rootname+'.paramnames'
print(parsfile)

parnames = open(parsfile,'r')

latexnames=[]
for row in parnames:
	latexnames.append(row[0])


figure = corner.corner(samples, labels=latexnames[0:ndim],\
                       bins = 20,\
                       weights = weights,\
                       color='g',\
                       quantiles=[0.5],\
                       show_titles=True,\
                       title_fmt = '.4f',\
                       smooth=True,\
                       smooth1d=True,\
                       fill_contours=True,\
                       title_kwargs={"fontsize": 12})


figure.savefig(rootname+"_burn.png")
figure.savefig(rootname+"_burn.pdf")
	
from PIL import Image
img = Image.open(rootname+"burn.png")
img.show()