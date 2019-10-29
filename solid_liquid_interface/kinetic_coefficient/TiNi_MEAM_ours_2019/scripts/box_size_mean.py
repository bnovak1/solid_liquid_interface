import numpy as np
import sys

infile = sys.argv[1]
outfile = sys.argv[2]
xcol = int(sys.argv[3])
ycol = int(sys.argv[4])
zcol = int(sys.argv[5])

data = np.loadtxt(infile)

xmean = np.mean(data[:, xcol])
ymean = np.mean(data[:, ycol])
zmean = np.mean(data[:, zcol])

outdata = np.array([xmean, ymean, zmean]).reshape(1, 3)
np.savetxt(outfile, outdata, header='Box sizes (x, y, z) in angstroms')
