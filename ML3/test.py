import matplotlib.pylab as pyl

import numpy as np

x=[]
y=[]
for i in range(1000000):
    x.append(i+1)
    y.append(i+1)

pyl.plot(x, y)

pyl.show()
