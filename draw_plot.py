# import matplotlib.pyplot as plt
from pylab import *
import numpy as np


f = open('log150M-150000000')

a = []
for i in f.readlines():
	if i.startswith('Mean'):
		a.append(float(i.split()[1]))
f. close()

b = []
f = open('log150M-150000000_poison')
for i in f.readlines():
	if i.startswith('Mean'):
		b.append(float(i.split()[1]))
f. close()

x = np.linspace(101600, 132211200, 132)

plot(x, a)
plot(x, b)

savefig("150M-150M.jpg")
show()
