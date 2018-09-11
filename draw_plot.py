# import matplotlib.pyplot as plt
from pylab import *
import numpy as np

f = open('log150M-200000')

# a = []
# for i in f.readlines():
# 	if i.startswith('Mean'):
# 		a.append(float(i.split()[1]))
# f. close()

# b = []
# f = open('log150M-150000000_before_pioson')
# for i in f.readlines():
# 	if i.startswith('Mean'):
# 		b.append(float(i.split()[1]))
# f. close()

a = []
for i in f.readlines():
	if i.startswith('total'):
		a.append(float(i.split()[7]))
f. close()

b = []
f = open('log150M-200000_poison')
for i in f.readlines():
	if i.startswith('total'):
		b.append(float(i.split()[7]))
f. close()

x = np.linspace(48000, 94400, 30)

plot(x, a)
plot(x, b)

savefig("150M-0.2M_distribution.jpg")
show()