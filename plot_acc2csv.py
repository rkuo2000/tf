# read imu.csv & convert to numpy array
import sys
import numpy as np

if len(sys.argv)>1:
    filename = sys.argv[1]
else:
    filename = '0_000.csv'
	
f = open(filename,'r')
line = f.readlines()
print(line)
acc = np.fromstring(line[0], dtype=float, sep=',')
print(len(acc))
acc = acc.reshape(int(len(acc)/3),3)
print(acc)
print(acc.shape)

print(len(acc[:,0]))
import matplotlib.pyplot as plt
x = np.linspace(0,len(acc[:,0]),len(acc[:,0]))
print(len(x))
plt.plot(x, acc[:,0])
plt.plot(x, acc[:,1])
plt.plot(x, acc[:,2])
plt.legend(['accX', 'accY', 'accZ'], loc='upper right')
plt.show()
