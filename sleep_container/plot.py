import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

# default = [10,50,100]
# y1 = [0.603,1.426,2.161]
# plt.plot(default, y1, label = "Default")

ema = [10,50,100,150,200,250]
y2 = [0.734,1.48,2.300,2.866,3.368,3.763]
data = np.array([[10,0.734], [50,1.48], [100,2.3], [150,2.866], [200,3.368], [250,3.763]])
f = interpolate.interp1d(ema, y2, fill_value = "extrapolate")

fit = np.polyfit(data[:,0], data[:,1] ,1)
line = np.poly1d(fit)
new_points = [250, 300, 500, 750, 1000, 1500, 2000]
extrapolation = line(new_points)
extrapolation2 = [float(f(250)),float(f(300)),float(f(500)),float(f(750)),float(f(1000)),float(f(1500)),float(f(2000))]
print(extrapolation)
print(extrapolation2)
plt.rcParams["figure.figsize"] = (15,5)
plt.plot(ema, y2, label = "EMA5-3",marker="v",markersize=13,color='steelblue',markerfacecolor='lightcoral',linewidth=3)
plt.plot(new_points, extrapolation2, label = "Extrapolation",marker="x", linestyle="--", markersize=13,color='steelblue',markerfacecolor='lightcoral',linewidth=3)
plt.xlabel('Number of concurrent VPAs',fontsize=25)
plt.ylabel('CPU Usage [%]',fontsize=25)
# plt.title('CPU usage of different numbers of concurrent VPAs',fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=19)
plt.show()
plt.savefig('./plots/results.png',bbox_inches='tight')