import psutil
import pickle
from matplotlib import pyplot as plt
filename = '300-ema53'
cpu = []
for i in range(60):
	cpu.append(psutil.cpu_percent(interval=1))
with open('./usage_data/{}'.format(filename),'wb') as f:
	pickle.dump(cpu, f)
average = sum(cpu)/len(cpu)
plt.plot(cpu)
plt.legend(['average:{}'.format(average)])
plt.savefig('./plots/{}.png'.format(filename))
