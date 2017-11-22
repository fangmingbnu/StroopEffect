import csv
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import t

data = pd.read_csv("stroopdata.csv")
print data.head()
print data.describe()
plt.figure(1)
data['Congruent'].hist(bins = 5, color='green', alpha = 0.5)
plt.xlabel('Time(s)',fontsize=14)
plt.ylabel('Count',fontsize=14)
plt.title('Time distribution',fontsize=14)

data['Incongruent'].hist(bins = 5, color='red', alpha = 0.5)
plt.xlabel('Time(s)',fontsize=14)
plt.ylabel('Count',fontsize=14)
plt.title('Time distribution',fontsize=14)
plt.legend(('Congruent','Incongruent'),fontsize=14)

plt.figure(2)
mu1, std1 = norm.fit(data['Congruent'])
x = np.linspace(0,50,100)
p1 = norm.pdf(x, mu1, std1)
plt.plot(x, p1, 'green', linewidth=2)

mu2, std2 = norm.fit(data['Incongruent'])
p2 = norm.pdf(x, mu2, std2)
plt.plot(x, p2, 'red', linewidth=2)
plt.title('Time distribution',fontsize=14)
plt.xlabel('Time(s)',fontsize=14)
plt.legend(('Congruent','Incongruent'),fontsize=14)


Congruent = np.array(data.values[:, 0])
Incongruent = np.array(data.values[:, 1])
diff = Congruent - Incongruent
print "Average = " , np.average(diff)
print "std deviation = " , np.std(diff)
print "t value = " , (np.average(diff) * np.sqrt(len(diff)-1)) / np.std(diff)
print "critical t ( CI = 95% double tailed) = " , t.ppf(1-0.025, 25)

plt.show()