import sys
import pandas as pd
import numpy as np
#from google.colab import drive
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=np.Inf)
np.set_printoptions(linewidth=np.Inf)

data = pd.read_csv('./train.csv', encoding = 'big5')

data = data.iloc[:,3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

month_data = {}
for month in range(12):
    sample = np.empty([18,480])
    for day in range(20):
        begin = day * 24
        end = begin + 24
        sample[:, begin : end] = raw_data[18 * (20 * month + day):18 * (20 * month + day + 1),
                                 0 : 24]
    month_data[month] =sample

# print(month_data)
x = np.empty([12*471,18*9], dtype=float)
y = np.empty([12*471,1],dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour >14:
                continue
            x[month*471+day*24+hour,:]=month_data[month][:,day*24+hour:day*24+hour+9].reshape(1,-1)
            y[month*471+day*24+hour,0]=month_data[month][9,day*24+hour+9]
mean_x = np.mean(x, axis=0)
std_x = np.std(x,axis=0)
for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j] != 0:
            x[i][j] = (x[i][j]-mean_x[j])/std_x[j]
dim = 18 * 9 + 1
w = np.zeros([dim,1])
x = np.concatenate((np.ones([12*471,1]),x),axis=1).astype(float)
learning_rate = 100
iter_time = 1000
adagrad = np.zeros([dim,1])
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x,w)-y,2))/471/12)
    if(t%100==0):
        print(str(t)+":"+str(loss))
    gradient = 2*np.dot(x.transpose(),np.dot(x,w)-y)
    adagrad += gradient ** 2
    w = w - learning_rate*gradient/np.sqrt(adagrad + eps)
np.save('weight.npy',w)

testdata = pd.read_csv('./test.csv',header = None,encoding='big5')
test_data = testdata.iloc[:,2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240,18*9], dtype=float)

# mean_x = np.mean(test_x, axis=0)
# std_x = np.std(test_x,axis=0)

for i in range(240):
    test_x[i, :]=test_data[18*i:18*(i+1),:].reshape(1,-1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j]-mean_x[j])/std_x[j]
test_x = np.concatenate((np.ones([240,1]),test_x),axis=1).astype(float)
w = np.load('weight.npy')
ans_y = np.dot(test_x,w)
print(ans_y)