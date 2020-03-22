import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d

URL_='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
data = pd.read_csv(URL_,header = None)#reading the data from data base
'''storing all the 150 data in data,sdata,ver_data,gin_data'''
data = data[:149]
sdata = data[:149]
ver_data = data[:149]
gin_data = data[:149]
sdata[4] = np.where(data.iloc[:, -1]=='Iris-setosa', 0, 1)
#converting the data to float
sdata = np.asmatrix(sdata, dtype = 'float64')
ver_data[4] = np.where(data.iloc[:,-1]=='Iris-versicolor',0,1)
ver_data = np.asmatrix(ver_data,dtype='float64')
gin_data[4] = np.where(data.iloc[:,-1]=='Iris-verginica',0,1)
gin_data = np.asmatrix(gin_data,dtype='float64')
points=sdata[0:50,[0,1,2,3]]
hull=ConvexHull(points)
points1=ver_data[50:99,[0,1,2,3]]
hull1=ConvexHull(points1)
points2=gin_data[100:149,[0,1,2,3]]
hull2=ConvexHull(points2)

plt.scatter(np.array(sdata[:50,[
    0,1,2,3]]),np.array(sdata[:50,[0,1,2,3]]),marker='o',Label = 'setosa')
plt.scatter(np.array(ver_data[50:99,[0,1,2,3]]),np.array(ver_data[50:99,[0,1,2,3]]),marker='x',Label = 'versicolor')
plt.scatter(np.array(gin_data[99:149,[0,1,2,3]]),np.array(gin_data[99:149,[0,1,2,3]]),marker='+',Label = 'setosa')
for simplex in hull.simplices:
    plt.plot(points[simplex,0],points[simplex,1],'k-')
for simplex in hull1.simplices:
    plt.plot(points1[simplex,0],points1[simplex,1],'-k')
for simplex in hull2.simplices:
    plt.plot(points2[simplex,0],points2[simplex,1],'-k')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('convex hull')
plt.legend()

#intialization
features=sdata[:,[0,1,2,3]] #considering only two dimension features in this sepal length and petal length is considered
labels = sdata[:, -1]  # desired output for every feature set this is nothing but the column you created by assigning 0 and 1 for appropriate classes
w=2*np.random.random_sample((1,3)) # intial assignment of random weights to start with [bias (W0) W1 W2]
learning_rate=1.0  #learning rate is fixed at 1
misclassified_ = [] # list to store the misclassified input patterns
w_=[] #list to store the new weight upadation
num_iter = 1100  # how many times we want to give the complete data set to a perceptron so that learning happens with weights and all the input patterns are classified correctly 

#Error calculation for every input and weight updation based on perceptron learning
for epoch in range(num_iter):
    misclassified = 0
    for x, label in zip(features, labels):
        
        x = np.insert(x,0,1) # to add X0 input with value 1 which is used with bias as X0*W0 
        y = np.dot(w, x.transpose())# to implement the inner dot product (multiply and add )
                
        target = 1.0 if (y > 0) else 0.0 # implementation of threshold activation function
                    
        delta = (label.item(0,0) - target) # error = desired output - output of the TLN neuron
                    
        if(delta==1): # misclassified for input patterns that belong to X1 class and weight updation according to the algorithm
            misclassified += 1
            w =w+(learning_rate * x)
            w_.append(w)
                
        if(delta==-1): # misclassified for input patterns that belong to X0 class and weight updation according to the algorithm
            misclassified += 1
            w = w-(learning_rate * x)
            w_.append(w)
                
    misclassified_.append(misclassified)
    
 
# To plot the misclassified input patterns in each iteration
plt.figure()
epochs = np.arange(1, num_iter+1)
plt.plot(epochs, misclassified_)
plt.xlabel('iterations')
plt.ylabel('misclassified')
plt.title('iterations')
plt.legend()





# observe that after some iterations there are no misclassification and the perceptron has learnt to classify the new weights

plt.figure()
xmin, xmax = 4, 8 #based on your x-axis parameter range of you need to modify this
X = np.arange(xmin, xmax, 0.1) # generate some points on x axis between the above range at the interval of 0.1
m = -(w[0,1]/w[0,2]) # do u remember this how we used to calculate the m=-(w1/w2) slope of the straight line 
c = -(w[0,0]/w[0,2]) # do u remember this how we used to calculate the c=-(w1/w2) constant of the straight line 
plt.plot(X, m * X + c ) # y = mX+C
plt.scatter(np.array(sdata[:50,0]), np.array(sdata[:50,2]), marker='o') # The data is plotted back
plt.scatter(np.array(sdata[50:,0]), np.array(sdata[50:,2]), marker='x')#
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.title('separation through straight line')
plt.show()
