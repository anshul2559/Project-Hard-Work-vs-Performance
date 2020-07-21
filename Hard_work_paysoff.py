import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dfx=pd.read_csv(r"C:\Users\Admin\Desktop\coding blocks files\Linear_X_Train.csv")
dfy=pd.read_csv(r"C:\Users\Admin\Desktop\coding blocks files\Linear_Y_Train.csv")

X=dfx.values
Y=dfy.values

#Normalize the data set

u=X.mean()
std=X.std()
X=X-u/std

print(X.shape,Y.shape)

#Visualize the data set

plt.scatter(X,Y)
plt.xlabel("hardwork")
plt.ylabel("performance")
plt.show()

#Apply linear regression algorithm on the data set

def hypothesis(x,theta):
    y_=theta[0]+theta[1]*x
    return y_

def gradient(x,y,theta):
    grad=np.zeros((2,))
    m=x.shape[0]
    for i in range(m):
        y_=hypothesis(x[i],theta)
        grad[0]+=(y_-y[i])
        grad[1]+=(y_-y[i])*x[i]
    return grad/m

def error(x,y,theta):
    total_error=0.0
    m=x.shape[0]
    for i in range(m):
        y_ = hypothesis(x[i], theta)
        total_error+=(y_-y[i])**2
    return total_error/m

def gradient_descent(x,y,lr=0.1,max_steps=100):
    error_list=[]
    theta=np.zeros((2,))
    for i in range(max_steps):
        grad=gradient(x,y,theta)
        e=error(x,y,theta)
        error_list.append(e)
        theta[0] = theta[0] - lr * grad[0]
        theta[1] = theta[1] - lr * grad[1]
    return theta,error_list
theta,error_list=gradient_descent(X,Y)
print(theta)
print(error_list)

plt.style.use("seaborn")
plt.plot(error_list)
plt.show()

# Making predictions on TRAINING DATA

y_=hypothesis(X,theta)
print(y_)

#visualiztion

plt.scatter(X,Y,color="orange")
plt.plot(X,y_,label="prediction")
plt.legend()
plt.show()

# Making predictions on TEST DATA

X_test=pd.read_csv(r"C:\Users\Admin\Desktop\coding blocks files\Linear_X_Test.csv.txt").values
y_test=hypothesis(X_test,theta)
print(y_test)
print(X_test)

plt.plot(X_test,y_test)
plt.show()

# Now,store the result in the csv file

df=pd.DataFrame(data=y_test,columns=['y'])
df.to_csv('y_predictions.csv',index=False)


# r2 score

def r2_score(y,y_):
    num=np.sum((y-y_)**2)
    denom=np.sum((y-y.mean())**2)
    r2=1-(num/denom)
    return r2*100
score=r2_score(Y,y_)
print(score)









