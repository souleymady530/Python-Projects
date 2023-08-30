# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:51:01 2023

@author: HP
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn
#Cette fonction va initialiser les parametres au depart,puisque l on donne des parametres aleatoires
#au depart puis en fonction de l erreur on corrige jusqu a trouver les meilleurs parametres permettant 
#de faire de bonne predictions
def initialisation(n0,n1,n2):
    w1=np.random.randn(n1,n0)
    b1=np.random.randn(n1,1)
    
    w2=np.random.randn(n2,n1)
    b2=np.random.randn(n2,1)
    
    parametres={
        'w1':w1,
        'b1':b1,
        'w2':w2,
        'b2':b2
        }
    return parametres

def forward_progagation(X,parametres):
    w1=parametres['w1'];
    b1=parametres['b1'];
    
    w2=parametres['w2'];
    b2=parametres['b2'];
    
    z1=w1.dot(X)+b1
    a1=1/(1+np.exp(-z1))
    
    z2=w2.dot(a1)+b2
    a2=1/(1+np.exp(-z2))
    
    activations={
        'a1':a1,
        'a2':a2,
        }
    return activations

def back_propagation(X,y,activations,parametres):
    a1=activations['a1']
    a2=activations['a2']
    w2=parametres['w2']
    
    m=y.shape[1]
    
    dz2=a2-y
    dw2=1/m * dz2.dot(a1.T)
    db2=1/m * np.sum(dz2,axis=1,keepdims=True)
    
    
     
    dz1=np.dot(w2.T,dz2)*a1*(1-a1)
    dw1=1/m * dz1.dot(X.T)
    db1=1/m * np.sum(dz1,axis=1,keepdims=True)
    
    gradients={
        'dw1':dw1,
        'db1':db1,
        'dw2':dw2,
        'db2':db2
        }
    
    return gradients;

def update(gradients,parametres,learning_rate):
    w1=parametres['w1']
    b1=parametres['b1']
    dw1=gradients['dw1']
    db1=gradients['db1']
    
    
    w2=parametres['w2']
    b2=parametres['b2']
    dw2=gradients['dw2']
    db2=gradients['db2']
    
    
    
    
    
    w1=w1-learning_rate*dw1
    b1=b1-learning_rate*db1
    w2=w2-learning_rate*dw2
    b2=b2-learning_rate*db2
    
    parametres={
        'w1':w1,
        'b1':b1,
        'w2':w2,
        'b2':b2
        }
def predict(X,parametres):
    activations=forward_progagation(X, parametres)
    A2=activations['a2']
    return A2>=0.5

def neural_network(X_train,Y_train,n1,learning_rate=0.1,n_iter=1000):
    #initialisation
    n0=X_train.shape[0]
    n2=Y_train.shape[0]
    parametres=initialisation(n0, n1, n2)
    
    train_loss=[]
    train_acc=[]
    
    for i in range(n_iter):
        activations=forward_progagation(X_train, parametres)
        gradients=back_propagation(X_train, Y_train, activations, parametres)
        parametres=update(gradients, parametres, learning_rate)
        
        if i%10==0 :
            train_loss.append(sklearn.log_loss(Y_train,activations['a2']))
            y_pred=predict(X_train, parametres)
            current_accuracy=sklearn.acaccuracy_score(Y_train.flattern(),y_pred.flattern())
            train_acc.append(current_accuracy)
            
    plt.figure(fig_size=(14,4))
    plt.subplot(1,2,1)
    plt.plot(train_loss,label="Train loss")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(train_acc,label="Train accuracy")
    plt.legend()
    plt.show()
    return parametres;
            