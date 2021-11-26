import tensorflow as tf
import numpy as np
import copy
import pandas as pd
import random
import datetime
import pickle
import matplotlib.pyplot as plt
from sklearn import datasets
import random
import warnings
import xgboost
from sklearn.linear_model import LogisticRegressionCV,LarsCV,RidgeCV
import seaborn

class MNN(object):

    def __init__(
            self, max_iter=1000, batch_size=None, 
            units_each_group=None, number_of_groups=None,
            trace=True,learning_rate=0.1,dropout_ratio=0.05,
            remain_weights_ratio = 0.1,drop_ratio_per_iter = 0.05,
            l2_reg=1e-4,num_model=20):
        # if target not in ['classifier', 'regression']:
        #     raise ValueError('target could only be "classifier" or "regression"')
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.units_each_group = units_each_group
        self.number_of_groups = number_of_groups
        self.trace = trace
        self.learning_rate = learning_rate
        self.dropout_ratio = dropout_ratio
        self.remain_weights_ratio = remain_weights_ratio
        self.drop_ratio_per_iter = drop_ratio_per_iter
        self.l2_reg = l2_reg
        self.num_model = num_model
        self.x_mean_data = None
        self.x_std_data = None
        self.x_num = None
        self.weights = None
        self.biases = None
        self.Loss = None
        self.x_min = None
        self.x_max = None
        self.x_set_len = None
        self.importance = None
        self.w0 = None
        self.w_min = None
        self.b0 = None
        self.b_min = None
        
    def plotLoss(self,num=None):
        if self.Loss is None:
            raise ValueError('Error: plotLoss before fit')
        if num is None:
            num = len(self.Loss)
        fig,ax=plt.subplots(1,1)
        plt.plot(self.Loss[-num:])
        plt.ylabel("Loss")
        plt.rcParams['figure.figsize'] = (6, 3)
        ax.set_title('Losses Track',size=15)
        plt.show()


    def key_feature(self,dataX,num=None):
        if num is None:
            num = max(dataX.shape)
        gradient_list = []
        range_list = []
        low_bount_list = []
        high_bount_list = []
        for i in range(len(dataX)):
            x_plot,y_plot,y_range,grediant = self.interpret(dataX,i,Plot=None)
            gradient_list.append(grediant)
            range_list.append(np.abs(y_range[1]-y_range[0]))
            low_bount_list.append(y_range[0])
            high_bount_list.append(y_range[1])
        # return(range_list,low_bount_list,high_bount_list)
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        order_a = np.argsort(range_list)
        # return(order_a)
        order_d = [order_a[-_] for _ in range(1,len(order_a)+1)]
        gradient_order_d = [gradient_list[_] for _ in order_a]
        range_order_d = [range_list[_] for _ in order_a]
        low_bount_list_d = [low_bount_list[_] for _ in order_a]
        high_bount_list_d = [high_bount_list[_] for _ in order_a]

        ylocs = np.arange(len(order_a))[-num:]
        plt.sca(ax)
        gradient_order_d_positive = np.array(copy.deepcopy(gradient_order_d))
        gradient_order_d_negative = np.array(copy.deepcopy(gradient_order_d))
        gradient_order_d_positive[gradient_order_d_positive<0] = 0
        gradient_order_d_negative[gradient_order_d_negative>0] = 0
        plt.barh(ylocs, width=gradient_order_d_positive[-num:],color='firebrick', height=0.5)
        plt.barh(ylocs, width=gradient_order_d_negative[-num:],color='darkgreen', height=0.5)
        plt.vlines(0, ylocs[0]-0.5, ylocs[-1]+0.5, colors = "black")
        plt.xlabel('Gradients')
        plt.ylabel("Features")
        ax.set_yticks(ylocs)
        ax.set_yticklabels(['f' + str(i) for i in order_a[-num:]])
        ax.grid(True)
        plt.title('Gradients')
        plt.sca(ax2)
        plt.barh(ylocs, width=high_bount_list_d[-num:],color='firebrick', height=0.5)
        plt.barh(ylocs, width=low_bount_list_d[-num:],color='darkgreen', height=0.5)


        plt.vlines(0, ylocs[0]-0.5, ylocs[-1]+0.5, colors = "black")
        plt.xlabel("Feature Influence Ranges")
        ax2.set_yticks(ylocs)
        ax2.set_yticklabels(['f' + str(i) for i in order_a[-num:]])
        plt.xlabel("Target Moving Bound")
        # plt.ylabel("Features")
        # ax22 = ax2.twinx()
        # ax22.barh(ylocs, width=low_bount_list_d,color='steelblue', height=0.5)
        ax2.grid(True)
        plt.title('Target Moving Bound')


    def plot_importance(self,num=None):
        '''
        Bar plot of feature importance.
        '''
        if num is None:
            num = len(self.importance)
        importance = [np.mean(i) for i in self.importance]
        order_a = np.argsort(importance)
        order_d = [order_a[-_] for _ in range(1,len(order_a)+1)]
        importance_order_d = [importance[_] for _ in order_d]
        importance_order_a = [importance[_] for _ in order_a]
        importance_data = pd.DataFrame(columns = ['Influence Range','Feature'])
        for i in order_d[:num]:
            tmp_data = pd.DataFrame(np.ones((len(self.importance[i]),2)),columns = ['Influence Range','Features'])
            tmp_data['Features'] = 'f'+str(i)
            tmp_data['Influence Range'] = self.importance[i]
            importance_data = pd.concat((importance_data,tmp_data),sort=False)

        # importance_data = importance_data.iloc[-num:,:]
        importance_data = importance_data.loc[importance_data['Influence Range']>0,]
        _, ax = plt.subplots(1, 1)
        seaborn.boxplot(x='Influence Range',y='Features',data=importance_data,orient='h',ax=ax,color='white')#color = '#0d75f8')

        # ylocs = np.arange(len(importance_order_a))
        # ax.barh(ylocs, width=importance_order_a,color='steelblue', height=0.5)
        # ax.set_yticks(ylocs)
        # ax.set_yticklabels(['f' + str(i) for i in order_a])

        # plt.xlabel("R score")
        # plt.ylabel("Features")
        # ax.grid(True)
        plt.title('Feature Influence Range')
        return(importance)

    def save_parameter(self, path):
        with open(path, 'wb') as f:
            pickle.dump([
                self.max_iter,
                self.batch_size,
                self.units_each_group,
                self.number_of_groups,
                self.trace,
                self.learning_rate,
                self.dropout_ratio,
                self.x_mean_data,
                self.x_std_data,
                self.x_num,
                self.w0,
                self.w_min,
                self.b0,
                self.b_min,
                self.units_each_group,
                self.number_of_groups,
                self.drop_ratio_per_iter,
                self.Loss,
                self.x_min,
                self.x_max,
                self.x_set_len,
                self.importance,
                self.num_model
                ], f)

    def load_parameter(self, path):
        with open(path, 'rb') as f:
                self.max_iter,
                self.batch_size,
                self.units_each_group,
                self.number_of_groups,
                self.trace,
                self.learning_rate,
                self.dropout_ratio,
                self.x_mean_data,
                self.x_std_data,
                self.x_num,
                self.w0,
                self.w_min,
                self.b0,
                self.b_min,
                self.units_each_group,
                self.number_of_groups,
                self.drop_ratio_per_iter,
                self.Loss,
                self.x_min,
                self.x_max,
                self.x_set_len,
                self.importance,
                self.num_model = pickle.load(f, encoding='latinl')

    def __get_stand_x(self,dataX):
        x_stand = (dataX-self.x_mean_data)/self.x_std_data
        return(x_stand)

    def active_feature(self,dataX,quantile=0.9):
        dataX = np.reshape(dataX,(1,max(dataX.shape)))
        dataX_stand = self.__get_stand_x(dataX)
        h0 = np.multiply(dataX_stand,self.w0) + self.b0
        h_min = np.matmul(h0,self.w_min)+self.b_min

        group_mins_array = -np.Inf*np.ones([len(dataX),1])
        for group_num in range(self.number_of_groups):
            group_array = h_min[:,group_num*self.units_each_group : (group_num+1)*self.units_each_group]
            group_min = np.reshape(np.min(group_array,axis=1),(len(group_array),1))
            group_mins_array = np.concatenate((group_mins_array,group_min),axis=1)
        group_mins_array = group_mins_array[:,1:]
        active_group = np.argmax(group_mins_array)
        active_group_values = group_array = h_min[:,active_group*self.units_each_group : (active_group+1)*self.units_each_group]
        active_min = np.argmin(active_group_values)
        active_index = active_group*self.units_each_group+active_min
        active_weights = self.w_min[:,active_index]
        # active_weights = np.multiply(self.w0,self.w_min[:,active_index])
        thresh = np.quantile(active_weights,quantile)
        active_feature_list = [i for i in range(len(active_weights)) if active_weights[i]>thresh]
        return(active_feature_list,active_index)



class MNNClassifier(MNN):
    '''
Parameters
----------
max_iter : int, default: 100
    Maximum number of iterations of the optimization algorithm.
batch_size : int, default: 1024
    Number of rows of each batch of the batch training.
    If equal to number of rows of training data, then the algorithm
    would be equivalent to full scale training.
units_each_group : int, default: None
    Number of nodes in the first hidden layers after sign layer.
    if set to be None, it will take value of
    int(np.log(#feature)*4.5)
number_of_groups : int, default: None
    Number of nodes in the hidden layers between the second hidden
    layer and output layer. If set to be None, it will take value of
    int(np.log10(#training set)**1.6)
trace : boolean, default: True
    If set to be True, loss will be printed during training
learning_rate : float, default: 0.01
dropout_ratio : float, default: 0.1
Q0 : float, default: 0.98
    When robust is 'quantile':
    if set to be a non-negative value less than 1, the loss function for
    the smaples of lable equals to 0 would be modified to exclude the
    elements greater than quantile Q0 before calculate the mean.
Q1 : float, default: 1.0
    When robust is 'quantile':
    if set to be a non-negative value less than 1, the loss function for
    the smaples of lable equals to 1 would be modified to exclude the
    elements greater than quantile Q1 before calculate the mean.
Attributes
----------
weights: list of length 5
    weights in the encoding network
biases: list of length 5
    biases in the encoding network
Methods
-------
fit: optimize the encoding network
predict: predict the labels
predict_proba: predict the probabilities
plotLoss : trace plot of loss
save_parameter: save the parameters into a pickle file
load_parameter: load the parameters from a pickle file
    '''
    def __init__(
            self,max_iter=1500, batch_size=None,
            units_each_group=None, number_of_groups=None,
            trace=True,learning_rate=0.01,dropout_ratio=0.15,
            Q1=1.0,Q0=1.0,remain_weights_ratio = 0.1,
            drop_ratio_per_iter = 0.05,l2_reg=1e-4,num_model=20):

        MNN.__init__(self,max_iter,batch_size,units_each_group,number_of_groups,trace,
            learning_rate,dropout_ratio,remain_weights_ratio,drop_ratio_per_iter,l2_reg,num_model)
        self.Q1 = Q1
        self.Q0 = Q0
        self.x_cov = None

    def __get_x_info(self,dataX):
        '''
        record number of feature in dataX, mean and std of
        each feature.
        '''
        self.x_num = dataX.shape[1]
        self.x_mean_data = dataX.mean(0)
        self.x_std_data = dataX.std(0)
        self.x_min = np.min(dataX,axis=0)
        self.x_max = np.max(dataX,axis=0)
        self.x_set_len = []
        for index_ in range(dataX.shape[1]):
            self.x_set_len.append(len(set(dataX[:,index_])))
        self.x_set_len = np.array(self.x_set_len)

    def __get_stand_x(self,dataX):
        x_stand = (dataX-self.x_mean_data)/self.x_std_data
        return(x_stand)


    def __get_importance(self,dataX):
        importance = []
        for index_x in range(dataX.shape[1]):
            x_min_array = copy.deepcopy(dataX)
            x_min_array[:,index_x] = self.x_min[index_x]
            y_min = self.predict_proba(x_min_array)[:,1]
            del x_min_array
            x_max_array = copy.deepcopy(dataX)
            x_max_array[:,index_x] = self.x_max[index_x]
            y_max = self.predict_proba(x_max_array)[:,1]
            del x_max_array
            y_range = np.abs(y_max - y_min)
            importance.append(y_range)
        self.importance = importance

    def fit(self,dataX,dataY):
        '''
        Train the parameters of monotonic neural network.
        dataX : float type ndarray, shape: [n,#feature]
        dataY : int type ndarray, shape: [n,#class]
                should be one-hot encoded
        '''
        self.__get_x_info(dataX)
        dataX_stand = self.__get_stand_x(dataX)
        self.x_cov = np.cov(dataX.T)

        x,y,y_linear,y_proba,rate,sess,Loss,w0_,w_min_,b0_,b_min_,units_each_group,number_of_groups = fit_network_strict_mono(
            dataX_stand,
            dataY,
            target = 'classifier',
            units_each_group=self.units_each_group,
            number_of_groups=self.number_of_groups,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            trace = self.trace,
            learning_rate = self.learning_rate,
            dropout_ratio = self.dropout_ratio,
            Q1 = self.Q1,
            Q0 = self.Q0,
            remain_weights_ratio = self.remain_weights_ratio,
            drop_ratio_per_iter = self.drop_ratio_per_iter,
            l2_reg= self.l2_reg,
            num_model = self.num_model
        )
        self.__x = x
        self.__y = y
        self.__y_linear = y_linear
        self.__y_proba = y_proba
        self.__sess = sess
        self.__rate = rate
        self.Loss = Loss
        self.w0 = w0_
        self.w_min = w_min_
        self.b0 = b0_
        self.b_min = b_min_
        self.units_each_group = units_each_group
        self.number_of_groups = number_of_groups
        self.__get_importance(dataX)


    def interpret(self,dataX,x_index,Plot=True,dataY=None,train_X=None,train_Y=None):
        '''
        Return function between feature of given x_index and target.
        input:
            dataX : ndarray, shape: [#feature,]
            x_index : int, index of the feature
        output:
            x_plot : ndarray, shape: [1000,]. x axis values of the points
                in the plot
            y_plot : ndarray, shape: [1000,]. y axis values of the points
                in the plot
            y_range : list, shape: [2,]. Maximum range of target as the given
                feature moves.
        '''
        dataX = np.reshape(dataX,(1,len(dataX)))
        x_index_max = max(self.x_max[x_index],dataX[0,x_index])
        x_index_min = min(self.x_min[x_index],dataX[0,x_index])
        x_0 = dataX[0,x_index]
        if self.x_set_len[x_index]<=2:
            x_delta = x_index_min if x_0==x_index_max else x_index_max
            y_0 = self.predict_proba(dataX)[0,1]
            x_array_delta = copy.deepcopy(dataX)
            x_array_delta[0,x_index] = x_delta
            y_delta = self.predict_proba(x_array_delta)[0,1]
            grediant = (y_delta-y_0)*(x_delta-x_0)/10
            y_d_higher = y_delta-y_0 if y_delta>y_0 else 0
            y_d_lower = y_delta-y_0 if y_delta<y_0 else 0
            return(0,0,[y_d_lower,y_d_higher],grediant)
        x_plus_delta = x_0 + (x_index_max-x_index_min)/10
        x_plus_delta_array = copy.deepcopy(dataX)
        x_plus_delta_array[0,x_index] = x_plus_delta
        y_0 = self.predict_proba(dataX)[0,1]
        y_plus_delta = self.predict_proba(x_plus_delta_array)[0,1]
        grediant = (y_plus_delta-y_0)

        x_array_max = copy.deepcopy(dataX)
        x_array_max[0,x_index] = x_index_max
        x_array_min = copy.deepcopy(dataX)
        x_array_min[0,x_index] = x_index_min
        y_hat_max = self.predict_proba(x_array_max)[0,1]
        y_hat_min = self.predict_proba(x_array_min)[0,1]
        y_d_higher = max(y_hat_max-y_0,y_hat_min-y_0)
        y_d_lower = min(y_hat_max-y_0,y_hat_min-y_0)

        mono_direction = 0
        if y_hat_max > y_hat_min:
            mono_direction = 1
        elif y_hat_max < y_hat_min:
            mono_direction = -1
        x_plot = x_index_min + np.array(range(1000))/1000*(x_index_max-x_index_min)
        y_plot = []
        for index_plot in range(len(x_plot)):
            x_plot_data = copy.deepcopy(dataX)
            x_plot_data[0,x_index] = x_plot[index_plot]
            y_plot.append(self.predict_proba(x_plot_data)[0,1])
        if Plot:
            _, ax = plt.subplots(1, 1)
            ax.plot(x_plot,y_plot,'-',color='black',label='Instance Reponse Curve')
            ax.plot(x_0,y_0,'o',color='black',label='Predicted Probability')
            if dataY is  not None:
                ax.plot(x_0,dataY,'o',color='red',label='True Value')
            ax.grid(True)
            ax.legend(loc=0)
            # ax.plot(self.white_sample[:,x_index],[0 for i in range(1000)],'o',\
            #     alpha = 0.1,color='darkgreen',label = '')#firebrick
            # ax.grid(True)#
            plt.ylabel('Predicted Probability')
            plt.xlabel("Feature Value")
            plt.title(r"Single Variable Influence Curve of $f_{"+str(x_index)+"}$")

            if train_X is not None and train_Y is not None:
                # b = np.ones((len(train_Y),2))
                # b[:,0] = train_Y<=max(y_hat_max,y_hat_min)
                # b[:,0] = train_Y>=min(y_hat_max,y_hat_min)
                # slice_bool = np.sum(b,axis=1)==2
                # train_Y_draw = copy.deepcopy(train_Y[slice_bool])
                # train_X_draw = copy.deepcopy(train_X[slice_bool,x_index])
                # ax.plot(train_X_draw,train_Y_draw,'o',color='grey',label='Traning Sample',alpha=0.3)
                makov_distance = np.matmul(np.matmul(dataX,np.linalg.pinv(self.x_cov)),train_X.T).T[:,0]
                rank_makov = np.argsort(makov_distance)
                slice_index = rank_makov[-int(len(makov_distance)*0.3):]
                train_Y_draw = copy.deepcopy(train_Y[slice_index])
                train_X_draw = copy.deepcopy(train_X[slice_index,x_index])
                y_mean = []
                es_width = (x_plot[-1]-x_plot[0])/10
                x_plot_copy = list(copy.deepcopy(x_plot))
                for i in x_plot:
                    b = np.ones((len(train_X_draw),2))
                    b[:,0] = train_X_draw<=i+es_width
                    b[:,1] = train_X_draw>=i-es_width
                    slice_bool = np.sum(b,axis=1)==2
                    if sum(slice_bool)==0:
                        x_plot_copy.remove(i)
                    else:
                        y_mean.append(np.mean(train_Y_draw[slice_bool]))
                ax.plot(x_plot_copy,y_mean,'--',color='grey',label='Kernel Regression')

                # ax2 = ax.twinx()
                # ax2.plot(train_X_draw,train_Y_draw,'o',color='grey',label='Traning Sample',alpha=0.3)
                # ax2.legend(loc=0)
                ax.plot(train_X_draw,train_Y_draw,'o',color='grey',label='Traning Sample',alpha=0.3)
                ax.legend(loc=0)
                return(x_plot_copy,y_mean,[train_X_draw,train_Y_draw],grediant)

            # ax2 = ax.twinx()
            # ax2.plot(self.__x_plot_seqs[x_index],self.__higher_bound[x_index],'--',color='grey',label=r'1.5$\sigma$ bound')
            # ax2.plot(self.__x_plot_seqs[x_index],self.__lower_bound[x_index],'--',color='grey')
            # ax2.plot(self.__x_plot_seqs[x_index],self.__centre_points[x_index],'-',color='grey',label='non-parameter estemation')
            # ax2.legend(loc=0)


        return(x_plot,y_plot,[y_d_lower,y_d_higher],grediant)



    def predict_proba(self, dataX):
        '''
        predict the probability of each label for each instancek
        input:
            dataX : ndarray, shape: [n,#feature]
        output:
            y_label : ndarray, shape: [n,#class]
        '''
        dataX_stand = self.__get_stand_x(dataX)
        h0 = np.multiply(dataX_stand,self.w0) + self.b0
        h_min = np.matmul(h0,self.w_min)+self.b_min

        group_mins_array = -np.Inf*np.ones([len(dataX),1])
        for group_num in range(self.number_of_groups):
            group_array = h_min[:,group_num*self.units_each_group : (group_num+1)*self.units_each_group]
            group_min = np.reshape(np.min(group_array,axis=1),(len(group_array),1))
            group_mins_array = np.concatenate((group_mins_array,group_min),axis=1)
        group_max_array = np.reshape(np.max(group_mins_array,axis=1),(len(group_mins_array),1))

        y_linear = group_max_array
        y_proba = np.exp(y_linear)/(1+np.exp(y_linear))

        y_proba2 = np.zeros((len(y_proba),2))
        y_proba2[:,0] = 1-y_proba[:,0]
        y_proba2[:,1] = y_proba[:,0]
        return(y_proba2)

    def predict(self, dataX):
        '''
        predict the label of each instance
        input:
            dataX : ndarray, shape: [n,#feature]
        output:
            y_label : ndarray, shape: [n,]
        '''
        y_proba = self.predict_proba(dataX)
        cols = [_ for _ in range(y_proba.shape[1])]
        colNameMatrix = np.array([cols for i in range(len(y_proba))])
        # If there exists more than one maximum values, only one of the index
        # will return randomly
        valueNoise = y_proba.values + np.random.normal(scale=1e-9,size=y_proba.shape)
        bool_slicer_max = np.transpose(np.transpose(valueNoise)==np.max(valueNoise,axis=1))
        y_label = colNameMatrix[bool_slicer_max]
        return(y_label)


class MNNRegressor(MNN):
    '''
Parameters
----------
max_iter : int, default: 100
    Maximum number of iterations of the optimization algorithm.
batch_size : int, default: None
    Number of rows of each batch of the batch training.
    If set to be None, batch_size would be 1/3 of number of
    instences contained in training data.
units_each_group : int, default: None
    Number of nodes in the first hidden layers after sign layer.
    if set to be None, it will take value of
    int(np.log(#feature)*4.5)
number_of_groups : int, default: None
    Number of nodes in the hidden layers between the second hidden
    layer and output layer. If set to be None, it will take value of
    int(np.log10(#training set)**1.6)
trace : boolean, default: True
    If set to be True, loss will be printed during training
learning_rate : float, default: 0.01
dropout_ratio : float, default: 0.02
Attributes
----------
weights: list of length 5
    weights in the encoding network
biases: list of length 5
    biases in the encoding network
Methods
-------
fit: optimize the encoding network
predict: predict the values
plotLoss : trace plot of loss
save_parameter: save the parameters into a pickle file
load_parameter: load the parameters from a pickle file
    '''
    def __init__(
            self,max_iter=1500, batch_size=None,
            units_each_group=None, number_of_groups=None,
            trace=True, learning_rate=0.01,
            dropout_ratio=0.15,Q=0.95,remain_weights_ratio = 0.1,
            drop_ratio_per_iter = 0.05, l2_reg = 1e-4,num_model = 20):
        MNN.__init__(self,max_iter,batch_size,units_each_group,number_of_groups,
            trace,learning_rate,dropout_ratio,remain_weights_ratio,drop_ratio_per_iter,l2_reg,num_model)
        self.x_cov = None
        self.Q = Q

    def __get_x_info(self,dataX):
        '''
        record number of feature in dataX, mean and std of
        each feature.
        '''
        self.x_num = dataX.shape[1]
        self.x_mean_data = dataX.mean(0)
        self.x_std_data = dataX.std(0)
        self.x_min = np.min(dataX,axis=0)
        self.x_max = np.max(dataX,axis=0)
        self.x_set_len = []
        for index_ in range(dataX.shape[1]):
            self.x_set_len.append(len(set(dataX[:,index_])))
        self.x_set_len = np.array(self.x_set_len)

    def __get_y_info(self,dataY):
        '''
        record number of feature in dataX, mean and std of
        each feature.
        '''
        if len(dataY.shape)==1:
            self.y_mean_data = dataY.mean()
            self.y_std_data = dataY.std()
        else:
            self.y_mean_data = dataY.mean(0)
            self.y_std_data = dataY.std(0)

    def __get_stand_x(self,dataX):
        x_stand = (dataX-self.x_mean_data)/self.x_std_data
        return(x_stand)

    def __get_stand_y(self,dataY):
        y_stand = (dataY-self.y_mean_data)/self.y_std_data
        return(y_stand)

    def __get_transed_y(self,dataY):
        y_transed = dataY*self.y_std_data+self.y_mean_data
        return(y_transed)

    def __get_importance(self,dataX):
        importance = []
        for index_x in range(dataX.shape[1]):
            x_min_array = copy.deepcopy(dataX)
            x_min_array[:,index_x] = self.x_min[index_x]
            y_min = self.predict(x_min_array)
            del x_min_array
            x_max_array = copy.deepcopy(dataX)
            x_max_array[:,index_x] = self.x_max[index_x]
            y_max = self.predict(x_max_array)
            del x_max_array
            y_range = np.abs(y_max - y_min)
            importance.append(y_range)
        self.importance = importance


    def fit(self,dataX,dataY):
        '''
        Train the parameters of monotonic neural network.
        dataX : float type ndarray, shape: [n,#feature]
        dataY : int type ndarray, shape: [n,#class]
                should be one-hot encoded
        '''
        self.__get_x_info(dataX)
        self.__get_y_info(dataY)

        dataX_stand = self.__get_stand_x(dataX)
        dataY_stand = self.__get_stand_y(dataY)
        self.x_cov = np.cov(dataX.T)
        x,y,y_linear,y_proba,rate,sess,Loss,w0_,w_min_,b0_,b_min_,units_each_group,number_of_groups = fit_network_strict_mono(
            dataX_stand,
            dataY_stand,
            target = 'regression',
            units_each_group=self.units_each_group,
            number_of_groups=self.number_of_groups,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
            trace = self.trace,
            learning_rate = self.learning_rate,
            dropout_ratio = self.dropout_ratio,
            Q = self.Q,
            remain_weights_ratio = self.remain_weights_ratio,
            drop_ratio_per_iter = self.drop_ratio_per_iter,
            l2_reg = self.l2_reg,
            num_model = self.num_model
        )
        self.__y = y
        self.__x = x
        self.__y_linear = y_linear
        self.__y_proba = y_proba
        self.__sess = sess
        self.__rate = rate
        self.Loss = Loss
        self.w0 = w0_
        self.w_min = w_min_
        self.b0 = b0_
        self.b_min = b_min_
        self.units_each_group = units_each_group
        self.number_of_groups = number_of_groups
        self.__get_importance(dataX)



    def interpret(self,dataX,x_index,dataY=None,Plot=True,train_X=None,train_Y=None):
        '''
        Return function between feature of given x_index and target.
        input:
            dataX : ndarray, shape: [#feature,]
            x_index : int, index of the feature
        output:
            x_plot : ndarray, shape: [1000,]. x axis values of the points
                in the plot
            y_plot : ndarray, shape: [1000,]. y axis values of the points
                in the plot
            y_range : list, shape: [2,]. Maximum range of target as the given
                feature moves.
        '''
        dataX = np.reshape(dataX,(1,len(dataX)))
        x_index_max = max(self.x_max[x_index],dataX[0,x_index])
        x_index_min = min(self.x_min[x_index],dataX[0,x_index])
        x_0 = dataX[0,x_index]
        if self.x_set_len[x_index]<=2:
            x_delta = x_index_min if x_0==x_index_max else x_index_max
            y_0 = self.predict(dataX)
            x_array_delta = copy.deepcopy(dataX)
            x_array_delta[0,x_index] = x_delta
            y_delta = self.predict(x_array_delta)
            grediant = (y_delta-y_0)*(x_delta-x_0)/10
            y_d_higher = y_delta-y_0 if y_delta>y_0 else 0
            y_d_lower = y_delta-y_0 if y_delta<y_0 else 0
            return(0,0,[y_d_lower,y_d_higher],grediant)
        x_plus_delta = x_0 + (x_index_max-x_index_min)/10
        x_plus_delta_array = copy.deepcopy(dataX)
        x_plus_delta_array[0,x_index] = x_plus_delta
        y_0 = self.predict(dataX)[0]
        y_plus_delta = self.predict(x_plus_delta_array)[0]
        grediant = (y_plus_delta-y_0)

        x_array_max = copy.deepcopy(dataX)
        x_array_max[0,x_index] = x_index_max
        x_array_min = copy.deepcopy(dataX)
        x_array_min[0,x_index] = x_index_min
        y_hat_max = self.predict(x_array_max)[0]
        y_hat_min = self.predict(x_array_min)[0]
        y_d_higher = max(y_hat_max-y_0,y_hat_min-y_0)
        y_d_lower = min(y_hat_max-y_0,y_hat_min-y_0)

        mono_direction = 0
        if y_hat_max > y_hat_min:
            mono_direction = 1
        elif y_hat_max < y_hat_min:
            mono_direction = -1
        if self.x_set_len[x_index]<=2:
            return(0)
        x_plot = x_index_min + np.array(range(1000))/1000*(x_index_max-x_index_min)
        y_plot = []
        for index_plot in range(len(x_plot)):
            x_plot_data = copy.deepcopy(dataX)
            x_plot_data[0,x_index] = x_plot[index_plot]
            y_plot.append(self.predict(x_plot_data)[0])
        if Plot:
            _, ax = plt.subplots(1, 1)
            ax.plot(x_plot,y_plot,'-',color='black',label='Instance Reponse Curve')
            ax.plot(x_0,y_0,'o',color='black',label='Predicted Probability')
            if dataY is  not None:
                ax.plot(x_0,dataY,'o',color='red',label='True Value')
            ax.grid(True)
            ax.legend(loc=0)
            plt.ylabel('Predicted Probability')
            plt.xlabel("Feature Value")
            plt.title(r"Single Variable Influence Curve of $f_{"+str(x_index)+"}$")

            if train_X is not None and train_Y is not None:
                # b = np.ones((len(train_Y),2))
                # b[:,0] = train_Y<=max(y_hat_max,y_hat_min)
                # b[:,0] = train_Y>=min(y_hat_max,y_hat_min)
                # slice_bool = np.sum(b,axis=1)==2
                # train_Y_draw = copy.deepcopy(train_Y[slice_bool])
                # train_X_draw = copy.deepcopy(train_X[slice_bool,x_index])
                # ax.plot(train_X_draw,train_Y_draw,'o',color='grey',label='Traning Sample',alpha=0.3)
                makov_distance = np.matmul(np.matmul(dataX,np.linalg.pinv(self.x_cov)),train_X.T).T[:,0]
                rank_makov = np.argsort(makov_distance)
                slice_index = rank_makov[-int(len(makov_distance)*0.3):]
                train_Y_draw = copy.deepcopy(train_Y[slice_index])
                train_X_draw = copy.deepcopy(train_X[slice_index,x_index])
                y_mean = []
                es_width = (x_plot[-1]-x_plot[0])/10
                x_plot_copy = list(copy.deepcopy(x_plot))
                for i in x_plot:
                    b = np.ones((len(train_X_draw),2))
                    b[:,0] = train_X_draw<=i+es_width
                    b[:,1] = train_X_draw>=i-es_width
                    slice_bool = np.sum(b,axis=1)==2
                    if sum(slice_bool)==0:
                        x_plot_copy.remove(i)
                    else:
                        y_mean.append(np.mean(train_Y_draw[slice_bool]))
                ax.plot(x_plot_copy,y_mean,'--',color='grey',label='Kernel Regression')

                # ax2 = ax.twinx()
                # ax2.plot(train_X_draw,train_Y_draw,'o',color='grey',label='Traning Sample',alpha=0.3)
                # ax2.legend(loc=0)
                ax.plot(train_X_draw,train_Y_draw,'o',color='grey',label='Traning Sample',alpha=0.3)
                ax.legend(loc=0)

                return(x_plot_copy,y_mean,[train_X_draw,train_Y_draw],grediant)

        return(x_plot,y_plot,[y_d_lower,y_d_higher],grediant)


    def predict(self, dataX):
        '''
        predict the probability of each label for each instancek
        input:
            dataX : ndarray, shape: [n,#feature]
        output:
            y_label : ndarray, shape: [n,#class]
        '''
        dataX_stand = self.__get_stand_x(dataX)
        h0 = np.multiply(dataX_stand,self.w0) + self.b0
        h_min = np.matmul(h0,self.w_min)+self.b_min

        group_mins_array = -np.Inf*np.ones([len(dataX),1])
        for group_num in range(self.number_of_groups):
            group_array = h_min[:,group_num*self.units_each_group : (group_num+1)*self.units_each_group]
            group_min = np.reshape(np.min(group_array,axis=1),(len(group_array),1))
            group_mins_array = np.concatenate((group_mins_array,group_min),axis=1)
        group_max_array = np.reshape(np.max(group_mins_array,axis=1),(len(group_mins_array),1))

        y_linear = group_max_array
        y_linear = np.reshape(y_linear,(y_linear.shape[0],))
        dataY_stand = self.__get_transed_y(y_linear)
        dataY_stand = np.reshape(dataY_stand,(dataY_stand.shape[0],))
        return(dataY_stand)




def make_positive(input_tensor):
    # out_ = input_tensor*tf.cast(input_tensor>0, tf.float32)
    out_ = tf.clip_by_value(input_tensor, 0.0, 9999)
    return out_

def make_greater_than_1(input_tensor):
    # out_ = input_tensor*tf.cast(input_tensor>0, tf.float32)
    out_ = tf.clip_by_value(input_tensor, 1e-12, 9999)
    return out_

def become_sign(input_tensor):
    # out_ = input_tensor*tf.cast(input_tensor>0, tf.float32)
    out_ = tf.sign(input_tensor)
    return out_


def fit_network_strict_mono(
        dataX_, dataY_, target,
        units_each_group=None, number_of_groups=None ,
        max_iter=150, batch_size=None,
        trace=True,learning_rate=0.01,
        dropout_ratio = 0.1,Q1=1,Q0=0.98,Q=0.95,
        remain_weights_ratio = 0.1, drop_ratio_per_iter = 0.05,l2_reg=1e-4,num_model=20):

    dataX = dataX_
    dataY = dataY_

    if batch_size is None:
        batch_size = int(len(dataY_)/3)
    if units_each_group is None:
        units_each_group = int(np.log(dataX_.shape[1])**2)
    if number_of_groups is None:
        number_of_groups = int(np.log10(dataX_.shape[0])*5)


    imbalance = False

    if target=='classifier':

        minimal_data_X = []
        minimal_data_Y = []

        sample_size_0 = sum(dataY==0)
        sample_size_1 = sum(dataY==1)
        white_bool = dataY==0
        index_0  = [_ for _ in range(len(dataY)) if white_bool[_]]
        index_1  = [_ for _ in range(len(dataY)) if not white_bool[_]]

        if sample_size_0 >= sample_size_1*1.5 or sample_size_1 >= sample_size_0*1.5:

            if sample_size_0 >= sample_size_1:

                index_1 = index_1*int(sample_size_0/sample_size_1)
                imbalance = True
                print('黑样本复制')
            elif sample_size_1 >= sample_size_0:

                index_0 = index_0*int(sample_size_1/sample_size_0)
                imbalance = True
                print('白样本复制')

            batch_size = int(len(index_1)/3+len(index_0)/3)

            dataX = dataX[index_0+index_1]
            dataY = dataY[index_0+index_1]

    if target=='classifier':
        logistic_model = LogisticRegressionCV(solver='lbfgs')
        logistic_model.fit(dataX,dataY)
        w_logistic = logistic_model.coef_[0]
        b_logistic = logistic_model.intercept_[0]

    if target=='regression':
        lsr = RidgeCV()
        lsr.fit(dataX,dataY)
        w_logistic = lsr.coef_
        b_logistic = lsr.intercept_

    w_min_init = np.multiply(np.zeros([dataX.shape[1], units_each_group*number_of_groups]) + \
        1/(units_each_group*number_of_groups),np.random.uniform(0.9,1.1,(units_each_group*number_of_groups,)))

    w_min_mask = np.ones([dataX.shape[1], units_each_group*number_of_groups])

    ####第一次拟合全网络
    if remain_weights_ratio==1:
        x,y,y_linear,y_proba,rate,sess,Loss,w0_,w_min_,b0_,b_min_ = single_iter_LT(
            dataX, dataY, target, w_min_mask, w_logistic, b_logistic, units_each_group, number_of_groups , max_iter,
            batch_size, trace,learning_rate,dropout_ratio,Q1,Q0,Q,l2_reg,num_model)
    else:
        x,y,y_linear,y_proba,rate,sess,Loss,w0_,w_min_,b0_,b_min_ = single_iter_LT(
            dataX, dataY, target, w_min_mask, w_logistic, b_logistic, units_each_group, number_of_groups , max_iter,
            batch_size, trace,learning_rate,dropout_ratio,Q1,Q0,Q,l2_reg,1)

    ####迭代删除weights
    total_num_weights = dataX.shape[1]*units_each_group*number_of_groups
    final_num_weights = int(total_num_weights*remain_weights_ratio)
    num_weights_to_drop_per_iter = int(total_num_weights*drop_ratio_per_iter)
    num_weights_droped = 0

    while  np.sum(w_min_mask) > final_num_weights:
        num_weights_to_drop = int(min(num_weights_to_drop_per_iter,np.sum(w_min_mask)-final_num_weights))
        order_drop = np.argsort(np.reshape(w_min_,(w_min_.shape[0]*w_min_.shape[1],)))
        for position in order_drop[num_weights_droped:(num_weights_droped+num_weights_to_drop)]:
            num_row = position // w_min_.shape[1]
            num_col = position % w_min_.shape[1]
            w_min_mask[num_row,num_col] = 0
        if np.sum(w_min_mask) - final_num_weights > 0:
            x,y,y_linear,y_proba,rate,sess,Loss,w0_,w_min_,b0_,b_min_ = single_iter_LT(
                dataX, dataY, target, w_min_mask, w_logistic, b_logistic, units_each_group, number_of_groups , max_iter,
                batch_size, trace,learning_rate,dropout_ratio,Q1,Q0,Q,l2_reg,1)
        else:
            x,y,y_linear,y_proba,rate,sess,Loss,w0_,w_min_,b0_,b_min_ = single_iter_LT(
                dataX, dataY, target, w_min_mask, w_logistic, b_logistic, units_each_group, number_of_groups , max_iter,
                batch_size, trace,learning_rate,dropout_ratio,Q1,Q0,Q,l2_reg,num_model)
        num_weights_droped += num_weights_to_drop

    return(x,y,y_linear,y_proba,rate,sess,Loss,w0_,w_min_,b0_,b_min_,units_each_group,number_of_groups)



def single_iter_LT(
        dataX_, dataY_, target, w_min_mask,w_logistic,b_logistic,
        units_each_group=None, number_of_groups=None ,
        max_iter=150, batch_size=None,
        trace=True,learning_rate=0.01,
        dropout_ratio = 0.1,Q1=1,Q0=0.98,Q=0.95,l2_reg = 1e-4,num_model=20):
    '''
    The optimazation would stop if loss function reduces less than 0.01%, or number of iterations reaches
    max_iter.
    '''
    # w_logistic = np.random.normal(size=[dataX_.shape[1]])
    # b_logistic = 0

    dataX = dataX_
    dataY = dataY_

    tf.compat.v1.reset_default_graph()
    if len(dataY.shape)==1:
        dataY = np.reshape(dataY,[len(dataY),1])

    x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, dataX.shape[1]], name='x')
    y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, dataY.shape[1]], name='y')
    l_rate = tf.compat.v1.placeholder('float')
    rate = tf.compat.v1.placeholder('float')

    ##########################
    #########网络参数#########
    ##########################
    ###初始值
    w_min_init = np.multiply(np.zeros([dataX.shape[1], units_each_group*number_of_groups]) + \
        1/(units_each_group*number_of_groups),np.random.uniform(0.9,1.1,(units_each_group*number_of_groups,)))

    # w_min_init = tf.constant(w_min_init, dtype=tf.float32)

    w_min_init = tf.constant(np.zeros_like(w_min_init), dtype=tf.float32)
    w_min_mask_tf = tf.constant(w_min_mask,dtype=tf.float32)
    # b_min_mask_tf = tf.constant(np.sum(w_min_mask,axis=0)>0,dtype=tf.float32)

    with tf.compat.v1.variable_scope('weights'):
        w0 = tf.constant(w_logistic,dtype=tf.float32)
        w_min = tf.multiply(w_min_mask_tf,tf.Variable(w_min_init, dtype=tf.float32,constraint=make_greater_than_1), name='w_min')

    with tf.compat.v1.variable_scope('biases', initializer=tf.random_normal_initializer()):
        b0 = tf.constant(b_logistic,dtype=tf.float32)
        # b0 = tf.constant(0,dtype=tf.float32)
        b_min = tf.Variable(tf.zeros([units_each_group*number_of_groups]), dtype=tf.float32, name='b_min')

    tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(l2_reg)(w_min))

    # tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(l2_reg)(w0))

    ##########################
    #########网络构建#########
    ##########################
    h0 = tf.multiply(x,w0)
    h_min = tf.matmul(tf.nn.dropout(h0,rate=rate),w_min)+b_min
    # h_min_2d = tf.expand_dims(h_min,0)
    h_min_3d = tf.expand_dims(h_min,1)
    h_min_4d = tf.expand_dims(h_min_3d,3)
    h_min_pool = -tf.nn.max_pool(-h_min_4d,ksize=[1,1,units_each_group,1],strides=[1,1,units_each_group,1],padding='SAME')
    h_max_pool = tf.nn.max_pool(h_min_pool,ksize=[1,1,number_of_groups,1],strides=[1,1,number_of_groups,1],padding='SAME')
    h_max_pool_sq = tf.squeeze(h_max_pool,axis=[2,3])


    y_linear = h_max_pool_sq

    y_proba = tf.sigmoid(y_linear)
    if target == 'classifier':
        loss = tf.reduce_mean(-y * tf.math.log(tf.clip_by_value(y_proba, 1e-10, 1.0 - 1e-10)) -
            (1 - y) * tf.math.log(tf.clip_by_value(1 - y_proba, 1e-10, 1.0 - 1e-10)))
        loss_pointwise = -y * tf.math.log(tf.clip_by_value(y_proba, 1e-10, 1.0 - 1e-10)) - \
            (1 - y) * tf.math.log(tf.clip_by_value(1 - y_proba, 1e-10, 1.0 - 1e-10))
    elif target == 'regression':
        loss = tf.reduce_mean(tf.compat.v1.losses.mean_squared_error(y,y_linear))
        loss_pointwise = tf.square(y-y_linear)
    tf.add_to_collection("losses",loss)
    loss = tf.add_n(tf.get_collection("losses"))

    global_step = tf.Variable(0)
    train_step = tf.compat.v1.train.AdamOptimizer(l_rate).minimize(loss, global_step=global_step)

    ##########################
    #########参数训练#########
    ##########################

    sess = tf.compat.v1.Session()

    init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
    sess.run(init_op)
    Loss = []
    for i in range(100):
        index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]

        batch_X = dataX[index_batch]
        batch_Y = dataY[index_batch]

        sess.run(train_step, feed_dict={x: batch_X, y: batch_Y, rate:dropout_ratio,l_rate:learning_rate})
        if i > 19:
            Loss.append(sess.run(loss, feed_dict={x: dataX, y: dataY, rate:0.0}))

    step_ = 2

    start_embedding = False
    w_min_list = []
    b_min_list = []
    weight_embedding = []
    loss_diff_track = []

    while len(w_min_list) <= num_model:
        if not start_embedding:
            if ((Loss[-2] - Loss[-1]) / (np.abs(Loss[-2]) + 1e-10) < .025) and \
                ((Loss[-5] - Loss[-1]) / (np.abs(Loss[-5]) + 1e-10) < .025) and \
                ((Loss[-10] - Loss[-1]) / (np.abs(Loss[-10]) + 1e-10) < .05) and \
                (((Loss[-int(step_/10)] - Loss[-1]) / (np.abs(Loss[-int(step_/10)]) + 1e-10) < .1) and ((Loss[-int(step_/10)] - Loss[-1]) < 0)) and \
                (((Loss[-int(step_/5)] - Loss[-1]) / (np.abs(Loss[-int(step_/5)]) + 1e-10) < .15) and ((Loss[-int(step_/5)] - Loss[-1]) < 0)) and \
                (Loss[-1] < min(Loss[-int(step_/5):]) + (max(Loss[-int(step_/5):])-min(Loss[-int(step_/5):]))/3) and \
                step_ > int(max_iter/4) or\
                step_ >= max_iter:

                if step_>max_iter:
                    warnings.warn("ConvergenceWarning : MNN failed to converge")

                start_embedding = True
                loss_now = sess.run(loss, feed_dict={x: dataX, y: dataY, rate:0.0})
                weight_embedding.append(loss_now)
                w_min_list.append(sess.run(w_min))
                b_min_list.append(sess.run(b_min))

        # if ((Loss[-2] - Loss[-1]) / (np.abs(Loss[-2]) + 1e-10) < .05) and \
        #     ((Loss[-5] - Loss[-1]) / (np.abs(Loss[-5]) + 1e-10) < .1) and \
        #     ((Loss[-10] - Loss[-1]) / (np.abs(Loss[-10]) + 1e-10) < .15) and \
        #     ((Loss[-int(step_/10)] - Loss[-1]) < 0) and \
        #     ((Loss[-int(step_/5)] - Loss[-1]) < 0) and \
        #     ((Loss[-int(step_/2)] - Loss[-1]) < 0) and \
        #     step_ > int(max_iter/5) or\
        #     step_ > int(max_iter/2):

        #     start_embedding = True

        step_ += 1
        index_batch = [random.choices(range(dataY.shape[0]))[0] for i in range(batch_size)]

        batch_X = dataX[index_batch]
        batch_Y = dataY[index_batch]

        if (Q1<1 or Q0<1) and target=='classifier':
            batch_Y = np.reshape(batch_Y,(len(batch_Y),))
            batch_X_white = batch_X[batch_Y==0]
            batch_Y_white = batch_Y[batch_Y==0]
            batch_X_black = batch_X[batch_Y==1]
            batch_Y_black = batch_Y[batch_Y==1]
            white_NoOutlier = []
            black_NoOutlier = []
            if len(batch_X_white) > 0:
                Loss_white = sess.run(loss_pointwise,feed_dict={x:batch_X_white,y:np.reshape(batch_Y_white,(batch_Y_white.shape[0],1)),rate:0.0})
                Loss_white = np.reshape(Loss_white,max(Loss_white.shape))
                white_NoOutlier = np.array(Loss_white<=np.quantile(Loss_white,Q0))
            if len(batch_X_black) > 0:
                Loss_black = sess.run(loss_pointwise,feed_dict={x:batch_X_black,y:np.reshape(batch_Y_black,(batch_Y_black.shape[0],1)),rate:0.0})
                Loss_black = np.reshape(Loss_black,max(Loss_black.shape))
                black_NoOutlier = np.array(Loss_black<=np.quantile(Loss_black,Q1))

            if len(batch_X_white) == 0:
                batch_X = batch_X_black[black_NoOutlier]
                batch_Y = batch_Y_black[black_NoOutlier]
            elif len(batch_X_black) == 0:
                batch_X = batch_X_white[black_NoOutlier]
                batch_Y = batch_Y_white[black_NoOutlier]
            else:
                batch_X = np.concatenate((batch_X_white[white_NoOutlier],batch_X_black[black_NoOutlier]))
                batch_Y = np.concatenate((batch_Y_white[white_NoOutlier],batch_Y_black[black_NoOutlier]))

            batch_Y = np.reshape(batch_Y,(len(batch_Y),1))

        if target=='regression' and Q<1:
            batch_Y = np.reshape(batch_Y,(len(batch_Y),))
            Loss_batch = sess.run(loss_pointwise,feed_dict={x:batch_X,y:np.reshape(batch_Y,(batch_Y.shape[0],1)),rate:0.0})
            Loss_batch = np.reshape(Loss_batch,max(Loss_batch.shape))
            batch_NoOutlier = np.array(Loss_batch<=np.quantile(Loss_batch,Q1))
            batch_X = batch_X[batch_NoOutlier]
            batch_Y = batch_Y[batch_NoOutlier]

            batch_Y = np.reshape(batch_Y,(len(batch_Y),1))

        if not start_embedding:
            sess.run(train_step, feed_dict={x: batch_X, y: batch_Y, rate:dropout_ratio, l_rate:learning_rate})
        else:
            c = 300
            learning_rate_2 = learning_rate*20
            t_i = 1/c*((step_-1)%c+1)
            if t_i <= 0.5:
                l_iter = (1-2*t_i)*learning_rate_2+2*t_i*learning_rate
                sess.run(train_step, feed_dict={x: batch_X, y: batch_Y, rate:dropout_ratio, l_rate:l_iter})
            else:
                l_iter = (2-2*t_i)*learning_rate+(2*t_i-1)*learning_rate_2
                sess.run(train_step, feed_dict={x: batch_X, y: batch_Y, rate:dropout_ratio, l_rate:l_iter})

            if t_i==0.5:
                loss_now = sess.run(loss, feed_dict={x: dataX, y: dataY, rate:0.0})
                if loss_now < Loss[-5]:
                    loss_diff_track.append(1)
                else:
                    loss_diff_track.append(0)
                weight_embedding.append(loss_now)
                w_min_list.append(sess.run(w_min))
                b_min_list.append(sess.run(b_min))

        Loss.append(sess.run(loss, feed_dict={x: dataX, y: dataY, rate:0.0}))

        if step_ % int(max_iter/10) == 0 and trace:
            time_ = datetime.datetime.now().strftime('%Y-%D %H:%M:%S')
            print(time_ + '\t' + str(step_) + '\t' + str(Loss[-1]))



    w_min_ = np.zeros_like(w_min_list[0])
    b_min_ = np.zeros_like(b_min_list[0])
    weight_embedding = np.array(np.max(weight_embedding)-weight_embedding)**2
    weight_embedding = weight_embedding/np.sum(weight_embedding)
    for i in range(len(w_min_list)):
        w_min_ += w_min_list[i]*weight_embedding[i]
        b_min_ += b_min_list[i]*weight_embedding[i]

    # total_weight = 0
    # for i in range(len(w_min_list)):
    #     w_min_ += w_min_list[i]*(i+0.5)**2
    #     b_min_ += b_min_list[i]*(i+0.5)**2
    #     total_weight += (i+0.5)**2

    # w_min_ = w_min_/total_weight
    # b_min_ = b_min_/total_weight
    # print(sum(loss_diff_track)/len(loss_diff_track))

    # w_min_ = w_min_list[-1]
    # b_min_ = b_min_list[-1]

    w0_ = sess.run(w0)
    b0_ = sess.run(b0)

    return (x,y,y_linear,y_proba,rate,sess,Loss,w0_,w_min_,b0_,b_min_)

