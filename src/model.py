
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from fuzzy_membership import Membership 
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt



class Analysis: 
    def __init__(self) -> None:

        self.df = Membership().df

        self.shuffled_df = self.df.sample(frac=1, ignore_index=True)

        self.trafficData = pd.get_dummies(self.shuffled_df, columns=["day"], drop_first=True)

        self.scaler = MinMaxScaler()
    
    def initiate(self):

        inputs = self.trafficData.drop(['degree'], axis=1)

        outputs = self.trafficData['degree']

        x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, train_size=0.67, random_state=2)

        return x_train, x_test, y_train, y_test
    
    def scale(self, xTrain, xTest):
        self.scaler.fit(xTrain)

        return self.scaler.transform(xTrain), self.scaler.transform(xTest)

    def model(self, act, solver, alpha):
        isStopping = False
        batchSize = None
        if solver == "adam" or solver == "sgd":
            isStopping = True   
            batchSize = 100
        else: 
            isStopping = False
            batchSize = None
        
        return MLPRegressor(hidden_layer_sizes=(100,100,), activation=act, max_iter=self.max_iter, solver=solver, alpha=alpha, max_fun=1500, early_stopping=isStopping, batch_size=batchSize)    

    def train(self, act, solver, alpha):

        self.x_train, self.x_test, self.y_train, self.y_test = self.initiate()

        self.x_train_s, self.x_test_s = self.scale(self.x_train, self.x_test)

        self.nn = self.model(act, solver, alpha)

        self.nn.fit(self.x_train_s,self.y_train)

        self.predicts = self.nn.predict(self.x_train_s)
        
    def scores(self):

        mae = metrics.mean_absolute_error(self.y_train, self.predicts)
        mse = metrics.mean_squared_error(self.y_train, self.predicts)
        rsq = metrics.r2_score(self.y_train, self.predicts)

        print(mae, mse, rsq)
        print(self.nn.score(self.x_test_s, self.y_test))
        print("Loss")
        print(self.nn.loss_)
        
    def visualize(self, saveName = None):
        self.plt = plt
        fig, axes = self.plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

        self.plt.xlabel("Hour")
        self.plt.ylabel("Membership Degree")
        self.plt.ylim(-0.5, 1.5)
        self.plt.xticks(np.arange(min(self.x_train["hour"]), max(self.x_train["hour"])+1, 1.0))
        self.plt.title(f"Activation Func:{self.actName.capitalize()}, Optimizer: {self.solverName.capitalize()}, Alpha: {self.alphaValue}, Iteration: {self.max_iter}")

        axes.plot(self.x_train["hour"], self.y_train, "o", color="blue", label="Train Value", alpha=0.5)
        axes.plot(self.x_train["hour"], self.predicts, "o", color="red", label="Predicted Value", alpha=0.5)
        axes.grid(True)
        axes.legend(loc=4)

        self.plt.tight_layout()        

        if saveName:
            self.plt.savefig(f'assets/{saveName}.png')
        else:
            self.plt.savefig(f'assets/model.png')
                 
    def plotShow(self):
        self.plt.show()

        try:
            plt.title("Loss Curve of Training Set for each iteration")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.plot(pd.DataFrame(self.nn.loss_curve_), label="Loss Function")

            self.loss_plt = plt

            self.loss_plt.show()
        except:
            print("No loss curve has found")   

    def analyze(self, act, solver, alpha, max_iter=100):
        self.max_iter = max_iter
        self.train(act, solver, alpha)
        self.actName = act
        self.solverName = solver
        self.alphaValue = alpha
        

dataAnalysis = Analysis()


# Logistic

# print("Logistic Results")
# dataAnalysis.analyze("logistic", "lbfgs", 0.001)
# dataAnalysis.scores()
# dataAnalysis.visualize(saveName="logistic_model")
# dataAnalysis.plotShow()
# print("*******************************************")

#Tanh

# print("Tanh Results")
# dataAnalysis.analyze("tanh", "adam", 0.0001)
# dataAnalysis.scores()
# dataAnalysis.visualize(saveName="tanh_model")

# ReLu
print("ReLU Results")
dataAnalysis.analyze("relu", "adam", 0.001, 500)
dataAnalysis.scores()
dataAnalysis.visualize()
dataAnalysis.plotShow()