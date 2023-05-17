
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from fuzzy_membership import Membership 
from sklearn import metrics
import pandas as pd


class Analysis: 
    def __init__(self) -> None:

        self.df = Membership().df

        self.shuffled_df = self.df.sample(frac=1, ignore_index=True)

        self.trafficData = pd.get_dummies(self.shuffled_df, columns=["day"], drop_first=True)


    
    def initiate(self):


        pass


membership = Membership()

shuffled_df = membership.df.sample(frac=1, ignore_index=True)

df = pd.get_dummies(shuffled_df, columns=["day"], drop_first=True)

inputs = df.drop(['degree'], axis=1)

outputs = df['degree']

x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, train_size=0.6, random_state=2)


scaler = MinMaxScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


nn = MLPRegressor(hidden_layer_sizes=(100,100,), activation="tanh", max_iter=100)

nn.fit(x_train, y_train)

predicts = nn.predict(x_train)

mae = metrics.mean_absolute_error(y_train,predicts)
mse = metrics.mean_squared_error(y_train,predicts)
rsq = metrics.r2_score(y_train,predicts)


print(mae, mse, rsq)

print(nn.score(x_test, y_test))