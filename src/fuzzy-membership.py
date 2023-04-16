import numpy as np
import matplotlib.pyplot as plt
import skfuzzy
import pandas as pd


class Membership:
    def __init__(self) -> None:
        
        self.weekDaysList = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        self.weekendDaysList = ['Saturday', 'Sunday']

    def assign(self):

        self.tables = {}

        for day in self.weekDaysList:
            
            data = pd.read_excel('./assets/traffic-analysis-dataframe.xlsx', sheet_name=day)

            speed = data['Avg Speed Forward']

            hour = data['Hour']

            self.tables[f'{day}'] = pd.DataFrame({'hour': hour, 'degree': self.smf(speed)})
                

    def smf(self, data):
    
        max = data.max()
        min = data.min()

        return skfuzzy.membership.smf(data, min, max)
    
    def polynomialModel(self):
        
        for day in self.weekDaysList:

            data = self.tables[f'{day}']

            plt.scatter(data.hour, data.degree)

            model = np.poly1d(np.polyfit(data.hour, data.degree, 18))
            polyline = np.linspace(0, 23, 24)

            plt.scatter(data.hour, data.degree)

            plt.plot(polyline, model(polyline), color="red")
            plt.show()

        pass

    def tables(self):
        return self.tables
    

membership = Membership()

membership.assign()
