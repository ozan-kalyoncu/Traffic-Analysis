import numpy as np
import matplotlib.pyplot as plt
import skfuzzy
import pandas as pd
import numpy as np

class Membership:
    def __init__(self) -> None:
        
        self.weekDaysList = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        self.weekendDaysList = ['Saturday', 'Sunday']

        self.df = pd.DataFrame([], index=["hour", "degree", "duration", "dist", "day"]).T

        self.assign()

    def assign(self):

        for day in self.weekDaysList:
            
            data = pd.read_excel('./assets/traffic-analysis-dataframe.xlsx', sheet_name=day)

            speed = data['Avg Speed Forward']

            hour = data['Hour']

            tableu = pd.DataFrame([hour, self.smf(speed), data["Duration Forward"], data["Distance Forward"]], index=["hour", "degree", "duration", "dist"]).T

            tableu = tableu.groupby("hour", group_keys=False).mean()

            tableu["hour"] = hour
            tableu['day'] = day

            self.df = pd.concat([self.df, tableu], ignore_index=True)
                        

    def smf(self, data):
    
        max = data.max()
        min = data.min()

        return abs(skfuzzy.membership.smf(data, min, max) - 1)
    
    def polynomialModel(self):
        
        for day in self.weekDaysList:

            data = self.df[self.df["day"] == day]

            plt.scatter(data.hour, data.degree)

            model = np.poly1d(np.polyfit(data.hour, data.degree, 18))
            polyline = np.linspace(0, 23, 24)

            plt.scatter(data.hour, data.degree)

            plt.plot(polyline, model(polyline), color="red")
            plt.show()

        pass


membership = Membership()


# Saving new data schema

# dataToExcel = pd.ExcelWriter('TrafficAnalysisDataFrame.xlsx')

# membership.df.to_excel(dataToExcel)

# dataToExcel.save()