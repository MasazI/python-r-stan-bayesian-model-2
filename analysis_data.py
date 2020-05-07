import pandas
import matplotlib.pyplot as plt
import seaborn as sns


class AnalysisData():
    def __init__(self, file: str, model_file: str):
        self.file = file
        self.data = pandas.read_csv(file)
        self.model_file = model_file

    def describe(self):
        print(self.data.head())
        print(self.data.describe())

    def hist(self):
        ax = self.data.plot.hist(bins=30, alpha=0.5, legend=True)
        self.data.plot.kde(legend=True, alpha=0.5, secondary_y=True, ax=ax)
        plt.show()

    def observe(self, hue):
        sns.pairplot(self.data, diag_kind='hist', hue=hue)
        plt.show()
        plt.close()


class AnalysisData2Dim():
    def __init__(self, file: str, file2: str, model_file: str):
        self.file = file
        self.data = pandas.read_csv(file, header=None)
        self.process = pandas.read_csv(file2, header=None)
        self.model_file = model_file

    def describe(self):
        print(self.data.head())
        print(self.data.describe())

        print(self.process.head())
        print(self.process.describe())
