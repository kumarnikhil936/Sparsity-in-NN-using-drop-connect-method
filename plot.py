import pandas
import matplotlib.pyplot as plt

data = pandas.read_csv("data.csv", sep=',')
data.boxplot(column='Accuracy', by='Keep probability')

plt.show()
# plt.yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
