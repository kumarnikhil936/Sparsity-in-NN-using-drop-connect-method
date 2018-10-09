# libraries and data
import matplotlib.pyplot as plt
import pandas

accuracyTest = pandas.read_csv("accuracyTest.csv", sep=',')
accuracyTrain = pandas.read_csv("accuracyTrain.csv", sep=',')

plt.figure(1)
plt.subplot(211)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('8 Hidden layers : Accuracy over Test data')
plt.plot(accuracyTest, linewidth=2, markersize=5)
plt.grid(True)

plt.subplot(212)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('8 Hidden layers : Accuracy over Training data')
plt.plot(accuracyTrain, linewidth=2, markersize=5)
plt.grid(True)
# plt.show()

plt.gcf().set_size_inches(18.5, 10.5)
plt.savefig('8HiddenLayers.png')

