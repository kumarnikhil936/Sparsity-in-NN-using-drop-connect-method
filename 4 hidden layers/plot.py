# libraries and data
import matplotlib.pyplot as plt
import pandas

uniaccuracyTest1 = pandas.read_csv("uniformDropconnectAccuracyTestWithRetrain.csv", sep=',', header=None)
uniaccuracyTrain1 = pandas.read_csv("uniformDropconnectAccuracyTrainWithRetrain.csv", sep=',', header=None)

uniaccuracyTest2 = pandas.read_csv("uniformDropconnectAccuracyTestNoRetrain.csv", sep=',', header=None)
uniaccuracyTrain2 = pandas.read_csv("uniformDropconnectAccuracyTrainNoRetrain.csv", sep=',', header=None)

dl1accuracyTest1 = pandas.read_csv("dl1AccuracyTestWithRetrain.csv", sep=',', header=None)
dl1accuracyTrain1 = pandas.read_csv("dl1AccuracyTrainWithRetrain.csv", sep=',', header=None)

dl1accuracyTest2 = pandas.read_csv("dl1AccuracyTestNoRetrain.csv", sep=',', header=None)
dl1accuracyTrain2 = pandas.read_csv("dl1AccuracyTrainNoRetrain.csv", sep=',', header=None)

dl2accuracyTest1 = pandas.read_csv("dl2AccuracyTestWithRetrain.csv", sep=',', header=None)
dl2accuracyTrain1 = pandas.read_csv("dl2AccuracyTrainWithRetrain.csv", sep=',', header=None)

dl2accuracyTest2 = pandas.read_csv("dl2AccuracyTestNoRetrain.csv", sep=',', header=None)
dl2accuracyTrain2 = pandas.read_csv("dl2AccuracyTrainNoRetrain.csv", sep=',', header=None)

dl3accuracyTest1 = pandas.read_csv("dl3AccuracyTestWithRetrain.csv", sep=',', header=None)
dl3accuracyTrain1 = pandas.read_csv("dl3AccuracyTrainWithRetrain.csv", sep=',', header=None)

dl3accuracyTest2 = pandas.read_csv("dl3AccuracyTestNoRetrain.csv", sep=',', header=None)
dl3accuracyTrain2 = pandas.read_csv("dl3AccuracyTrainNoRetrain.csv", sep=',', header=None)

dl4accuracyTest1 = pandas.read_csv("dl4AccuracyTestWithRetrain.csv", sep=',', header=None)
dl4accuracyTrain1 = pandas.read_csv("dl4AccuracyTrainWithRetrain.csv", sep=',', header=None)

dl4accuracyTest2 = pandas.read_csv("dl4AccuracyTestNoRetrain.csv", sep=',', header=None)
dl4accuracyTrain2 = pandas.read_csv("dl4AccuracyTrainNoRetrain.csv", sep=',', header=None)

dlInputaccuracyTest1 = pandas.read_csv("dlinputAccuracyTestWithRetrain.csv", sep=',', header=None)
dlInputaccuracyTrain1 = pandas.read_csv("dlinputAccuracyTrainWithRetrain.csv", sep=',', header=None)

dlInputaccuracyTest2 = pandas.read_csv("dlinputAccuracyTestNoRetrain.csv", sep=',', header=None)
dlInputaccuracyTrain2 = pandas.read_csv("dlinputAccuracyTrainNoRetrain.csv", sep=',', header=None)

x = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]

plt.figure(1)
plt.xlabel('Percentage of weights kept from 5% to 100%')
plt.ylabel('Accuracy')
plt.grid(True)
plt.title('\n4 Hidden layers : Accuracy for sparsity applied uniformly over all hidden layers with retraining')
plt.plot(x, uniaccuracyTest1, linewidth=2, markersize=5, label="test")
plt.plot(x, uniaccuracyTrain1, linewidth=2, markersize=5, label="train")
# plt.show()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gcf().set_size_inches(18.5, 10.5)
plt.savefig('uniformDropConnectWithRetrain.png')

plt.figure(2)
plt.xlabel('Percentage of weights kept from 5% to 100%')
plt.ylabel('Accuracy')
plt.grid(True)
plt.title('\n4 Hidden layers : Accuracy for sparsity applied uniformly over all hidden layers without retraining')
plt.plot(x, uniaccuracyTest2, linewidth=2, markersize=5, label="test")
plt.plot(x, uniaccuracyTrain2, linewidth=2, markersize=5, label="train")
# plt.show()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gcf().set_size_inches(18.5, 10.5)
plt.savefig('uniformDropConnectNoRetrain.png')

plt.figure(3)
plt.xlabel('Percentage of weights kept from 5% to 100%')
plt.ylabel('Accuracy')
plt.grid(True)
plt.title('\n4 Hidden layers : Accuracy for sparsity applied between hidden layer 1 and 2 with retraining')
plt.plot(x, dl1accuracyTest1, linewidth=2, markersize=5, label="test")
plt.plot(x, dl1accuracyTrain1, linewidth=2, markersize=5, label="train")
# plt.show()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gcf().set_size_inches(18.5, 10.5)
plt.savefig('dl1DropConnectWithRetrain.png')

plt.figure(4)
plt.xlabel('Percentage of weights kept from 5% to 100%')
plt.ylabel('Accuracy')
plt.grid(True)
plt.title('\n4 Hidden layers : Accuracy for sparsity applied between hidden layer 1 and 2 without retraining')
plt.plot(x, dl1accuracyTest2, linewidth=2, markersize=5, label="test")
plt.plot(x, dl1accuracyTrain2, linewidth=2, markersize=5, label="train")
# plt.show()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gcf().set_size_inches(18.5, 10.5)
plt.savefig('dl1DropConnectNoRetrain.png')

plt.figure(5)
plt.xlabel('Percentage of weights kept from 5% to 100%')
plt.ylabel('Accuracy')
plt.grid(True)
plt.title('\n4 Hidden layers : Accuracy for sparsity applied between hidden layer 2 and 3 with retraining')
plt.plot(x, dl2accuracyTest1, linewidth=2, markersize=5, label="test")
plt.plot(x, dl2accuracyTrain1, linewidth=2, markersize=5, label="train")
# plt.show()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gcf().set_size_inches(18.5, 10.5)
plt.savefig('dl2DropConnectWithRetrain.png')

plt.figure(6)
plt.xlabel('Percentage of weights kept from 5% to 100%')
plt.ylabel('Accuracy')
plt.grid(True)
plt.title('\n4 Hidden layers : Accuracy for sparsity applied between hidden layer 2 and 3 without retraining')
plt.plot(x, dl2accuracyTest2, linewidth=2, markersize=5, label="test")
plt.plot(x, dl2accuracyTrain2, linewidth=2, markersize=5, label="train")
# plt.show()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gcf().set_size_inches(18.5, 10.5)
plt.savefig('dl2DropConnectNoRetrain.png')

plt.figure(7)
plt.xlabel('Percentage of weights kept from 5% to 100%')
plt.ylabel('Accuracy')
plt.grid(True)
plt.title('\n4 Hidden layers : Accuracy for sparsity applied between input layer and hidden layer 1 with retraining')
plt.plot(x, dlInputaccuracyTest1, linewidth=2, markersize=5, label="test")
plt.plot(x, dlInputaccuracyTrain1, linewidth=2, markersize=5, label="train")
# plt.show()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gcf().set_size_inches(18.5, 10.5)
plt.savefig('dlinputDropConnectWithRetrain.png')

plt.figure(8)
plt.xlabel('Percentage of weights kept from 5% to 100%')
plt.ylabel('Accuracy')
plt.grid(True)
plt.title('\n4 Hidden layers : Accuracy for sparsity applied between input layer and hidden layer 1 without retraining')
plt.plot(x, dlInputaccuracyTest2, linewidth=2, markersize=5, label="test")
plt.plot(x, dlInputaccuracyTrain2, linewidth=2, markersize=5, label="train")
# plt.show()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gcf().set_size_inches(18.5, 10.5)
plt.savefig('dlinputDropConnectNoRetrain.png')

plt.figure(9)
plt.xlabel('Percentage of weights kept from 5% to 100%')
plt.ylabel('Accuracy')
plt.grid(True)
plt.title('\n4 Hidden layers : Accuracy for sparsity applied between hidden layer 3 and 4 with retraining')
plt.plot(x, dl3accuracyTest1, linewidth=2, markersize=5, label="test")
plt.plot(x, dl3accuracyTrain1, linewidth=2, markersize=5, label="train")
# plt.show()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gcf().set_size_inches(18.5, 10.5)
plt.savefig('dl3DropConnectWithRetrain.png')

plt.figure(10)
plt.xlabel('Percentage of weights kept from 5% to 100%')
plt.ylabel('Accuracy')
plt.grid(True)
plt.title('\n4 Hidden layers : Accuracy for sparsity applied between hidden layer 3 and 4 without retraining')
plt.plot(x, dl3accuracyTest2, linewidth=2, markersize=5, label="test")
plt.plot(x, dl3accuracyTrain2, linewidth=2, markersize=5, label="train")
# plt.show()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gcf().set_size_inches(18.5, 10.5)
plt.savefig('dl3DropConnectNoRetrain.png')

plt.figure(11)
plt.xlabel('Percentage of weights kept from 5% to 100%')
plt.ylabel('Accuracy')
plt.grid(True)
plt.title('\n4 Hidden layers : Accuracy for sparsity applied between hidden layer 4 and output layer with retraining')
plt.plot(x, dl4accuracyTest1, linewidth=2, markersize=5, label="test")
plt.plot(x, dl4accuracyTrain1, linewidth=2, markersize=5, label="train")
# plt.show()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gcf().set_size_inches(18.5, 10.5)
plt.savefig('dl4DropConnectWithRetrain.png')

plt.figure(12)
plt.xlabel('Percentage of weights kept from 5% to 100%')
plt.ylabel('Accuracy')
plt.grid(True)
plt.title('\n4 Hidden layers : Accuracy for sparsity applied between hidden layer 4 and output layer without retraining')
plt.plot(x, dl4accuracyTest2, linewidth=2, markersize=5, label="test")
plt.plot(x, dl4accuracyTrain2, linewidth=2, markersize=5, label="train")
# plt.show()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gcf().set_size_inches(18.5, 10.5)
plt.savefig('dl4DropConnectNoRetrain.png')
