import matplotlib.pyplot as plt

def PlotHistoryNew(_model, feature, start_epoch = 0, path_file = None):
    val = "val_" + feature
    
    plt.xlabel('Epoch Number - ' + str(start_epoch))
    plt.ylabel(feature)
    plt.plot(_model.history[feature][start_epoch:])
    plt.plot(_model.history[val][start_epoch:])
    plt.legend(["train_"+feature, val])    
    if path_file:
        plt.savefig(path_file)