import pickle
import os
from matplotlib import pyplot as plt
import os

DIR = "./history/"
PLOTS_DIR = './plots_LSTM/'

for filename in os.listdir(DIR):
    FILENAME = filename
    with open(DIR+FILENAME, 'rb') as handle:
        history = pickle.load(handle)

    print('model ',FILENAME)
    leng = len(history['accuracy'])-1
    print('accuracy ', history['accuracy'][leng])
    print('val accuracy ', history['val_accuracy'][leng])
    print('loss ', history['loss'][leng])
    print('val loss ', history['val_loss'][leng])
    print('--------')
    #  "Accuracy"
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title(FILENAME+' model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(PLOTS_DIR+FILENAME+'_accuracy.pdf')  
    plt.clf()
    #  "Loss"
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(FILENAME+' model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(PLOTS_DIR+FILENAME+'_loss.pdf')  
    plt.clf()