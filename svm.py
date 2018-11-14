from sklearn import svm
from sklearn.externals import joblib
from sklearn import svm
import numpy as np
import os
import sys

def read_data(src):
    files = os.listdir(src)
    labels = []
    datas = []
    for i, file in enumerate(files):
        temp = os.path.join(src, file)        
        data = np.load(temp)        
        label = -1
        if file.find('cat') != -1:
            label = 0

        else:
            label = 1
        print("[+] Load file : ", temp , " with label : ",label, " shape : ", data.shape, " data : ", data)
        datas.append(data[0])
        labels.append(label)
    return datas, labels

if __name__ == '__main__':
    src = '/content/svm_cats_and_dogs/features/vgg16_fc2/train'
    datas, labels = read_data(src)
    print("Training...")
    clf = svm.SVC(gamma='scale')
    clf.fit(datas, labels) 
    print("Train finished")
    save_path = "model/svm.joblib"
    print("Save model to file : ", save_path)
    joblib.dump(clf, save_path)
