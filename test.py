from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import numpy as np
import os
import sys
import csv
from classifier import read_data

if __name__ == '__main__':
    test_folder = sys.argv[1]
    print("[!] Load model...")
    knn = joblib.load('model/knn.joblib')
    print("[+] Load finished")
    with open('result.csv', 'w') as csvfile:
        fieldnames = ['image_name', 'predict_class']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()        
        files = os.listdir(test_folder)
        for file in files:
            print("[+] Read file : " , file)
            data = np.load(os.path.join(test_folder, file))
            label = knn.predict(data)
            file_name = file.replace('.npy', '.jpg')
            class_name = None
            if label == 1:
                class_name = 'dog'
            else:
                class_name = 'cat'
            writer.writerow({'image_name' : file_name, 'predict_class' : class_name})
    print("[+] Finished, view result.csv for result")

