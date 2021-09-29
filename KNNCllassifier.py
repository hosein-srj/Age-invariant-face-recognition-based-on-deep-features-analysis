import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import cross_val_score

def main():
    f = open('features.txt', 'r')
    labels = np.zeros((1002))
    datas = []
    for i in range(1002):
        line = f.readline()
        line = line.replace('\n','')
        line = line.rstrip()
        line = line.split(' ')
        line = np.array(line)
        line = line.astype(np.float)
        labels[i] = int(line[0])
        datas.append(line[1:82])
    datas = np.array(datas)

    clf = svm.SVC(kernel='linear', C=2, random_state=64)
    scores = cross_val_score(clf, datas, labels, cv=5,)
    print(np.mean(scores))

    print(scores)




if "__main__":
    main()
