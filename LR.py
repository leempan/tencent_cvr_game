import numpy as numpy
import cPickle
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
import csv
import numpy as np
import time
train_file = '/home/administrator/Limingpan/pre/merged_new_fea_train.csv'
test_file = '/home/administrator/Limingpan/pre/merged_new_fea_test.csv'
maxiter = 3
def sparse(x,number):
    sparse_x = a = np.zeros(number*len(x))
    iid = 0
    for _x in x:
        a[iid+_x] = 1
        iid = iid+number
    # print sparse_x.shape
    return sparse_x
def iter_minibatches(data_stream, minibatch_size=1000):    
    X = []
    y = []
    cur_line_num = 0

    csvfile = file(data_stream, 'rb')
    reader = csv.reader(csvfile)

    for line in reader:
        y.append(float(line[0]))
        numbers = sparse([int(x) for x in line[1:]],512)
        # print numbers.shape
        X.append(numbers)  
        cur_line_num += 1
        if cur_line_num >= minibatch_size:
            X, y = np.array(X), np.array(y)
            yield X, y
            X, y = [], []
            cur_line_num = 0
    csvfile.close()


def train(train_file):
    sgd_clf = SGDClassifier(loss='log',penalty='l2',learning_rate='optimal',shuffle=True,class_weight={0:1,1:50},alpha=1)  

    for it in range(maxiter):
        tick = time.time()
        minibatch_train_iterators = iter_minibatches(train_file, minibatch_size=2000)
        for i, (X_train, y_train) in enumerate(minibatch_train_iterators):
            print X_train.shape
            print 'iter',it,',',i,'th minibatch training,',sum(y_train),'positive samples'
            sgd_clf.partial_fit(X_train, y_train, classes=np.array([0, 1]))
        tick1 = time.time()
        print 'Iter',it,'training finished. cost time:',tick1-tick,'seconds'

#     with open('LR_classifier.pkl', 'wb') as fid:
#         cPickle.dump(sgd_clf, fid)    
def test(testfile):
    result = open('submission.csv','w')
    with open('LR_classifier.pkl', 'rb') as fid:
        sgd_clf = cPickle.load(fid)
    print sgd_clf.coef_,sgd_clf.coef_.shape    
    print sgd_clf.intercept_,sgd_clf.intercept_.shape    
    test_csv = csv.reader(file(test_file,'rb'))
    result.write('instanceID,prob\n')
    id = 1
    for line in test_csv:
        X = np.array([int(x) for x in line]).reshape(1,-1)
        # print X,X.shape
        prob = sgd_clf.decision_function(X)
        
        print str(id)+','+str(prob)
        # result.write(str(id)+','+str(prob[0][1])+'\n')
        id = id+1
    result.close()
# load it again
if __name__ == '__main__':
    # phase = True
    train(train_file)
    test(test_file)
