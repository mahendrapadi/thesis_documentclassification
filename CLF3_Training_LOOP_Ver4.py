from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression
from operator import itemgetter

from sklearn import svm
import numpy as np
import pandas
import pickle
import os
import sys
import shutil
from langdetect import detect

#####################################################################################                                    >>>>>>>>>>>>  Variable diclaration

performance_init = 0
performance_train = 0.0001

performance_MNB = 0
performance_LR = 0
performance_SGD = 0


Training_docs_yes = []
Training_docs_no = []
Prediction_docs = []

performance=[]


X_train_MNB = 0
X_train_LR = 0
X_train_SGD = 0


iteration = 1
loop = 0

while  performance_train > performance_init:
#while loop < 10:
    loop=loop+1

    performance_init = performance_train
    performance_list = []

    # >>>>>>>>>>>>>      Data paths

    path_train = "C:\Users\D065921\Documents\Mahi\Thesis\Crawling\Crawled Data\last versions\inal_versions\practice\Sports\Sports_Training"

    path_test = "C:\Users\D065921\Documents\Mahi\Thesis\Crawling\Crawled Data\last versions\inal_versions\practice\Sports\Sports_Development"

    path_predict = "C:\Users\D065921\Documents\Mahi\Thesis\Crawling\Crawled Data\last versions\inal_versions\practice\Sports\Unknown"

    # >>>>>>>>>>>>>>>>>   Language detection on training files

    for root,dirs,files in os.walk(path_train):
        for file in files:

            os.chdir(root)
            open_file = open(file,'r')
            read_file = open_file.read()

            conv_file = unicode(read_file,encoding='utf-8',errors='ignore')

            lan = detect(conv_file)
            open_file.close()

            if lan !='en':
                os.remove(file)

    # >>>>>>>>>>>>>>   Loading the Data

    categories = ['Sports', 'Not_Sports']

    data_train = load_files(path_train, categories=categories)
    data_test = load_files(path_test, categories=categories)
    data_predict = load_files(path_predict)


    print '****************** Input *******************'

    print ('\n')

    print ' In Iteration : ',iteration
    iteration = iteration+1
    Training_docs_yes.append(len(data_train.filenames))

    print("Training documents: %d " % len(data_train.filenames))
    print("Testing(Silver standard) documents : %d " % len(data_test.filenames))
    Prediction_docs.append(len(data_predict.filenames))
    print("Documents for prediction %d " % len(data_predict.filenames))
    print("Number of categories : %d " % len(data_train.target_names)), data_train.target_names

    print ('\n')

    ###########################################################################                                        >>>>>>>>>>>>>>>     Vectorization

    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=3, max_df=0.70, lowercase=True, stop_words='english',
                                 decode_error='replace', analyzer='word')

    # Train Data
    X_train = vectorizer.fit_transform(data_train.data)
    y_train = data_train.target

    # Test Data
    X_test = vectorizer.transform(data_test.data)
    y_test = data_test.target

    # Unknown data
    X_predict = vectorizer.transform(data_predict.data)

    #############################################################################################                          >>>>>>>>>>>>     Classification

    print '***********************  MultinominalNB *************************'

    # >>>>>>>>>>  Grid Search

    alpha=[0.01,0.001,0.1,1]
    param_grid = dict(alpha=alpha)

    grid_MNB = GridSearchCV(MultinomialNB(),param_grid,cv=10,scoring='f1')

    #  >>>>>>>>>>  Model Training

    # model_MNB= MultinomialNB(alpha=0.1).fit(X_train,y_train)

    model_MNB = grid_MNB.fit(X_train,y_train)

    # Prediction on Test data

    predict_test1 = model_MNB.predict(X_test)
    accuracy = metrics.accuracy_score(predict_test1, y_test)
    print ('\n')
    print "Accuracy : ", accuracy

    performance_train_MNB = metrics.f1_score(predict_test1, y_test)
    performance_list.append(performance_train_MNB)
    print "F1-score: ", performance_train_MNB

    score_roc_auc_score_MNB = metrics.roc_auc_score(predict_test1, y_test)
    print 'roc_auc_score_MNB :', score_roc_auc_score_MNB

    print 'confusion matrics_MNB :'
    print metrics.confusion_matrix(predict_test1, y_test)
    print ('\n')

    # >>>>>>>>>>>>  Predictions on Unknown Data

    predicted_MNB = model_MNB.predict(X_predict)
    predicted_cnf_MNB = model_MNB.predict_proba(X_predict)

    # >>>>>>>>>>>>   Saving the Model

    if performance_train_MNB > performance_MNB:

        Model_MNB = 'Model_MNB.sav'
        pickle.dump(model_MNB, open(Model_MNB, 'wb'))
        performance_MNB = performance_train_MNB

        # >>>>>>>>>>>>      Saving the Vectorizer according to the Saved model. We use this vectoriser to transform Gold evaluation dataset

        vectorizer_MNB = TfidfVectorizer(sublinear_tf=True, min_df=3, max_df=0.70, lowercase=True, stop_words='english',
                                 decode_error='replace', analyzer='word')

        # vectorisation of the training dada set of the above saved modal
        X_train_MNB = vectorizer_MNB.fit_transform(data_train.data)
        #y_train_MNB = data_train.target

    # ------------------------------>>>>>>>>>>>>>>>>>>>  Logistic Regression

    print '********************** LogisticRegression *******************'

    # >>>>>>>>>>>    Grid Search

    # penalty = ['L1', 'l2']
    # loss = ['squared_hinge']
    # dual = [True, False]
    # tol = [0.0001, 0.001, 0.00001, 0.01]
    C = [0.001, 0.01, 0.1, 1, 10]
    # multi_class = ['ovr', 'crammer_singer']
    # fit_intercept = [True, False]

    max_iter = [1000, 100, 10000]

    # param_grid = dict(penalty=penalty, loss=loss, dual=dual, tol=tol, C=C, multi_class=multi_class, fit_intercept=fit_intercept,max_iter=max_iter)

    param_grid = dict(C=C,max_iter=max_iter)

    grid_LR = GridSearchCV(LogisticRegression(), param_grid, cv=10, scoring='f1')

    # >>>>>>>>>>>>>     Model Training

    # model_LR = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0,class_weight=None, random_state=None).fit(X_train, y_train)

    model_LR = grid_LR.fit(X_train,y_train)

    # >>>>>>>>>>>>>      Prediction on test data

    predict_test2 = model_LR.predict(X_test)
    accuracy_LR = metrics.accuracy_score(predict_test2, y_test)
    print ('\n')
    print "Accuracy_LR: ", accuracy_LR

    performance_train_LR = metrics.f1_score(predict_test2, y_test)
    performance_list.append(performance_train_LR)
    print "F1-score_LR: ", performance_train_LR

    score_roc_auc_scor_LR = metrics.roc_auc_score(predict_test2, y_test)
    print 'roc_auc_score_LR:', score_roc_auc_scor_LR

    print 'confusion matrics_LR:'
    print metrics.confusion_matrix(predict_test2, y_test)
    print ('\n')

    # >>>>>>>>>>>>     Predictions on unknown data

    predicted_LR = model_LR.predict(X_predict)
    predicted_cnf_LR = model_LR.predict_proba(X_predict)

    # >>>>>>>>>>>>>   Saving the Model

    if performance_train_LR > performance_LR:

        Model_LR = 'Model_LR.sav'
        pickle.dump(model_LR, open(Model_LR, 'wb'))

        performance_LR = performance_train_LR

        # >>>>>>>>>>>> Saving the Vectorizer according to the Saved model. We use this vectoriser to transform Gold evaluation dataset

        vectorizer_LR = TfidfVectorizer(sublinear_tf=True, min_df=3, max_df=0.70, lowercase=True, stop_words='english',
                                 decode_error='replace', analyzer='word')

        # vectorisation of the training dada set of the above saved modal
        X_train_LR = vectorizer_LR.fit_transform(data_train.data)
        #y_train_LR = data_train.target


    # ########################################################                                                           >>>>>>>>>>>>>>>>>        SGD Classifier

    print '********************** SGD Classifier *******************'

    #   >>>>>>>>>>  Grid Search

    alpha = [0.0001,0.001,0.01,0.1]
    average = [True,3,5,10]
    class_weight = [None,'balanced']
    epsilon = [0.1,0.01,0.5]
    # eta0 = 0.0,
    # fit_intercept = True,
    # l1_ratio = 0.15,
    # learning_rate = 'optimal',
    loss = ['hinge','modified_huber','perceptron']
    n_iter =[5,1]
    # n_jobs = 1,
    penalty = ['l2']
    # power_t = 0.5,
    # random_state = None,
    # shuffle = True,
    # verbose = 0, warm_start = False).fit(X_train, y_train)

    param_grid= dict(alpha=alpha, average=average, class_weight=class_weight, epsilon=epsilon,loss=loss, n_iter=n_iter,penalty=penalty)
    grid_SGD = GridSearchCV(SGDClassifier(), param_grid, cv=10, scoring='f1')

    # >>>>>>>>>>>  Model Training

    #model_SGD = SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,eta0=0.0, fit_intercept=True, l1_ratio=0.15,learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5, random_state=None, shuffle=True,verbose=0, warm_start=False).fit(X_train, y_train)

    model_SGD= grid_SGD.fit(X_train,y_train)

    #  >>>>>>>>>>>   Saving the Model

    # Model_SGD = 'Model_SGD.sav'
    # pickle.dump(model_SGD, open(Model_SGD, 'wb'))

    # Load_model_SGD = pickle.load(open(Model_SGD, 'rb'))

    # >>>>>>>>>>>    Prediction on test data

    predict_test3 = model_SGD.predict(X_test)
    accuracy_SGD = metrics.accuracy_score(predict_test3, y_test)
    print ('\n')
    print "Accuracy_SGD: ", accuracy_SGD

    performance_train_SGD = metrics.f1_score(predict_test3, y_test)
    performance_list.append(performance_train_SGD)
    print "F1-score_SGD: ", performance_train_SGD

    score_roc_auc_scor_SGD = metrics.roc_auc_score(predict_test3, y_test)
    print 'roc_auc_score_SGD:', score_roc_auc_scor_SGD

    print 'confusion matrics_SGD:'
    print metrics.confusion_matrix(predict_test3, y_test)
    print ('\n')
    print '-' * 100

    # >>>>> predictions on unknown data

    predicted_SGD = model_SGD.predict(X_predict)
    predicted_cnf_SGD = model_SGD.decision_function(X_predict)

    # >>>>>>   Saving the Model

    if performance_train_SGD > performance_SGD:
        Model_SGD = 'Model_SGD.sav'
        pickle.dump(model_SGD, open(Model_SGD, 'wb'))

        performance_SGD = performance_train_SGD

        # >>>>  Saving the Vectorizer according to the Saved model. We use this vectoriser to transform Gold evaluation dataset

        vectorizer_SGD = TfidfVectorizer(sublinear_tf=True, min_df=3, max_df=0.70, lowercase=True, stop_words='english',
                                        decode_error='replace', analyzer='word')

        # vectorisation of the training dada set of the above saved modal
        X_train_SGD = vectorizer_SGD.fit_transform(data_train.data)
        y_train_SGD = data_train.target


    performance.append(performance_list)  # appending the performance_list which has the performances of the above classifiers

    performance_train = max(performance_list)   # taking the maximum performance from the performance_list

    prediction_datapool = []

    yes_datapool = []
    no_datapool = []

    data_pool = []

    # ################################################################          >>>>>>>>>>>>>>>>     Zipping the Results

    for category_MNB,category_LR,category_SGD, doc, conf_MNB,conf_LR,conf_SGD in zip(predicted_MNB,predicted_LR,predicted_SGD, data_predict.filenames, predicted_cnf_MNB,predicted_cnf_LR,predicted_cnf_SGD):

        list_result = []  # temparay list to collect prediction results from all classifiers

        list_result.append(doc)

        list_result.append(data_train.target_names[category_MNB])
        list_result.append(data_train.target_names[category_LR])
        list_result.append(data_train.target_names[category_SGD])

        list_result.append(conf_MNB)
        list_result.append(conf_LR)
        list_result.append(conf_SGD)

        data_pool.append(list_result)

    # ------------------------->>>>>>>>>>>>>>>>>  Making Voting on the prediction results(categories) of classifiers
    for i in data_pool:
        vote_yes = 0
        vote_no = 0
        for j in range(1,4):

            if i[j] == categories[0]:
                vote_yes = vote_yes+1
            else:
                vote_no = vote_no+1
        if vote_yes > vote_no:
            yes_datapool.append(i)
        else:
            no_datapool.append(i)

    #     >>>>>>>>>  Sorting the Datapools based on the confidence scores of the 1st classifier (MNB)

    yes_datapool.sort(key=lambda x: x[4][0], reverse=True)
    no_datapool.sort(key=lambda x: x[4][1], reverse=True)

    # ###############################    >>>>>>>>>>>>>>>>>>>> getting the directory paths of training data

    directory_train_yes = os.path.join(path_train,categories[0])
    directory_train_no = os.path.join(path_train, categories[1])

    # list1 = []
    ele = 0
    # len_y = len(yes_datapool)
    # for i in range(0, len_y):
    for i in range(0,30):
        if ele < 30:
        #if ((yes_datapool[i][2][0]) > .75):
            #for j in range(0, 10):
                shutil.copy(yes_datapool[i][0], directory_train_yes)
                os.remove(yes_datapool[i][0])
                yes_datapool.remove(yes_datapool[i])
                #len_y=len_y-1
                ele = ele+1
        else:
                break

    print "Yes_documents moved from un know to training : ", ele

    ele = 0
    # len_n = len(no_datapool)
    # for i in range(0, len_n):
    for i in range(0,30):
        if ele < 30:
            #if ((no_datapool[i][2][1]) > .60):
                    # for j in range(0, 10):
                    shutil.copy(no_datapool[i][0], directory_train_no)
                    os.remove(no_datapool[i][0])
                    no_datapool.remove( no_datapool[i])
                    # len_n=len_n-1
                    ele = ele + 1

        else:
                break
    print "No_documents moved from unknown to training : " , ele

print "#"*100
print '\n'
print 'Training finished....'
print 'Train docs: '

print Training_docs_yes

print 'Prediction docs : '
print Prediction_docs

print 'Final Performance : '

print performance
print '\n'

print '*'*70
print 'Best performanvces of the classifiers on Silver standard Data :'
print 'Permance_MNB : ',performance_MNB
print 'Performance_LR : ',performance_LR
print 'Performance_SGD : ',performance_SGD
print '*'*70

#  ------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>      Final Evaluation on Gold Standard

print ('\n')
print "*************************  Final Evaluation on Gold Standard   ************************"

path_test_gold = "C:\Users\D065921\Documents\Mahi\Thesis\Crawling\Crawled Data\last versions\inal_versions\practice\Sports\Sports_Gold"
data_test_gold = load_files(path_test_gold, categories=categories, )

X_test_gold_MNB = vectorizer_MNB.transform(data_test_gold.data)
y_test_gold = data_test_gold.target

X_test_gold_LR = vectorizer_LR.transform(data_test_gold.data)
y_test_gold = data_test_gold.target

X_test_gold_SGD = vectorizer_SGD.transform(data_test_gold.data)
y_test_gold = data_test_gold.target

# >>>>>>>>>>>> loading model from disk

Load_model_MNB = pickle.load(open(Model_MNB, 'rb'))
Load_model_LR = pickle.load(open(Model_LR,'rb'))
Load_model_SGD = pickle.load(open(Model_SGD,'rb'))

# >>>>>>>>>>>>>>   Prediction on Gold standard data

predict_test_gold_MNB = Load_model_MNB.predict(X_test_gold_MNB)
predict_test_gold_LR = Load_model_LR.predict(X_test_gold_LR)
predict_test_gold_SGD = Load_model_SGD.predict(X_test_gold_SGD)

accuracy_gold_MNB= metrics.accuracy_score(predict_test_gold_MNB, y_test_gold)
accuracy_gold_LR= metrics.accuracy_score(predict_test_gold_LR, y_test_gold)
accuracy_gold_SGD= metrics.accuracy_score(predict_test_gold_SGD, y_test_gold)

print ('\n')
print "Accuracy_MNB: ", accuracy_gold_MNB
print "Accuracy_LR: ", accuracy_gold_LR
print "Accuracy_SGD: ", accuracy_gold_SGD
print '.'*50

performance_gold_MNB = metrics.f1_score(predict_test_gold_MNB, y_test_gold)
performance_gold_LR = metrics.f1_score(predict_test_gold_LR, y_test_gold)
performance_gold_SGD = metrics.f1_score(predict_test_gold_SGD, y_test_gold)

print "F1-score_MNB: ", performance_gold_MNB
print "F1-score_LR: ", performance_gold_LR
print "F1-score_SGD: ", performance_gold_SGD
print '.'*50

score_roc_auc_scor_MNB = metrics.roc_auc_score(predict_test_gold_MNB, y_test_gold)
score_roc_auc_scor_LR = metrics.roc_auc_score(predict_test_gold_LR, y_test_gold)
score_roc_auc_scor_SGD = metrics.roc_auc_score(predict_test_gold_SGD, y_test_gold)
print 'roc_auc_score_MNB :', score_roc_auc_scor_MNB
print 'roc_auc_score_LR :', score_roc_auc_scor_LR
print 'roc_auc_score_SGD :', score_roc_auc_scor_SGD
print '.'*50

print 'confusion matrics_MNB :'
print metrics.confusion_matrix(predict_test_gold_MNB, y_test_gold)

print 'confusion matrics_LR :'
print metrics.confusion_matrix(predict_test_gold_LR, y_test_gold)

print 'confusion matrics_SGD :'
print metrics.confusion_matrix(predict_test_gold_SGD, y_test_gold)

