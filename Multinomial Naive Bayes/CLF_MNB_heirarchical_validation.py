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

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity


from sklearn import svm
import numpy as np
import pandas
import pickle
import os
import sys
import nltk
import re
import shutil
from langdetect import detect



top_dir=os.listdir(os.getcwd())
start_dir=os.getcwd()


for dirx in top_dir:

    if dirx.endswith('.py'):
        x=0

    else:

        Path_main = os.path.join(start_dir,dirx)

        os.chdir(Path_main)

        filename  = open(dirx+"_outputfile_Heirarchy_2.txt",'w')
        sys.stdout = filename

        #          >>>>>>>>>>>>  Variable diclaration

        performance_init = 0
        performance_train = 0.0001

        performance_MNB = 0
        performance_LR = 0
        performance_SGD = 0


        Training_docs_yes = []
        Training_docs_no = []
        Prediction_docs = []

        performances_loop=[]


        X_train_MNB = 0
        X_train_LR = 0
        X_train_SGD = 0

        len_y = 11
        len_n =11


        iteration = 1
        loop = 0

        lang_check = 0

        # >>>>>>>>>>>>>>>>>>>>>  Getting Data Paths

        #Path_main=os.getcwd()

        path_test_gold = 0
        path_train = 0
        path_test = 0
        path_predict = 0
        categories = []

        sub_dirs=os.listdir(Path_main)

        #for root, dirs, files in os.walk(Path_main):



        for dir in sub_dirs:

              #if dir!='New folder' :
                if dir=='Gold':
                    path_test_gold =str(os.path.join(Path_main,'Gold'))
                    print "Gold_Path : ",path_test_gold
                #if dir=='Silver':
                #    path_test=str(os.path.join(Path_main,'Silver'))
                 #   print 'Silver_Path : ',path_test
                if dir=='Training':
                    path_train=str(os.path.join(Path_main,'Training'))
                    print 'Training_Path : ',path_train
                #if dir=='Unknown':
                 #   path_predict=os.path.join(Path_main,'Unknown')
                    #path_predict= str(os.path.normpath(Unknown_x + os.sep + os.pardir))
                  #  print 'Unknown_Path : ',path_predict

        for root,dirs,files in os.walk(path_train):
            for dir in dirs:
                categories.append(dir)
                #print dir
        print 'Categories : ',categories
        print 'yes_data : ',categories[0]
        print 'no_data : ',categories[1]


        # >>>>>>>>>>>>>>>>     Data preprocessing with NLTK

        stemmer = SnowballStemmer("english")
        wordnet_lemmatizer = WordNetLemmatizer()

        def tokenize_and_stem(text):
            # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
            tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
            filtered_tokens = []
            # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
            for token in tokens:
                if re.search(u'[a-zA-Z]', token):

                    filtered_tokens.append(token)
            #stems = [stemmer.stem(t) for t in filtered_tokens]

            lemmats =[wordnet_lemmatizer.lemmatize(t,pos='v') for t in filtered_tokens]
            stems = [stemmer.stem(k) for k in lemmats]
            #synonyms=[wordnet.synset(k) for k in lemmats]
            #print type(lemmats)
            #return stems
            return stems

        # >>>>>>>>>>>>>>   Loading the Data

        data_train = load_files(path_train, categories=categories)

        print '****************** Input *******************'

        print ('\n')

        Training_docs_yes.append(len(data_train.filenames))

        print("Training documents: %d " % len(data_train.filenames))

        print ('\n')

        # ##############        >>>>>>>>>>>>>>>     Vectorization
        vectorizer = TfidfVectorizer(sublinear_tf=True, lowercase=True, stop_words='english', min_df=3,max_df=.70,
                                         decode_error='ignore', strip_accents='ascii', tokenizer=tokenize_and_stem)

        # Train Data
        X_train = vectorizer.fit_transform(data_train.data)
        y_train = data_train.target


        # ################       >>>>>>>>>>>>     Classification

        print '********************** MNB Classifier *******************'

        #   >>>>>>>>>>  Grid Search

        alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        # cross_valid=5,10,15

        param_grid = dict(alpha=alpha)
        # param_grid_cv=dict(cv=cross_valid)

        grid_MNB = GridSearchCV(MultinomialNB(), param_grid, cv=10, scoring='f1')
        # >>>>>>>>>>>  Model Training

        # >>>>>>>>>>>>>     Model Training

        # model_MNB = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0,class_weight=None, random_state=None).fit(X_train, y_train)

        model_MNB = grid_MNB.fit(X_train, y_train)

        # >>>>>>>>>> Best Parameters on development set
        print '**********   Best parameters on Development set ********** '

        print model_MNB.best_params_
        print '\n'
        print '**********  Best Estimater which gave hei score  *********'

        print model_MNB.best_estimator_

        # >>>>>>   Saving the Model

        os.chdir(Path_main)

        Model_MNB = categories[0]+'_Model_Heirarchy_MNB_2.sav'
        pickle.dump(model_MNB, open(Model_MNB, 'wb'))

        Vector_MNB=categories[0]+'_Vector_Heirarchy_MNB_2.sav'
        os.chdir(Path_main)
        pickle.dump(vectorizer,open(Vector_MNB,'wb'))

        #print 'Saved training vector shape ', X_train_SGD.shape
        #y_train_SGD = data_train.target

        #  ------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>      Final Evaluation on Gold Standard

        print ('\n')
        print "*************************  Final Evaluation on Gold Standard   ************************"

        data_test_gold = load_files(path_test_gold, categories=categories, )

        Load_vector_MNB = pickle.load(open((categories[0] + '_Vector_Heirarchy_MNB_2.sav'), 'rb'))

        X_test_gold_MNB = Load_vector_MNB.transform(data_test_gold.data)

        print 'gold data vector : ', X_test_gold_MNB.shape
        y_test_gold = data_test_gold.target

        # >>>>>>>>>>>> loading model from disk

        Load_model_MNB = pickle.load(open((categories[0] + '_Model_Heirarchy_MNB_2.sav'), 'rb'))

        # >>>>>>>>>>>>>>   Prediction on Gold standard data


        predict_test_gold_MNB = Load_model_MNB.predict(X_test_gold_MNB)

        accuracy_gold_MNB = metrics.accuracy_score(predict_test_gold_MNB, y_test_gold)

        print ('\n')

        print "Accuracy_MNB: ", accuracy_gold_MNB
        print '.' * 50

        performance_gold_MNB = metrics.f1_score(predict_test_gold_MNB, y_test_gold)

        print "F1-score_MNB: ", performance_gold_MNB
        print '.' * 50

        # score_roc_auc_scor_MNB = metrics.roc_auc_score(predict_test_gold_MNB, y_test_gold)

        # print 'roc_auc_score_MNB :', score_roc_auc_scor_MNB
        # print '.'*50


        print 'confusion matrics_MNB :'
        print metrics.confusion_matrix(predict_test_gold_MNB, y_test_gold)

