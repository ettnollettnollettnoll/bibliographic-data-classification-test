#!/usr/bin/python3
# -*- coding: utf-8 -*-

#from helper_functions import slugify
from cl_rpt import ClassificationReportHandler as crpth
from cm import ConfusionMatrixHandler as cmth

import argparse
import numpy
import csv
import logging
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from imblearn.over_sampling import RandomOverSampler 
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced
from collections import Counter
from joblib import dump, load

#constants and variables
GLOBAL_MAX_DF = 0.7 #do not change for final experiments
GLOBAL_K_FOLD_N_SPLITS = 3 #do not change for final experiments
GLOBAL_N_GRAMS = 2 #do not change for final experiments

global_data_dir_in = 'english_titles_001'
global_csv_file_name = 'english_titles_001.csv'
global_log_file_name = 'TEST_001.log'
global_experiment_name = 'Test_E01'
global_level = 0
global_use_only_title = False
global_model = 'MultinomialNB'
#global_model = 'LinearSVC'
global_save_model = False
global_model_name = 'my_model'

#collect command line arguments
parser = argparse.ArgumentParser(description='Train and evaluate classiniers')
parser.add_argument('-d','--directory',type=str,dest='directory',default=False,help='Directory of input file')
parser.add_argument('-f','--file',type=str,dest='file',default=False,help='Input file name')
parser.add_argument('-l','--log',type=str,dest='log',default=False,help='Logfile name')
parser.add_argument('-e','--experiment',type=str,dest='experiment',default=False,help='Experiment name')
parser.add_argument('-v','--level',type=int,dest='level',default=False,help='Classification level')
parser.add_argument('-t','--feature',type=str,dest='feature',default=False,help='Base for features (\'title\' for titles)')
parser.add_argument('-m','--model',type=str,dest='model',default=False,help='Machine learning model')
parser.add_argument('-s','--save',type=str,dest='save',default=False,help='Should the model be saved (\'Y/N\')')
parser.add_argument('-n','--name',type=str,dest='name',default=False,help='Name used when saving model')
args = parser.parse_args()

if(args.directory):
    global_data_dir_in = args.directory
if(args.file):
    global_csv_file_name = args.file
if(args.log):
    global_log_file_name = args.log
if(args.experiment):
    global_experiment_name = args.experiment
if(args.level):
    global_level = args.level
if(args.feature == 'title'):
    global_use_only_title = True
if(args.model):
    global_model = args.model
if(args.save and (args.save.lower() == 'y' or args.save.lower() == 'yes')):
    global_save_model = True
if(args.name):
    #global_model_name = slugify(args.name)
    global_model_name = args.name

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',filename=global_log_file_name,level=logging.INFO)
logging.info('START')

logging.info('global_experiment_name = ' + global_experiment_name)
logging.info('global_data_dir_in = ' + global_data_dir_in)
logging.info('global_csv_file_name = ' + global_csv_file_name)
logging.info('global_log_file_name = ' + global_log_file_name)
logging.info('global_level = ' + str(global_level))
logging.info('global_use_only_title = ' + str(global_use_only_title))
logging.info('GLOBAL_MAX_DF = ' + str(GLOBAL_MAX_DF))
logging.info('GLOBAL_K_FOLD_N_SPLITS = ' + str(GLOBAL_K_FOLD_N_SPLITS))
logging.info('GLOBAL_N_GRAMS = ' + str(GLOBAL_N_GRAMS))
logging.info('global_model = ' + global_model)
logging.info('global_save_model = ' + str(global_save_model))
logging.info('global_model_name = ' + str(global_model_name))

row_no = 0 #row no in the csv file

#read the data into a pandas dataframe
data = DataFrame({'text': [], 'class': []})
with open(global_data_dir_in + '/' + global_csv_file_name, newline='', encoding='utf-8') as csvfile:
    csvreader = csv.DictReader(csvfile, ['identifier','title','full_ddc','ddc','language','subject_headings'])
    rows = []
    index = []
    for row in csvreader:
        row_no += 1 #the first row is a header row (this can be used if csv.DictReader is called with only one argument)
        if(row_no > 1):

            if(global_level > 0):
                ddc = row['ddc'][:global_level]
            else:
                ddc = row['ddc']

            if(global_use_only_title):
                text = row['title']
            else:
                text = row['title'] + ' ' + row['subject_headings']          
            
            rows.append({'text': text, 'class': ddc})
            
            index.append(row['identifier'])
            #index.append(row_no) #if you want to use the row_no for index
            
        if(row_no % 5000 == 0):
            print('Reading row:',row_no)
            logging.info('Reading row: ' + str(row_no))
data = DataFrame(rows, index=index)

#randomize the data
data = data.reindex(numpy.random.permutation(data.index))

#pipelineing the steps
if(global_model == 'MultinomialNB'):
    pipeline = make_pipeline_imb(CountVectorizer(max_df=GLOBAL_MAX_DF, ngram_range=(1, GLOBAL_N_GRAMS)),
                                 TfidfTransformer(),
                                 RandomOverSampler(),
                                 MultinomialNB())
elif(global_model == 'LinearSVC'):
        pipeline = make_pipeline_imb(CountVectorizer(max_df=GLOBAL_MAX_DF, ngram_range=(1, GLOBAL_N_GRAMS)),
                                 TfidfTransformer(),
                                 RandomOverSampler(),
                                 LinearSVC())
elif(global_model == 'SGDClassifier'):
            pipeline = make_pipeline_imb(CountVectorizer(max_df=GLOBAL_MAX_DF, ngram_range=(1, GLOBAL_N_GRAMS)),
                                 TfidfTransformer(),
                                 RandomOverSampler(),
                                 SGDClassifier())



logging.info('Original dataset shape {}'.format(Counter(data['class'])))

#store the class names
class_names = list(Counter(data['class']).keys())
class_names.sort()

#instantiate helper classes
cmt = cmth() #instantiate our confusion matrix handler class 
crpt = crpth() #instantiate our classification report handler class 

k_fold_i = 0 #k_fold counter
k_fold = StratifiedKFold(n_splits=GLOBAL_K_FOLD_N_SPLITS) #split the data

#for each fold
for train_indices, test_indices in k_fold.split(data['text'].values, data['class'].values):

    k_fold_i += 1
    
    train_text = data.iloc[train_indices]['text'].values #the basis for training features 
    train_y = data.iloc[train_indices]['class'].values #the basis for training labels

    test_text = data.iloc[test_indices]['text'].values #the basis for test features 
    test_y = data.iloc[test_indices]['class'].values #the basis for test labels

    pipeline.fit(train_text, train_y) #build the model
    predictions = pipeline.predict(test_text) #make predictions

    #create a classification report with imblearn
    classification_report = classification_report_imbalanced(test_y, predictions)
    #print the report
    print(classification_report)
    #log the report
    logging.info('K-fold: ' + str(k_fold_i) + "\n" + classification_report)
    #save the report so averages can be calculated later
    crpt.addReport(classification_report)

    #accuracy
    accuracy = accuracy_score(test_y, predictions)
    #print the accuracy
    print("accuracy_score: " + str(accuracy))
    #log the accuracy
    logging.info("accuracy_score: " + str(accuracy))
    #save the accuracy so averages can be calculated later
    crpt.addAccuracy(accuracy)

    #f1
    f1_micro = f1_score(test_y, predictions, average='micro')
    f1_macro = f1_score(test_y, predictions, average='macro')
    f1_weighted = f1_score(test_y, predictions, average='weighted')
    #print the f1 scores
    print("f1_micro: " + str(f1_micro))
    print("f1_macro: " + str(f1_macro))
    print("f1_weighted: " + str(f1_weighted) + "\n")
    #log the f1 scores
    logging.info("f1_micro: " + str(f1_micro))
    logging.info("f1_macro: " + str(f1_macro))
    logging.info("f1_weighted: " + str(f1_weighted) + "\n")
    #save the f1 scores so averages can be calculated later
    crpt.addF1(f1_micro,f1_macro,f1_weighted)

    #confusion matrix
    cnf_matrix = confusion_matrix(test_y, predictions)
    #print the confusion matrix
    print("confusion matrix:\n" + str(cnf_matrix) + "\n")
    #log the confusion matrix
    logging.info("confusion matrix:\n" + str(cnf_matrix) + "\n")
    #save the confusion matrix so averages can be calculated later
    cmt.addConfusionMatrix(cnf_matrix)

#get the average results
avg_report_string = crpt.getReportString()
#print the average results
print('global_experiment_name = ' + global_experiment_name)
print(avg_report_string)
#log the average results
logging.info('global_experiment_name = ' + global_experiment_name)
logging.info('Average: ' + "\n" + avg_report_string)

#save the results in a csv file
clf_rpt_file = open(global_experiment_name + '_classification_report.csv', 'w', -1, "utf-8")
clf_rpt_file.write(avg_report_string)
clf_rpt_file.close()

#get the average confusion matrix
avg_confusion_matrix = cmt.getAvgConfusionMatrix()
#log the average confusion matrix
logging.info("confusion matrix:\n" + str(avg_confusion_matrix) + "\n")
#save average confusion matrix as two png files
cmt.plotConfusionMatrix(global_level,global_experiment_name,class_names)

#saving the model
if(global_save_model):
    #https://scikit-learn.org/stable/modules/model_persistence.html
    print('\n'  + 'Training with the full set.')
    logging.info('\n'  + 'Training with the full set.')
    pipeline.fit(data['text'].values, data['class'].values) #build the model
    print('Saving the model to: ' + global_model_name + '.joblib')
    logging.info('Saving the model to: ' + global_model_name + '.joblib')
    dump(pipeline, global_model_name + '.joblib')

logging.info("END\n\n\n")

