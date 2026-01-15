#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy
import itertools
import matplotlib.pyplot as plt


###########################################################################################
# This class is inspired by an example on sklearn's website:          #
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html #
###########################################################################################
class ConfusionMatrixHandler:
    __k_fold = 0

    __confusion_matrix_sum = numpy.array([],[]) #this will be used for the confusion matrix

    def addConfusionMatrix(self, cnf_matrix):
        self.__k_fold += 1
        if self.__k_fold == 1: #if this is the first iteration
            self.__confusion_matrix_sum = cnf_matrix.copy()
        else:
            self.__confusion_matrix_sum = self.__confusion_matrix_sum + cnf_matrix

    def getAvgConfusionMatrix(self):
        return self.__confusion_matrix_sum / self.__k_fold

        
    def plotConfusionMatrixHlp(self, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        cm = self.getAvgConfusionMatrix()
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
            print("Normalized confusion matrix")
            with numpy.printoptions(precision=2, suppress=True):
                print(cm)
        else:
            cm = cm.astype('int')
            print('Confusion matrix, without normalization')
            print(cm)
        

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = numpy.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else '.0f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt,),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        #WORKAROUND FOR BUG IN MATPLOTLIB https://github.com/matplotlib/matplotlib/issues/14751
        plt.ylim(len(classes)-0.5, -0.5)
        plt.subplots_adjust(bottom=0.1)
        plt.subplots_adjust(left=0.1)

    def plotConfusionMatrix(self, level, experiment_name, class_names):
        
        numpy.set_printoptions(precision=2)
        #plot non-normalized confusion matrix
        fig_non_normalized = plt.figure()
        if(level != 1):
            fig_non_normalized.set_figheight(18, forward=True)
            fig_non_normalized.set_figwidth(22, forward=True)
        #fig_non_normalized.canvas.set_window_title(experiment_name + '_confusion_matrix_non_normalized') #old API
        fig_non_normalized.canvas.manager.set_window_title(experiment_name + '_confusion_matrix_non_normalized')
        self.plotConfusionMatrixHlp(classes=class_names,
                              title=experiment_name + ' - Confusion matrix, without normalization')
        #plot normalized confusion matrix
        fig_normalized = plt.figure()
        if(level != 1):
            fig_normalized.set_figheight(18, forward=True)
            fig_normalized.set_figwidth(20, forward=True)
        #fig_normalized.canvas.set_window_title(experiment_name + '_confusion_matrix_normalized') #old API
        fig_normalized.canvas.manager.set_window_title(experiment_name + '_confusion_matrix_normalized')
        self.plotConfusionMatrixHlp(classes=class_names, normalize=True,
                              title=experiment_name + ' - Normalized confusion matrix')

        if(level != 1):
            pass
            #fig_normalized.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.01)
            #fig_non_normalized.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.01)
        else:
            pass
            #fig_normalized.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            #fig_non_normalized.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        fig_normalized.savefig(experiment_name + '_confusion_matrix_normalized.png', dpi=300)
        fig_non_normalized.savefig(experiment_name + '_confusion_matrix_non_normalized.png', dpi=300)

        
