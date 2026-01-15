#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy
from collections import OrderedDict

class ClassificationReportHandler:
    __k_fold = 0

    __class_metrics_dict = OrderedDict() #this will be used for the metrics of specific classes
    __avg_metrics_dict = {} #this will be used for the average metrics

    def addReport(self, rpt):
        self.__k_fold += 1

        classification_report_rows = rpt.split('\n')
        no_classification_report_rows = len(classification_report_rows)
        i = 0
        while i < no_classification_report_rows:
            classification_report_row_values = classification_report_rows[i].split()
            if i == no_classification_report_rows - 2: #this is the row with avg / total  
                if self.__k_fold == 1: #if this is the first iteration
                    self.__avg_metrics_dict['sum_sup'] = int(classification_report_row_values[-1])
                    self.__avg_metrics_dict['sum_iba'] = float(classification_report_row_values[-2])
                    self.__avg_metrics_dict['sum_geo'] = float(classification_report_row_values[-3])
                    self.__avg_metrics_dict['sum_f1'] = float(classification_report_row_values[-4])
                    self.__avg_metrics_dict['sum_spe'] = float(classification_report_row_values[-5])
                    self.__avg_metrics_dict['sum_rec'] = float(classification_report_row_values[-6])
                    self.__avg_metrics_dict['sum_pre'] = float(classification_report_row_values[-7])
                else:
                    self.__avg_metrics_dict['sum_sup'] += int(classification_report_row_values[-1])
                    self.__avg_metrics_dict['sum_iba'] += float(classification_report_row_values[-2])
                    self.__avg_metrics_dict['sum_geo'] += float(classification_report_row_values[-3])
                    self.__avg_metrics_dict['sum_f1'] += float(classification_report_row_values[-4])
                    self.__avg_metrics_dict['sum_spe'] += float(classification_report_row_values[-5])
                    self.__avg_metrics_dict['sum_rec'] += float(classification_report_row_values[-6])
                    self.__avg_metrics_dict['sum_pre'] += float(classification_report_row_values[-7])
            elif classification_report_row_values and self.convertibleToInt(classification_report_row_values[-1]): #the last value (sup) is an int
                if self.__k_fold == 1: #if this is the first iteration
                    tmp_dict = {}
                    tmp_dict['class'] = classification_report_row_values[-8]
                    tmp_dict['sum_sup'] = int(classification_report_row_values[-1])
                    tmp_dict['sum_iba'] = float(classification_report_row_values[-2])
                    tmp_dict['sum_geo'] = float(classification_report_row_values[-3])
                    tmp_dict['sum_f1'] = float(classification_report_row_values[-4])
                    tmp_dict['sum_spe'] = float(classification_report_row_values[-5])
                    tmp_dict['sum_rec'] = float(classification_report_row_values[-6])
                    tmp_dict['sum_pre'] = float(classification_report_row_values[-7])
                    self.__class_metrics_dict[classification_report_row_values[-8]] = tmp_dict
                else:
                    #print('i: ', i)
                    #print('i-2: ', i-2)
                    #print(self.__class_metrics_dict[i-2]['sum_sup'])
                    #print(int(classification_report_row_values[-1]))
                    self.__class_metrics_dict[classification_report_row_values[-8]]['sum_sup'] += int(classification_report_row_values[-1])
                    self.__class_metrics_dict[classification_report_row_values[-8]]['sum_iba'] += float(classification_report_row_values[-2])
                    self.__class_metrics_dict[classification_report_row_values[-8]]['sum_geo'] += float(classification_report_row_values[-3])
                    self.__class_metrics_dict[classification_report_row_values[-8]]['sum_f1'] += float(classification_report_row_values[-4])
                    self.__class_metrics_dict[classification_report_row_values[-8]]['sum_spe'] += float(classification_report_row_values[-5])
                    self.__class_metrics_dict[classification_report_row_values[-8]]['sum_rec'] += float(classification_report_row_values[-6])
                    self.__class_metrics_dict[classification_report_row_values[-8]]['sum_pre'] += float(classification_report_row_values[-7])
            i += 1

    def addAccuracy(self, acc):
        if self.__k_fold == 1: #if this is the first iteration
            self.__avg_metrics_dict['sum_acc'] = acc
        else:
            self.__avg_metrics_dict['sum_acc'] += acc

    def addF1(self, f1_micro, f1_macro, f1_weighted):
        if self.__k_fold == 1: #if this is the first iteration
            self.__avg_metrics_dict['f1_micro'] = f1_micro
            self.__avg_metrics_dict['f1_macro'] = f1_macro
            self.__avg_metrics_dict['f1_weighted'] = f1_weighted
        else:
            self.__avg_metrics_dict['f1_micro'] += f1_micro
            self.__avg_metrics_dict['f1_macro'] += f1_macro
            self.__avg_metrics_dict['f1_weighted'] += f1_weighted

    def getReportString(self):
        avg_report_string = "\tpre\trec\tspe\tf1\tgeo\tiba\tsup"
        avg_report_string += "\n"
        for x in self.__class_metrics_dict.values():
            print("X: ", x)
            print("X class: ", x['class'])
            avg_report_string += str(x['class']) + "\t"
            avg_report_string += str(round((x['sum_pre']/float(self.__k_fold)),2)) +  "\t"
            avg_report_string += str(round((x['sum_rec']/float(self.__k_fold)),2)) +  "\t"
            avg_report_string += str(round((x['sum_spe']/float(self.__k_fold)),2)) +  "\t"
            avg_report_string += str(round((x['sum_f1']/float(self.__k_fold)),2)) +  "\t"
            avg_report_string += str(round((x['sum_geo']/float(self.__k_fold)),2)) +  "\t"
            avg_report_string += str(round((x['sum_iba']/float(self.__k_fold)),2)) +  "\t"
            avg_report_string += str(round((x['sum_sup']/float(self.__k_fold)),2)) + "\n"
        avg_report_string += "\n"
        avg_report_string += "avg / total\t"
        avg_report_string += str(round((self.__avg_metrics_dict['sum_pre']/float(self.__k_fold)),2)) +  "\t"
        avg_report_string += str(round((self.__avg_metrics_dict['sum_rec']/float(self.__k_fold)),2)) +  "\t"
        avg_report_string += str(round((self.__avg_metrics_dict['sum_spe']/float(self.__k_fold)),2)) +  "\t"
        avg_report_string += str(round((self.__avg_metrics_dict['sum_f1']/float(self.__k_fold)),2)) +  "\t"
        avg_report_string += str(round((self.__avg_metrics_dict['sum_geo']/float(self.__k_fold)),2)) +  "\t"
        avg_report_string += str(round((self.__avg_metrics_dict['sum_iba']/float(self.__k_fold)),2)) +  "\t"
        avg_report_string += str(round((self.__avg_metrics_dict['sum_sup']/float(self.__k_fold)),2)) + "\n"
        avg_report_string += "\n"
        avg_report_string += 'accuracy: ' + str(round((self.__avg_metrics_dict['sum_acc']/float(self.__k_fold)),2)) + "\n"
        avg_report_string += 'f1_macro: ' + str(round((self.__avg_metrics_dict['f1_macro']/float(self.__k_fold)),2)) + "\n"
        avg_report_string += 'f1_micro: ' + str(round((self.__avg_metrics_dict['f1_micro']/float(self.__k_fold)),2)) + "\n"
        avg_report_string += 'f1_weighted: ' + str(round((self.__avg_metrics_dict['f1_weighted']/float(self.__k_fold)),2)) + "\n"
        return avg_report_string

    def convertibleToInt(self,value):
        try:
            int(value)
            return True
        except:
            return False

        
