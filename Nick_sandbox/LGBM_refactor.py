

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight,shuffle
from collections import Counter
from sklearn.model_selection import train_test_split


def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cls: float(majority/count) for cls, count in counter.items()}

class LGBM_processed():
    def __init__(self,input_values=os.path.join("../data/","train_values_blasted.csv"),
                 input_lables=os.path.join("../data/","train_labels.csv"),
                 test_values=os.path.join("../data/","test_values_blasted.csv")):
        self.train_values= pd.read_csv(input_values)
        self.train_labels=pd.read_csv(input_lables)
        self.test_values=pd.read_csv(test_values)

    def add_new_features(self):
        self.train_values['sequence_len'] = self.train_values.sequence.str.len()
        self.test_values['sequence_len'] = self.test_values.sequence.str.len()
        self.train_values.blast_ft_eng_1[self.train_values.blast_ft_eng_1.isna()] = 'NA'
        self.test_values.blast_ft_eng_1[self.test_values.blast_ft_eng_1.isna()] = 'NA'

    def merge_labels_with_values(self):
        y = self.train_labels.loc[:, ["sequence_id"]].copy()
        y["label"] = self.train_labels.drop(["sequence_id"], axis=1).idxmax(axis=1)

        # get class weights
        self.class_weights=get_class_weights(y["label"])

        self.train_labels_values = self.train_values.merge(y, how="left", on="sequence_id")


    def refactor_features_to_single_columns(self):
        col2keep = ["pident", "length", "sstart", "send", "mismatch", "gapopen",
                    "evalue", "blast_ft_eng_1", "blast_aligned_fraction", "sequence_len", "label"]

        refactored_train = self.train_labels_values.reindex(columns=col2keep)
        refactored_test = self.test_values.reindex(columns=col2keep)
        refactored_dfs=[refactored_train,refactored_test]

        growth_temp_recode = {
            "growth_temp_37": 37,
            "growth_temp_30": 30,
            "growth_temp_other": -999,
        }
        copy_number_recode = {
            "copy_number_high_copy": 1,
            "copy_number_low_copy": 0,
            "copy_number_unknown": -999
        }
        for l,dataset in enumerate([self.train_labels_values,self.test_values]):
            for column_code in ['bacterial_','growth_strain','selectable_markers','species_']:
                idx = dataset.columns.str.contains(column_code)
                refactored_dfs[l][column_code] = dataset.loc[:, idx].apply(lambda x:''.join([str(y) for y in x.values]),axis=1)

            # now growth temp
            idx = dataset.columns.str.contains("growth_temp")
            refactored_dfs[l]["growth_temp"] = dataset.loc[:, idx].idxmax(axis=1)
            refactored_dfs[l].growth_temp = refactored_dfs[l].growth_temp.map(growth_temp_recode)
            # now copy number
            idx = dataset.columns.str.contains("copy_number")
            refactored_dfs[l]["copy_number"] = dataset.loc[:, idx].idxmax(axis=1)
            refactored_dfs[l].copy_number = refactored_dfs[l].copy_number.map(copy_number_recode)

        self.train_labels_values_refactored=refactored_dfs[0]
        self.test_values_refactored=refactored_dfs[1]


    def encodee_labels(self):
        features_to_encode= ["blast_ft_eng_1", 'bacterial_','growth_strain','selectable_markers','species_']
        lb_enc_dict = {}

        # fix
        self.test_values_refactored.blast_ft_eng_1 = self.test_values_refactored.blast_ft_eng_1.apply(lambda x: "NA" if 'tpe' in x else x)

        for feature in features_to_encode:
            lb_enc_dict[feature] = LabelEncoder()

            temp_values=pd.concat([self.train_labels_values_refactored[feature],self.test_values_refactored[feature]],ignore_index=True)
            lb_enc_dict[feature].fit(temp_values)
            self.train_labels_values_refactored[feature] = lb_enc_dict[feature].transform(self.train_labels_values_refactored[feature])
            self.train_labels_values_refactored[feature] = self.train_labels_values_refactored[feature].astype('category')

            self.test_values_refactored[feature] = lb_enc_dict[feature].transform(self.test_values_refactored[feature])
            self.test_values_refactored[feature] = self.test_values_refactored[feature].astype('category')


        ## encoding y
        temp_values=self.train_labels_values_refactored['label']

        lb_enc_dict['label'] = LabelEncoder()
        lb_enc_dict['label'].fit(temp_values)
        self.train_labels_values_refactored['label'] = lb_enc_dict['label'].transform(self.train_labels_values_refactored['label'])



        self.lb_enc_dict=lb_enc_dict




    def split_test_train(self,frac=0.0001):
        train_data, val_data= train_test_split(self.train_labels_values_refactored, test_size=frac,
                                                                                      random_state=88)
        return train_data,val_data

    def run_all(self):
        self.add_new_features()
        self.merge_labels_with_values()
        self.refactor_features_to_single_columns()



if __name__ == '__main__':
    processed_lgbm = LGBM_processed()
    processed_lgbm.run_all()
    processed_lgbm.encodee_labels()