
import pandas as pd
from LGBM_refactor import LGBM_processed
from lightgbm import LGBMClassifier
import lightgbm as lgb
import os

submission_format=pd.read_csv('../data/submission_format.csv')
processed_lgbm=LGBM_processed()
processed_lgbm.run_all()
processed_lgbm.encodee_labels()
training_data,validation_data=processed_lgbm.split_test_train()


x_feats = list(training_data.columns)
y_feats = "label"

x_feats.remove(y_feats)



x_factors = ["blast_ft_eng_1", "bacterial_", "growth_strain",
             "selectable_markers", "species_"]

d_train = lgb.Dataset(training_data[x_feats], label=training_data[y_feats],
                      feature_name = x_feats, categorical_feature=x_factors)
d_valid = lgb.Dataset(validation_data[x_feats], label=validation_data[y_feats],
                      feature_name = x_feats, categorical_feature=x_factors)

params = {
    "max_bin": 512,
    "num_classes": training_data[y_feats].nunique(),
        'class_weight':[processed_lgbm.class_weights[z] for z in submission_format.columns[1:]],

    "is_unbalance": True,
    "learning_rate": 0.003,
    "num_leaves": 100,
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "verbose": -1,
    "saved_feature_importance_type": 1,
    "feature_fraction": 0.8,
    #"device_type": "gpu",
}

model = lgb.train(params,
                  d_train,
                  num_boost_round=10,
                  valid_sets=d_valid,
                  early_stopping_rounds=50)

model_path='modles/'
if not os.path.exists(f'{model_path}training_data'):
    os.makedirs(f'{model_path}training_data')

model.save_model(model_path+"temp_model.txt")