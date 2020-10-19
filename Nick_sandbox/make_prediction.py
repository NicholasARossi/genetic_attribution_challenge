from LGBM_refactor import LGBM_processed
import lightgbm as lgb
import pandas as pd
import os

model = lgb.Booster(model_file='/Users/nicholas.rossi/Documents/Scratch/modles/temp_model.txt')
submission_format=pd.read_csv('../data/submission_format.csv')
processed_lgbm=LGBM_processed()
processed_lgbm.run_all()
processed_lgbm.encodee_labels()

y_lbe=processed_lgbm.lb_enc_dict['label']
print('making prediction ... \n')
test_preds = model.predict(processed_lgbm.test_values_refactored.drop(labels='label', axis=1))
test_preds = pd.DataFrame(test_preds, columns=y_lbe.classes_)

test_preds.insert(0, 'sequence_id', processed_lgbm.test_values.sequence_id)

model_path='predictions/'
if not os.path.exists(f'{model_path}'):
    os.makedirs(f'{model_path}')


test_preds.to_csv(f"{model_path}temp.csv", index=False)