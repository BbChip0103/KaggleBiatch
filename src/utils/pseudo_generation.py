import os, sys 
import pandas as pd
import numpy as np
import glob
sys.path.insert(0,'..')
import config
from sklearn.utils import shuffle 
from string import Template
import joblib

class PseudoGenerator(object):
    def __init__(self, prediction_folder):
        self.prediction_folder = prediction_folder
        self.N_limit_per_label = 1000
        self.percentage = 0.2
        self.out_folder = config.OUTPUT_FOLDER + '%s/'%(nn_folder)
    
    def read(self):

        if 'nn' in self.prediction_folder:
            self.flag = 'nn'
            name = Template("pred_${flag}.csv")
        elif 'stacker' in self.prediction_folder:
            self.flag = 'stacker'
            name = Template("sub_${flag}.csv")
        
        self.test = pd.read_csv(self.out_folder  + name.substitute(flag = 'test'))

        if self.flag == 'nn':
            le = joblib.load(self.out_folder  + 'label_encoder.dump')
            self.test.rename(columns = {str(i) : le.inverse_transform(i) for i in np.arange(31)}, inplace = True)
            self.test['target'] = self.test[config.mapping_dict.keys()].idxmax(axis = 1)

        self.test['confidence'] = self.test[config.mapping_dict.keys()].max(axis = 1)
        self.test['person_id'] = self.test['id']
        self.test['path'] = config.DATA_FOLDER + 'test/audio/' + self.test['id']  + '.wav'

    def __call__(self):   
        self.read()
        self.final_df = []   
        # for c in [x for x in config.mapping_dict.keys() if x not in  config.allowed_train_labels] + ['silence']:
        for c in config.mapping_dict.keys():
            df =  self.test[self.test['target'] == c].copy()
            N = np.clip(int(self.percentage* config.categories[c]), 0, self.N_limit_per_label)
            df = df.sort_values('confidence', ascending = False).head(N)
            print('Label: %s; amount: %d, confidence in [%f, %f]'%(c, N, df['confidence'].min(), df['confidence'].max()))
            df = shuffle(df)
            df.reset_index(drop = True, inplace = True)
            
            df = df[['id', 'path', 'person_id', 'target']]
            self.final_df.append(df)

        self.final_df = pd.concat(self.final_df)
        self.final_df.reset_index(drop = True, inplace = True)
        print(self.final_df.info())
        print(self.final_df['target'].value_counts())

        #save
        save_name = config.OUTPUT_FOLDER + self.prediction_folder + '/pseudo_' + self.prediction_folder + '_percent_%d.csv'%(self.percentage * 100)
        print('Saving to %s'%save_name)
        self.final_df.to_csv(save_name, index = False)

if __name__ == '__main__':
           
    nn_folder = 'stacker_xgb_14'
    PseudoGenerator(prediction_folder = nn_folder)()


