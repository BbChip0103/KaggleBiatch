import sys, os
import pandas as pd
import joblib

sys.path.insert(0,'..')
import config

class SubGenerator(object):
    def __init__(self, read_folder):
        self.short_folder_name = read_folder
        self.read_folder  = config.OUTPUT_FOLDER +  read_folder + '/'
    
    def read(self):
        self.df = {}
        for t in ['train', 'test']:
            self.df[t] = pd.read_csv(self.read_folder + 'pred_%s.csv'%t)

    def process(self):
        for t in ['train', 'test']:
            self.df[t][config.sub_target_col] = self.df[t].drop('id', 1).idxmax(axis = 1).astype(int)
            # inverse label
            if 'neural_clf' not in self.short_folder_name:
                self.le = joblib.load(self.read_folder + 'label_encoder.dump')
                self.df[t][config.sub_target_col] = self.le.inverse_transform(self.df[t][config.sub_target_col])
            else:
                self.df[t][config.sub_target_col] = self.df[t][config.sub_target_col].map({v:k for k, v in config.mapping_dict.iteritems()})

            self.df[t][config.sub_target_col] = self.df[t][config.sub_target_col].apply(lambda x: x if x in config.allowed_train_labels else 'unknown')
            print(self.df[t][config.sub_target_col].value_counts())

            sub = self.df[t][['id', config.sub_target_col]]
            sub.to_csv(self.read_folder + 'sub_%s.csv'%t, index = False)
            if t == 'test':
                sub['id'] = sub['id'] + '.wav'
                sub.rename(columns = {'id' : 'fname'}, inplace = True)
                sub.to_csv(self.read_folder + 'submission_%s.csv'%self.short_folder_name, index = False)



    def __call__(self):
        self.read()
        self.process()

if __name__ == '__main__':
    sg = SubGenerator('neural_clf')
    sg()