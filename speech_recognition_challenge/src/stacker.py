import config
from dataset.classifier_loader import CLFDataset
from classifier.classifier import Classifier
import utils.utils as utils
import json
import numpy as np

class Stacker(object):
    def __init__(self, nn_folders, clf_name):
        self.nn_folders = nn_folders
        self.RS = 15
        self.clf_name = clf_name
        self.clf_ds = CLFDataset(nn_folders = self.nn_folders)

        self.output_folder = utils.create_unique_path(config.OUTPUT_FOLDER + 'stacker_' + self.clf_name + '/', create = True)
        utils.zip_src_folder(self.output_folder, exclude_folders = [])
        self.train, self.test, self.splits = self.clf_ds()

        self.train.to_csv(self.output_folder + 'training_dataset.csv', index = False)
        self.test.to_csv(self.output_folder + 'testing_dataset.csv', index = False)


        print('Saved to %s'%self.output_folder)
    def __call__(self):
        # test
        # self.train = self.train.head(200)
        # self.test = self.test.head(100)

        self.clf = Classifier(output_folder = self.output_folder, RS = 15, train = self.train, test = self.test, 
                                                        fold_splits = self.splits, clf_name = self.clf_name, mapping_dict = config.mapping_dict)
        self.clf()
        print('Saved to %s'%self.output_folder)

if __name__ == '__main__':
    nn_folders = ["nn_0", "nn_1", "nn_3", "nn_4", "nn_5", "nn_6", "nn_7", 'nn_8', 'nn_9', "nn_11",
                "nn_12", "nn_13", "nn_14", "nn_15", "nn_16", "nn_17", "nn_18", "nn_19", "nn_20", "nn_22", ]
    # nn_folders = ["nn_0"]
    # nn_folders = ['custom_predictions_%d'%i for i in [0,1] + np.arange(3, 10).tolist() + [11] ]
    print(nn_folders)
    s = Stacker(nn_folders = nn_folders,
            clf_name = 'xgb')
    s()