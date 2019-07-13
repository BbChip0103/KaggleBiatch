# What is it?
KaggleBiatch - The complete pipeline for Kaggle Img competitons (multiclass, multilable, segmentation). 
* Train different DL models (PNasNet, Resnet etc) with cross validation, different ugs,  samplers and lr schedulers. 
* Save the top5 checkpoints fro latter inference
* Inference models
* Stack them using xgboost, lightgbm and scikit-learn
* Generate submission

**It's in pre-alpha stage** because I abandoned Kaggle [https://www.kaggle.com/heyt0ny](https://www.kaggle.com/heyt0ny) and only tried this code for Doodle Competiton. There are may be a lot of bugs.


# Where is src code for Kaggle Speech Recognition Challenge?
I refactored code from all Kaggle Competions, where I participated to make this repo. The code for Speech Recognition Challenge is in `speech_recognition_challenge` folder. It's really bad code, written on python2. So if I were you, I grabed the idea from `speech_recognition_challenge` but cloned the clean code from `src`.

# Requirements
0) `python3`
1) `pip3 install -r requirementes.txt`

# How to use

1) Create in home directory file `.kaggle/path.json`:

```json
{
  "output_path": "/media/large_HDD/output",
  "data_path": "/media/large_HDD/data"
}
```

2) Edit `src/competition.json` to scpecific comptetion

```json
{
  "competition_data_folder": "competition_name",
  "competition_id_col": "Id",
  "competition_img": {
    "type": {
      "test": "png",
      "train": "png"
    }
  },
  "competition_name": "competition_name",
  "competition_num_channels": 4,
  "competition_predict_target_col": "Predicted",
  "competition_target_col": "Target",
  "competition_type": "binary",
  "dataset_split": {
    "RS": 2018,
    "n_folds": 5,
    "shuffle": true,
    "stratified": true
  },
  "debug_rows": 1000,
  "img_denominator": 255,
  "num_classes": 2
}
```

3) Train and predict `cd src && python3 main.py`

4) Generate submission `cd srd && python3 sub.py`
