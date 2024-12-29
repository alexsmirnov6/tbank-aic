This folder contains jupyter notebooks used for data processing, pretraining, training, and submit creation.
 - pretrain_train_clean_code.ipynb - notebook with pre-training and training
 - final_data_process.ipynb - notebook with audio preprocessing using whisper (transcription + word splitting)
 - final_submission_creating.ipynb is a notebook with an inference that creates submissions

You may also download weights of our final classification model after all training stages [from here](https://drive.google.com/file/d/10o4PeDrDs3nYXGCnT9FoyesFEh-nxZLw/view?usp=sharing)

[This archive](https://drive.google.com/drive/folders/1mVx-uGAhWtvDMnIhdcZW4mqouLoCoTKJ?usp=sharing) contains processed data from all stages of the competition.
- _.parquet_ files - dataframes containing the labels and transcription of each word selected by the model in a whisper
- _.pkl_ files contain the audio themselves
