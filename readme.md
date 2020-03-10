# Final Project

The folder of this repo illustrate the process of our model development

0. Glove Dictionary:

We preprocessed GloVe twitter dictionary to keep only words in english alphabet. We also removed punctuation and tags, as they won't help us a lot in identifying the sentiment. After preprocessing, the dictionary is smaller and the machine learning model will be easier to train

1. Preprocess Library:

In this step, we update the preprocess library. We added the function that allow us to identify as many meaningful word as possible. For instance, "cat/dog" will  be tokenized as ["cat","dog"], "loooooooooog time ago" will be tokenzied as ["looong", "time", "ago"]. There are further improvement we can implement, such as translating each text-emoji or emoji to a specific sentiment

2. Prepare Data:

Use preprocess library to transform tweet into json file with features. This is achieved using AWS Glue, in this folder, we use a local demonstration of work. AWS S3 link to the complete json file:  

3. model_training:

Contain the model training code. This is a version that works both in local and AWS, just need to change config file. Model details are different from AWS, since in AWS we can keep the notebook running and pay little attention to the computation burden. For access to the notebook, contact owner of this github repo.

4. model_performance

local checking performance by loading the save model. refer to the notebook in this folder

5. model_for_deployment

our major model result. For instance, "lstm_64_32_828" means this is the saved model for LSTM structure, with first layer having 64 units, second layer has 32 units, and final test accuracy is 0.828. Our best model is in "lstm_64_32_828", which is also what we are serving.

# Online access of some file:

dev.json: https://final-project-4577.s3.us-east-2.amazonaws.com/data/dev/dev.json

eval.json: https://final-project-4577.s3.us-east-2.amazonaws.com/data/eval/eval.json

train.json: https://final-project-4577.s3.us-east-2.amazonaws.com/data/train/train.json

training.full.csv: https://final-project-4577.s3.us-east-2.amazonaws.com/training.full.csv