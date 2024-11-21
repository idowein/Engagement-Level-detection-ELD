import xgboost as xgb
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import os

def model_trainig_and_evaluation (daisee_path, MWDataset_path, features_columns, label_column):

    # Data loading
    train_set = pd.read_csv(daisee_path)
    val_set = pd.read_csv(MWDataset_path)
    # Model training
    model = xgb.XGBClassifier()
    model.fit(train_set[features_columns], train_set["EngagementLevel"])

    # Predictions
    val_set['PredictedEngagementLevel'] = model.predict(val_set[features_columns])

    # Metrics for val set
    print('===========================================================================\nValidation Metrics:')
    # Metrics for validation set
    print(f"F1: {f1_score(val_set[label_column], val_set['PredictedEngagementLevel'])}")
    print(f"Precision: {precision_score(val_set[label_column], val_set['PredictedEngagementLevel'])}")
    print(f"Recall: {recall_score(val_set[label_column], val_set['PredictedEngagementLevel'])}")
    print(f"Accuracy: {accuracy_score(val_set[label_column], val_set['PredictedEngagementLevel'])}")

def main():

    # DAiSEE labels path
    daisee_training_path = r"C:\Users\idowe\PycharmProjects\MWD\engagement_level_detection\new_labels\new_train_path.csv"

    #MWDatsdet labels path
    MWDataset_path = r"C:\Users\idowe\Datasets\MWDataset\Labels\media_pipe_labels.csv"

    # Predicting level of engagement based on 4 classifications
    features_columns = ['Blinking', 'Smiling', 'Head Movement', 'GenerativeTimer']
    label_column = "MindWandering"
    model_trainig_and_evaluation(daisee_training_path, MWDataset_path, features_columns, label_column)

if __name__ == '__main__':
    main()