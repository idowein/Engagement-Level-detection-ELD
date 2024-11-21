import xgboost as xgb
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import os
import re

def table_arranging(input_path, binary_path, base_directory):
    "Creating a new binary CSV that include generative timer and specific timer from the last non-engagement"

    # change input file to binary label
    df = pd.read_csv(input_path) # Read the data into a DataFrame
    df["NonEngagementSum"] = df[["Boredom", "Confusion", "Frustration"]].sum(axis=1) # Calculate the sum of all non-engagement columns
    df["EngagementLevel"] = df.apply(lambda row: 1 if row["Engagement"] - row["NonEngagementSum"] >= 0 else 0, axis=1) # Compare Engagement with NonEngagementSum and define the new engagement level
    df.drop(columns=["NonEngagementSum"], inplace=True) # Drop the intermediate NonEngagementSum column if not needed
    df.to_csv(binary_path, index=False) # Save the modified DataFrame to a new CSV file

    # Remove the file extensions from 'ClipID' in the DataFrame
    df['ClipID'] = df['ClipID'].astype(str).str.replace('.mp4', '', regex=False).str.replace('.avi', '', regex=False)

    # Prepare a list to hold all ClipID and Directory pairs
    clip_dir_list = []

    for user_id in os.listdir(base_directory):
        user_dir_path = os.path.join(base_directory, user_id)
        if os.path.isdir(user_dir_path):  # Ensure it's a directory
            for clip_id in os.listdir(user_dir_path):
                clip_dir_path = os.path.join(user_dir_path, clip_id)
                if os.path.isdir(clip_dir_path):
                    # Append the ClipID and UserID to the list
                    clip_dir_list.append({'ClipID': clip_id, 'Directory': user_id})

    # Create a DataFrame from the list
    clip_dir_df = pd.DataFrame(clip_dir_list)

    # Merge the original DataFrame with the clip_dir_df on 'ClipID'
    df = df.merge(clip_dir_df, on='ClipID', how='left')

    # Handle missing mappings (if any)
    missing_clips = df[df['Directory'].isnull()]
    if not missing_clips.empty:
        print(f"Warning: {len(missing_clips)} ClipIDs were not found in the directory structure.")
        print(missing_clips[['ClipID']])

    # Save the updated CSV after creating generative Timer
    df.to_csv(binary_path, index=False)

    # Group by 'Directory' and assign 'Timer' values

    # Save the updated CSV
    df.to_csv(binary_path, index=False)

    # Create a mask where Label == 1
    mask = df['EngagementLevel'] == 1

    # Identify when a new sequence starts
    # A new sequence starts when:
    # - The 'Directory' changes
    # - or 'Label' changes from 0 to 1
    # For the first row, we consider it a new sequence
    new_sequence = df['Directory'].ne(df['Directory'].shift()) | (
                (df['EngagementLevel'] == 1) & (df['EngagementLevel'].shift().fillna(-1) != 1))

    # Only consider new sequences where Label == 1
    # So if Label is not 1, set new_sequence to False
    new_sequence = new_sequence & mask

    # Create group IDs by cumulatively summing the new_sequence flags
    group_ids = new_sequence.cumsum()

    # Zero out group IDs where Label != 1
    group_ids = group_ids * mask

    # Calculate the cumulative count within each group
    df['EngagementCounter'] = df.groupby(group_ids).cumcount()

    # For rows where Label == 1, increment the counter by 1 and multiply by 10
    df.loc[mask, 'EngagementCounter'] = (df.loc[mask, 'EngagementCounter'] + 1) * 10

    # For rows where Label == 0, set EngagementCounter to 0
    df.loc[~mask, 'EngagementCounter'] = 0

    # Save the updated CSV
    df.to_csv(binary_path, index=False)

def model_trainig_and_evaluation (train_path, val_path, test_path):
    features_columns = ['Blinking', 'Smiling', 'Head Movement', 'GenerativeTimer']
    label_column = "EngagementLevel"

    train_set, val_set, test_set = pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)

    model = xgb.XGBClassifier()
    model.fit(train_set[features_columns], train_set[label_column])

    train_predictions, val_predictions, test_predictions = model.predict(train_set[features_columns]), model.predict(val_set[features_columns]), model.predict(test_set[features_columns])
    print('Train:')
    print(f"F1: {f1_score(train_set[label_column], train_predictions)}")
    print(f"Precision: {precision_score(train_set[label_column], train_predictions)}")
    print(f"Recall: {recall_score(train_set[label_column], train_predictions)}")
    print(f"Accuracy: {accuracy_score(train_set[label_column], train_predictions)}")

    print('===========================================================================')
    print('val:')
    print(f"F1: {f1_score(val_set[label_column], val_predictions)}")
    print(f"Precision: {precision_score(val_set[label_column], val_predictions)}")
    print(f"Recall: {recall_score(val_set[label_column], val_predictions)}")
    print(f"Accuracy: {accuracy_score(val_set[label_column], val_predictions)}")

    print('===========================================================================')
    print('Test:')
    print(f"F1: {f1_score(test_set[label_column], test_predictions)}")
    print(f"Precision: {precision_score(test_set[label_column], test_predictions)}")
    print(f"Recall: {recall_score(test_set[label_column], test_predictions)}")
    print(f"Accuracy: {accuracy_score(test_set[label_column], test_predictions)}")

def main():

    #DAiSEE original labels path
    train_original_path = r"C:\Users\idowe\Mind wandering research\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\Labels\OriginalLables\TrainLabels.csv"
    base_train_dir = r"C:\Users\idowe\Mind wandering research\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\DataSet\Train"
    val_original_path = r"C:\Users\idowe\Mind wandering research\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\Labels\OriginalLables\ValidationLabels.csv"
    base_val_dir = r"C:\Users\idowe\Mind wandering research\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\DataSet\Validation"
    test_original_path = r"C:\Users\idowe\Mind wandering research\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\Labels\OriginalLables\TestLabels.csv"
    base_test_dir = r"C:\Users\idowe\Mind wandering research\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\DataSet\Test"

    # new binary labels path
    binary_train_path = r"C:\Users\idowe\Mind wandering research\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\Labels\binary_labels\binary_train_path.csv"
    binary_val_path = r"C:\Users\idowe\Mind wandering research\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\Labels\binary_labels\binary_val_path.csv"
    binary_test_path = r"C:\Users\idowe\Mind wandering research\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\Labels\binary_labels\binary_test_path.csv"

    table_arranging(train_original_path, binary_train_path, base_train_dir)
    table_arranging(val_original_path, binary_val_path,base_val_dir)
    table_arranging(test_original_path, binary_test_path, base_test_dir)

    # Counter CSVs that Media Pipe created
    train_counter_path = r"C:\Users\idowe\Mind wandering research\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\ELD\3Lables\TrainSegment\TrainCounter.csv"
    val_counter_path = r"C:\Users\idowe\Mind wandering research\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\ELD\3Lables\ValidationSegment\Valcounter.csv"
    test_counter_path = r"C:\Users\idowe\Mind wandering research\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\ELD\3Lables\TestSegment\TestCounter.csv"

    # Merged CSVs path
    train_merged_path = r"C:\Users\idowe\Mind wandering research\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\Labels\merged_labels\merged_train.csv"
    val_merged_path = r"C:\Users\idowe\Mind wandering research\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\Labels\merged_labels\merged_val.csv"
    test_merged_path = r"C:\Users\idowe\Mind wandering research\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\Labels\merged_labels\merged_test.csv"

    model_trainig_and_evaluation(train_merged_path, val_counter_path, test_counter_path)

if __name__ == '__main__':
    main()