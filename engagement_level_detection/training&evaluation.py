import xgboost as xgb
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import os

def binary_adjusting(original_path, input_path, base_directory):
    "Creating a new binary CSV with dir name"

    # change input file to binary label
    df = pd.read_csv(original_path) # Read the data into a DataFrame
    df["NonEngagementSum"] = df[["Boredom", "Confusion", "Frustration"]].sum(axis=1) # Calculate the sum of all non-engagement columns
    df["EngagementLevel"] = df.apply(lambda row: 1 if row["Engagement"] - row["NonEngagementSum"] >= 0 else 0, axis=1) # Compare Engagement with NonEngagementSum and define the new engagement level
    df.drop(columns=["NonEngagementSum"], inplace=True) # Drop the intermediate NonEngagementSum column if not needed

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
    df.to_csv(input_path, index=False)

def create_generative_timer(input_path):
    "Append CSV with generative timer that represent the video length of each student"

    # Load the dataset
    df = pd.read_csv(input_path)

    # Ensure the Directory column exists
    if 'Directory' not in df.columns:
        raise KeyError("The 'Directory' column is missing from the dataset.")

    # Initialize the GenerativeTimer column
    generative_timer = []
    current_timer = 10

    # Iterate through rows and calculate GenerativeTimer
    previous_directory = None
    for directory in df['Directory']:
        if directory == previous_directory:
            current_timer += 10
        else:
            current_timer = 10  # Reset timer for a new directory
        generative_timer.append(current_timer)
        previous_directory = directory

    # Add the new column to the DataFrame
    df['GenerativeTimer'] = generative_timer

    # Save the updated dataset
    df.to_csv(input_path, index=False)
    print(f"Updated CSV with GenerativeTimer saved to {input_path}")

def dynamic_counter_creation(input_path):
    # Identify when a new sequence starts
    # A new sequence starts when:
    # - The 'Directory' changes
    # - or 'Label' changes from 0 to 1
    # For the first row, we consider it a new sequence

    df = pd.read_csv(input_path) # Read the data into a DataFrame
    mask = df['EngagementLevel'] == 1 # Create a mask where Label == 1

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
    df['DynamicCounter'] = df.groupby(group_ids).cumcount()

    # For rows where Label == 1, increment the counter by 1 and multiply by 10
    df.loc[mask, 'DynamicCounter'] = (df.loc[mask, 'DynamicCounter'] + 1) * 10

    # For rows where Label == 0, set EngagementCounter to 0
    df.loc[~mask, 'DynamicCounter'] = 0

    # Save the updated CSV
    df.to_csv(input_path, index=False)

def merge_label_csvs(new_seg_path, MediaPipe_path, merge_key):
    """
    Merges two label CSV files based on a common key.
    """
    # Load the CSV files
    df1 = pd.read_csv(new_seg_path)
    df2 = pd.read_csv(MediaPipe_path)

    # Remove the file extensions from 'ClipID' in df2
    df2['ClipID'] = df2['ClipID'].astype(str).str.replace('.mp4', '', regex=False).str.replace('.avi', '', regex=False)

    # Convert merge_key columns to string type in both dataframes
    df1[merge_key] = df1[merge_key].astype(str)
    df2[merge_key] = df2[merge_key].astype(str)

    # Check if the merge key exists in both dataframes
    if merge_key not in df1.columns or merge_key not in df2.columns:
        raise KeyError(f"'{merge_key}' column must exist in both CSV files.")

    # Merge the two CSV files
    merged_df = pd.merge(df1, df2, on=merge_key, how='outer', suffixes=('_x', '_y'))

    # Save the merged dataframe to a new CSV
    merged_df.to_csv(new_seg_path, index=False)
    print(f"Merged CSV saved to {new_seg_path}")

def model_trainig_and_evaluation (train_path, val_path, test_path, features_columns, label_column):

    # Data loading
    train_set, val_set, test_set = pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)

    # Model training
    model = xgb.XGBClassifier()
    model.fit(train_set[features_columns], train_set[label_column])

    # Predictions
    train_set['PredictedEngagementLevel'] = model.predict(train_set[features_columns])
    val_set['PredictedEngagementLevel'] = model.predict(val_set[features_columns])
    test_set['PredictedEngagementLevel'] = model.predict(test_set[features_columns])

    # Metrics for train set
    print('Train:')
    print(f"F1: {f1_score(train_set[label_column], train_set['PredictedEngagementLevel'])}")
    print(f"Precision: {precision_score(train_set[label_column], train_set['PredictedEngagementLevel'])}")
    print(f"Recall: {recall_score(train_set[label_column], train_set['PredictedEngagementLevel'])}")
    print(f"Accuracy: {accuracy_score(train_set[label_column], train_set['PredictedEngagementLevel'])}")

    print('===========================================================================\nValidation Metrics:')
    # Metrics for validation set
    print(f"F1: {f1_score(val_set[label_column], val_set['PredictedEngagementLevel'])}")
    print(f"Precision: {precision_score(val_set[label_column], val_set['PredictedEngagementLevel'])}")
    print(f"Recall: {recall_score(val_set[label_column], val_set['PredictedEngagementLevel'])}")
    print(f"Accuracy: {accuracy_score(val_set[label_column], val_set['PredictedEngagementLevel'])}")

    print('===========================================================================\nTest Metrics:')
    # Metrics for test set
    print(f"F1: {f1_score(test_set[label_column], test_set['PredictedEngagementLevel'])}")
    print(f"Precision: {precision_score(test_set[label_column], test_set['PredictedEngagementLevel'])}")
    print(f"Recall: {recall_score(test_set[label_column], test_set['PredictedEngagementLevel'])}")
    print(f"Accuracy: {accuracy_score(test_set[label_column], test_set['PredictedEngagementLevel'])}")

    return train_set, val_set, test_set

def main():

    #DAiSEE original labels path
    train_original_path = r"C:\Users\idowe\PycharmProjects\MWD\OriginalLables\TrainLabels.csv"
    val_original_path = r"C:\Users\idowe\PycharmProjects\MWD\OriginalLables\ValidationLabels.csv"
    test_original_path = r"C:\Users\idowe\PycharmProjects\MWD\OriginalLables\TestLabels.csv"

    # DAiSEE videos path
    base_train_dir = r"C:\Users\idowe\Mind wandering research\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\DataSet\Train"
    base_val_dir = r"C:\Users\idowe\Mind wandering research\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\DataSet\Validation"
    base_test_dir = r"C:\Users\idowe\Mind wandering research\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\DataSet\Test"

    # New binary labels path
    new_train_path = r"C:\Users\idowe\PycharmProjects\MWD\new_labels\new_train_path.csv"
    new_val_path = r"C:\Users\idowe\PycharmProjects\MWD\new_labels\new_val_path.csv"
    new_test_path = r"C:\Users\idowe\PycharmProjects\MWD\new_labels\new_test_path.csv"

    # Creation of binary CSV and dir name addition to new_segment_path
    binary_adjusting(train_original_path, new_train_path, base_train_dir)
    binary_adjusting(val_original_path, new_val_path,base_val_dir)
    binary_adjusting(test_original_path, new_test_path, base_test_dir)

    # Adjusting the new_seg.csv with generative timer
    create_generative_timer(new_train_path)
    create_generative_timer(new_val_path)
    create_generative_timer(new_test_path)

    # Counter CSVs that Media Pipe created
    train_counter_path = r"C:\Users\idowe\PycharmProjects\MWD\MediaPipeLabels\TrainLabels.csv"
    val_counter_path = r"C:\Users\idowe\PycharmProjects\MWD\MediaPipeLabels\ValidationLabels.csv"
    test_counter_path = r"C:\Users\idowe\PycharmProjects\MWD\MediaPipeLabels\TestLabels.csv"

    # Merge new_seg and media pipe counters CSVs
    merge_label_csvs(new_train_path, train_counter_path, "ClipID")
    merge_label_csvs(new_val_path, val_counter_path, "ClipID")
    merge_label_csvs(new_test_path, test_counter_path, "ClipID")

    # Predicting level of engagement based on 4 classifications
    features_columns = ['Blinking', 'Smiling', 'Head Movement', 'GenerativeTimer']
    label_column = "EngagementLevel"
    train_set, val_set, test_set = model_trainig_and_evaluation(new_train_path, new_val_path, new_test_path, features_columns, label_column)
    predicted_train_path = new_train_path
    train_set.to_csv(predicted_train_path, index=False)
    predicted_val_path = new_val_path
    val_set.to_csv(predicted_val_path, index=False)
    predicted_test_path = new_test_path
    test_set.to_csv(predicted_test_path, index=False)

    # Adjusting the new_seg with dynamic timer
    dynamic_counter_creation(new_train_path)
    dynamic_counter_creation(new_val_path)
    dynamic_counter_creation(new_test_path)

if __name__ == '__main__':
    main()