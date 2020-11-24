import pandas as pd

def preprocess(**kwargs):
    # Read movement data, this could be from several locations like S3 buckets
    # This is an initial training
    movements = pd.read_csv("../../data/SmartMovementExport.csv")

    # Create training and validation set
    val_dataframe = movements.sample(frac=0.2, random_state=1337)
    train_dataframe = movements.drop(val_dataframe.index)
    print(
        "Using %d samples for training and %d for validation"
        % (len(train_dataframe), len(val_dataframe))
    )

    return train_dataframe, val_dataframe