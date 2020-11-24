from pandas_profiling import ProfileReport
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

# Read movement data, this could be from several locations like S3 buckets
movements = pd.read_csv("../../data/SmartMovementExport.csv")
movements.head()
#CSV_COLUMN_NAMES = ['id', 'acceloX', 'acceloY', 'acceloZ', 'userAcceloX', 'userAcceloY','userAcceloZ', 'gyroX', 'gyroY', 'gyroZ', 'lightSensor', 'locked']

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(movements, movements['locked'], test_size=0.33, random_state=42)

# Explore the smart movement data with PandasProfiling
profile = ProfileReport(movements, title="Pandas Profiling Report")
profile.to_widgets()

# Create training and validation set
val_dataframe = movements.sample(frac=0.2, random_state=1337)
train_dataframe = movements.drop(val_dataframe.index)
print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

# Let's generate `tf.data.Dataset` objects for each dataframe:
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("locked")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

# Batch the datasets
train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

# Now build a model
# Numerical inout layer
acceloX = keras.Input(shape=(1,), name="acceloX")
acceloY = keras.Input(shape=(1,), name="acceloY")
acceloZ = keras.Input(shape=(1,), name="acceloZ")
userAcceloX = keras.Input(shape=(1,), name="userAcceloX")
userAcceloY = keras.Input(shape=(1,), name="userAcceloY")
userAcceloZ = keras.Input(shape=(1,), name="userAcceloZ")
gyroX = keras.Input(shape=(1,), name="gyroX")
gyroY = keras.Input(shape=(1,), name="gyroY")
gyroZ = keras.Input(shape=(1,), name="gyroZ")
lightSensor = keras.Input(shape=(1,), name="lightSensor")
all_inputs = [
        acceloX,
        acceloY,
        acceloZ,
        userAcceloX,
        userAcceloY,
        userAcceloZ,
        gyroX,
        gyroY,
        gyroZ,
        lightSensor
    ]

from tensorflow.keras.layers.experimental.preprocessing import Normalization
def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature

acceloX_encoded = encode_numerical_feature(acceloX, "acceloX", train_ds)
acceloY_encoded = encode_numerical_feature(acceloY, "acceloY", train_ds)
acceloZ_encoded = encode_numerical_feature(acceloZ, "acceloZ", train_ds)
userAcceloX_encoded = encode_numerical_feature(userAcceloX, "userAcceloX", train_ds)
userAcceloY_encoded = encode_numerical_feature(userAcceloY, "userAcceloY", train_ds)
userAcceloZ_encoded = encode_numerical_feature(userAcceloZ, "userAcceloZ", train_ds)
gyroX_encoded = encode_numerical_feature(gyroX, "gyroX", train_ds)
gyroY_encoded = encode_numerical_feature(gyroY, "gyroY", train_ds)
gyroZ_encoded = encode_numerical_feature(gyroZ, "gyroZ", train_ds)
lightSensor_encoded = encode_numerical_feature(lightSensor, "lightSensor", train_ds)

all_features = layers.concatenate(
    [
        acceloX_encoded,
        acceloY_encoded,
        acceloZ_encoded,
        userAcceloX_encoded,
        userAcceloY_encoded,
        userAcceloZ_encoded,
        gyroX_encoded,
        gyroY_encoded,
        gyroZ_encoded,
        lightSensor_encoded
    ]
)

x = layers.Dense(32, activation="relu")(all_features)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

# Train the model
model.fit(train_ds, epochs=50, validation_data=val_ds)

# Inference on new data
sample = {
    "acceloX": 10,
    "acceloY": -11,
    "acceloZ": 1.8,
    "userAcceloX": 0.48,
    "userAcceloY": 0.73,
    "userAcceloZ": 3.04,
    "gyroX": 5.1,
    "gyroY": -3.1,
    "gyroZ": 1.79,
    "lightSensor": 8.0
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = model.predict(input_dict)
print("Likelihood that smartphone is unlocked:", predictions[0][0])