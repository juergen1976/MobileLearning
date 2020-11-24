import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization

# Let's generate `tf.data.Dataset` objects for each dataframe:
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("locked")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

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

def train_model(**kwargs):
    ti = kwargs['ti']
    loaded = ti.xcom_pull(task_ids='preprocess')
    train_dataframe = loaded[0]
    val_dataframe = loaded[1]

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

    # Save the model for further usage in other workflows
    model.save(os.getcwd() + kwargs['initial_model_path'])