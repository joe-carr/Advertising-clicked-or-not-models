import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt

df = pd.read_csv('advertising.csv')
df.head(20)
df.pop('Timestamp')
df.columns = df.columns.str.replace(' ', '_')

train, val = train_test_split(df, test_size=0.2)
train_labels = train.pop('Clicked_on_Ad')
val_labels = val.pop('Clicked_on_Ad')

inputs = {}

for name, column in train.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

numeric_inputs = {name: data for name, data in inputs.items()
                  if data.dtype == tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = preprocessing.Normalization()
norm.adapt(np.array(train[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

preprocessed_inputs = [all_numeric_inputs]

for name, data in inputs.items():
    if data.dtype == tf.float32:
        continue
    lookup = preprocessing.StringLookup(vocabulary=np.unique(train[name]))
    one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())
    x = lookup(data)
    x = one_hot(x)
    preprocessed_inputs.append(x)

preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
advertising_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
train_features_dict, val_features_dict = {name: np.array(value) for name, value in train.items()}, {
    name: np.array(value) for name, value in val.items()}

model_input = advertising_preprocessing(inputs)
dense_1 = tf.keras.layers.Dense(64, activation='relu')(model_input)
output_tensor = tf.keras.layers.Dense(1, activation='sigmoid')(dense_1)
complete_model = tf.keras.Model(inputs, output_tensor, name='Advertising_model')
complete_model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

tf.keras.utils.plot_model(model=complete_model, rankdir='LR', dpi=72, show_shapes=True, show_layer_names=True,
                          to_file='complete_advertising_model.png')

history = complete_model.fit(x=train_features_dict, y=train_labels, validation_data=(val_features_dict, val_labels),
                             epochs=30)

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
