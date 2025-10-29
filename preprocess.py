import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Parameters
image_size = (128, 128)
batch_size = 32
num_classes = 2  # Chronic disease (1) and Normal (0)

# Load images from folders and label chronic diseases as 1 and normal as 0
def load_data(data_dir):
    data = []
    labels = []
    chronic_labels = ['Tumor', 'Stone', 'Cyst']
    
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            label = 1 if folder in chronic_labels else 0  # Chronic disease if in chronic_labels else Normal
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
                img = tf.keras.utils.load_img(image_path, target_size=image_size)
                img_array = tf.keras.utils.img_to_array(img)
                data.append(img_array)
                labels.append(label)
    
    return np.array(data), np.array(labels)

# Load your dataset
data_dir = 'C:/Users/MUSA/OneDrive/Desktop/capstone/ChronicKidneydisease/dataset'  # Set your folder path
X, y = load_data(data_dir)

# Normalize images
X = X.astype('float32') / 255.0

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN Model
def create_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # 2 classes: Chronic (1) vs Normal (0)
    return model

# LSTM Model (assuming sequential data)
def create_lstm_model(input_shape):
    model = models.Sequential()
    model.add(layers.LSTM(128, return_sequences=False, input_shape=input_shape))  # Set return_sequences=False
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Dense (Fully connected) Model
def create_dense_model(input_shape):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # 2 classes: Chronic (1) vs Normal (0)
    return model

# Create the models
cnn_model = create_cnn_model((image_size[0], image_size[1], 3))

# For LSTM, you need sequential data, not images
# Assuming you have some sequence data for LSTM, replace the following line with your actual sequence input
lstm_model = create_lstm_model((10, 128))  # This is dummy; replace with real sequential data

dense_model = create_dense_model((image_size[0], image_size[1], 3))

# Compile the models
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dense_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summaries
cnn_model.summary()
lstm_model.summary()
dense_model.summary()

# Prepare LSTM input data
# You need to provide the actual sequential input for LSTM if not already provided.
num_lstm_samples = len(X_train)
lstm_input_train = np.random.rand(num_lstm_samples, 10, 128)  # Dummy LSTM input
lstm_input_test = np.random.rand(len(X_test), 10, 128)  # Dummy LSTM input for testing

# Train the models separately
# CNN Model Training
cnn_model.fit(X_train, y_train, epochs=10, batch_size=batch_size, validation_data=(X_test, y_test))

# LSTM Model Training (Ensure proper sequential data)
lstm_model.fit(lstm_input_train, y_train, epochs=10, batch_size=batch_size, validation_data=(lstm_input_test, y_test))

# Dense Model Training
dense_model.fit(X_train, y_train, epochs=10, batch_size=batch_size, validation_data=(X_test, y_test))

# Save the models separately
cnn_model.save('cnn_model.h5')
lstm_model.save('lstm_model.h5')
dense_model.save('dense_model.h5')