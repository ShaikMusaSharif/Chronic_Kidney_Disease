import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues

from flask import Flask, request, render_template, url_for
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your three separate models
cnn_model = load_model('cnn_model.h5')
lstm_model = load_model('lstm_model.h5')
dense_model = load_model('dense_model.h5')

# Define sequence length for the LSTM model (adjust based on your architecture)
sequence_length = 10

# Map class indices to class names
class_names = ['No Chronic Kidney Disease', 'Chronic Kidney Disease']

# Preprocess the image for CNN and Dense models
def preprocess_image(image_path):
    from tensorflow.keras.preprocessing import image

    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Plot confusion matrix
def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # Specify labels here
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    # Save the plot image
    plot_path = f'static/images/{model_name}_confusion_matrix.png'
    plt.savefig(plot_path)
    plt.close()
    return f'images/{model_name}_confusion_matrix.png'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return predict()
    return render_template('index.html')

def predict():
    if 'file' not in request.files:
        return 'No file selected'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'

    # Save the uploaded image
    image_path = os.path.join('static/images', file.filename)
    file.save(image_path)

    # Preprocess the image for CNN and Dense models
    img_array = preprocess_image(image_path)

    # Generate a placeholder for LSTM model (if needed)
    sequence_input = np.random.random((1, sequence_length, 128))  # Adjust shape for LSTM input

    # Get predictions from all models
    cnn_prediction = cnn_model.predict(img_array)
    lstm_prediction = lstm_model.predict(sequence_input)
    dense_prediction = dense_model.predict(img_array)

    # Get class indices for each model
    cnn_class_index = np.argmax(cnn_prediction, axis=1)[0]
    lstm_class_index = np.argmax(lstm_prediction, axis=1)[0]
    dense_class_index = np.argmax(dense_prediction, axis=1)[0]

    # Map predictions to class names
    cnn_result = class_names[cnn_class_index]
    lstm_result = class_names[lstm_class_index]
    dense_result = class_names[dense_class_index]

    # Assuming actual result is stored in y_true (use the correct label here)
    y_true = [1]  # Placeholder, replace with actual ground truth

    # Calculate Metrics
    y_preds = [cnn_class_index, lstm_class_index, dense_class_index]
    cnn_accuracy = accuracy_score(y_true, [cnn_class_index])
    lstm_accuracy = accuracy_score(y_true, [lstm_class_index])
    dense_accuracy = accuracy_score(y_true, [dense_class_index])

    cnn_f1 = f1_score(y_true, [cnn_class_index], average='macro')
    lstm_f1 = f1_score(y_true, [lstm_class_index], average='macro')
    dense_f1 = f1_score(y_true, [dense_class_index], average='macro')

    # Plot confusion matrix for each model
    cnn_cm_image = save_confusion_matrix(y_true, [cnn_class_index], "cnn")
    lstm_cm_image = save_confusion_matrix(y_true, [lstm_class_index], "lstm")
    dense_cm_image = save_confusion_matrix(y_true, [dense_class_index], "dense")

    # Average the metrics to give final decision
    avg_accuracy = (cnn_accuracy + lstm_accuracy + dense_accuracy) / 3
    avg_f1 = (cnn_f1 + lstm_f1 + dense_f1) / 3

    if avg_accuracy >= 0.5 and avg_f1 >= 0.5:
        final_decision = "Chronic Kidney Disease Detected"
    else:
        final_decision = "No Chronic Kidney Disease Detected"

    # Return results and graphs to the template
    return render_template('index.html', 
                           cnn_prediction=cnn_result, 
                           lstm_prediction=lstm_result, 
                           dense_prediction=dense_result, 
                           cnn_accuracy=cnn_accuracy, 
                           lstm_accuracy=lstm_accuracy, 
                           dense_accuracy=dense_accuracy,
                           cnn_f1=cnn_f1, 
                           lstm_f1=lstm_f1, 
                           dense_f1=dense_f1,
                           cnn_cm_image=cnn_cm_image, 
                           lstm_cm_image=lstm_cm_image, 
                           dense_cm_image=dense_cm_image,
                           final_decision=final_decision,
                           image_file=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
