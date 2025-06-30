from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import shutil
import zipfile
from werkzeug.utils import secure_filename
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
import pickle
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from services.classification import MWPNN
from services.preprocessing import normalize_image,fuzzy_entropy,fuzzy_inclusion,fuzzy_preprocess
import time
from threading import Thread
from flask import Response
import json
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','zip'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dataset paths
DATASET_PATH = 'data/raw_dataset'
PROCESSED_DATASET_PATH = 'data/processed_dataset'
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATASET_PATH, exist_ok=True)

def load_models():
    model_path = 'models/brain_tumor_models.pkl'
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    base_model = Model.from_config(model_data['base_model_config'])
    base_model.set_weights(model_data['base_model_weights'])

    model = MWPNN(sigma=model_data['sigma'])
    model.X_train = model_data['X_train']
    model.y_train = model_data['y_train']
    model.classes = np.unique(model_data['y_train'])

    selected_features = model_data['selected_features']

    return base_model, model, selected_features

base_model, model, selected_features = load_models()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/load_dataset', methods=['POST'])
def load_dataset():
    if 'datasetFiles' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('datasetFiles')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected files'}), 400

    try:
        # Get dataset name from ZIP file or generate one
        dataset_name = None
        zip_file = next((f for f in files if f.filename.lower().endswith('.zip')), None)
        
        if zip_file:
            dataset_name = os.path.splitext(secure_filename(zip_file.filename))[0]
        else:
            dataset_name = f"dataset_{int(time.time())}"
        
        # Create dataset directory (without additional subdirectories)
        dataset_path = os.path.join(DATASET_PATH, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)
        
        # Process files
        total_images = 0
        with_tumor = 0
        without_tumor = 0
        errors = []
        
        for file in files:
            try:
                filename = secure_filename(file.filename)
                if not filename:
                    continue
                
                # Handle ZIP files
                if filename.lower().endswith('.zip'):
                    zip_path = os.path.join(dataset_path, filename)
                    file.save(zip_path)
                    
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            # Extract directly to dataset_path without subdirectories
                            for member in zip_ref.infolist():
                                # Skip directories and non-image files
                                if member.is_dir() or not member.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    continue
                                
                                # Extract to root of dataset directory
                                member.filename = os.path.basename(member.filename)
                                zip_ref.extract(member, dataset_path)
                                total_images += 1
                    except zipfile.BadZipFile:
                        errors.append(f"Invalid ZIP file: {filename}")
                        continue
                    finally:
                        if os.path.exists(zip_path):
                            os.remove(zip_path)
                else:
                    # Handle individual image files
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file.save(os.path.join(dataset_path, filename))
                        total_images += 1
            except Exception as e:
                errors.append(f"Error processing {file.filename}: {str(e)}")
                continue
        
        # Count tumor/no-tumor images
        for filename in os.listdir(dataset_path):
            lower_name = filename.lower()
            if lower_name.endswith(('.png', '.jpg', '.jpeg')):
                if 'tumor' in lower_name or 'yes' in lower_name:
                    with_tumor += 1
                elif 'no' in lower_name:
                    without_tumor += 1
        
        return jsonify({
            'message': f'Dataset "{dataset_name}" loaded successfully',
            'dataset_name': dataset_name,
            'dataset_path': dataset_path,
            'stats': {
                'total_images': total_images,
                'with_tumor': total_images-without_tumor,
                'without_tumor': without_tumor
            },
            'warnings': errors if errors else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/list_datasets')
def list_datasets():
    try:
        datasets = []
        for name in os.listdir(DATASET_PATH):
            dataset_path = os.path.join(DATASET_PATH, name)
            if os.path.isdir(dataset_path):
                images = [f for f in os.listdir(dataset_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                datasets.append({
                    'name': name,
                    'path': dataset_path,
                    'image_count': len(images)
                })
        
        return jsonify({'datasets': datasets})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Global variable to track preprocessing progress
preprocess_status = {
    'progress': 0,
    'message': '',
    'completed': False,
    'error': None,
    'processed_count': 0,
    'output_path': '',
    'sample_images': None
}

def run_preprocessing(dataset_name):
    try:
        dataset_path = os.path.join(DATASET_PATH, dataset_name)
        processed_path = os.path.join(PROCESSED_DATASET_PATH, dataset_name)
        
        os.makedirs(processed_path, exist_ok=True)
        
        image_files = [f for f in os.listdir(dataset_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(image_files)
        
        sample_original = None
        sample_processed = None
        processed_count = 0
        errors = []
        
        for i, filename in enumerate(image_files):
            try:
                # Update progress
                progress = int((i + 1) / total_images * 100)
                preprocess_status['progress'] = progress
                preprocess_status['message'] = f'Processing {i+1}/{total_images}: {filename}'
                
                # Process image
                img_path = os.path.join(dataset_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    errors.append(f"Could not read image: {filename}")
                    continue
                
                # Store first image as sample
                if processed_count == 0:
                    sample_dir = os.path.join('static', 'samples', dataset_name)
                    os.makedirs(sample_dir, exist_ok=True)
                    sample_original = os.path.join(sample_dir, f'original_{filename}')
                    cv2.imwrite(sample_original, img)
                
                # Apply preprocessing
                processed_img = fuzzy_preprocess(img)
                processed_filename = f'processed_{filename}'
                processed_filepath = os.path.join(processed_path, processed_filename)
                cv2.imwrite(processed_filepath, (processed_img * 255).astype(np.uint8))
                
                # Store first processed image as sample
                if processed_count == 0:
                    sample_processed = os.path.join(sample_dir, processed_filename)
                    cv2.imwrite(sample_processed, (processed_img * 255).astype(np.uint8))
                
                processed_count += 1
                
            except Exception as e:
                errors.append(f"Error processing {filename}: {str(e)}")
                continue
        
        # Always return success if any images were processed
        if processed_count > 0:
            preprocess_status['completed'] = True
            preprocess_status['progress'] = 100
            preprocess_status['message'] = f'Successfully processed {processed_count}/{total_images} images'
            if errors:
                preprocess_status['message'] += f' ({len(errors)} errors)'
            preprocess_status['processed_count'] = processed_count
            preprocess_status['output_path'] = processed_path
            preprocess_status['dataset_name'] = dataset_name
            preprocess_status['sample_images'] = {
                'original': f'/static/samples/{dataset_name}/original_{image_files[0]}' if sample_original else None,
                'processed': f'/static/samples/{dataset_name}/processed_{image_files[0]}' if sample_processed else None
            }
        else:
            preprocess_status['error'] = 'No images could be processed. ' + ' '.join(errors)
            preprocess_status['completed'] = True
            
    except Exception as e:
        preprocess_status['error'] = f'Preprocessing failed: {str(e)}'
        preprocess_status['completed'] = True


@app.route('/preprocess_dataset/<dataset_name>', methods=['POST'])
def start_preprocessing(dataset_name):
    dataset_path = os.path.join(DATASET_PATH, dataset_name)
    
    if not os.path.exists(dataset_path):
        return jsonify({'error': f'Dataset "{dataset_name}" not found'}), 404
    
    # Reset status
    global preprocess_status
    preprocess_status = {
        'progress': 0,
        'message': '',
        'completed': False,
        'error': None,
        'processed_count': 0,
        'output_path': '',
        'dataset_name': dataset_name,
        'sample_images': None
    }
    
    # Start preprocessing
    thread = Thread(target=run_preprocessing, args=(dataset_name,))
    thread.start()
    
    return jsonify({'message': f'Preprocessing started for dataset "{dataset_name}"'})
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return jsonify({'error': 'Could not read image'}), 400

            img = cv2.resize(img, (224, 224))
            processed_img = fuzzy_preprocess(img)
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
            cv2.imwrite(processed_path, (processed_img * 255).astype(np.uint8))

            img_rgb = cv2.cvtColor((processed_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            img_array = img_to_array(img_rgb)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = base_model.predict(img_array, verbose=0).flatten()

            if len(selected_features) > 0:
                features = features[selected_features]

            prediction = model.predict(features.reshape(1, -1))[0]
            result = "Tumor Detected" if prediction == 1 else "No Tumor Found"

            return jsonify({
                'result': result,
                'original_image': f'/static/uploads/{filename}',
                'processed_image': f'/static/uploads/processed_{filename}'
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/accuracy')
def accuracy():
    try:
        # Load the necessary data
        X = np.load('models/features.npy')
        y = np.load('models/labels.npy')
        selected_features = np.load('models/selected_features.npy')
        data_X = np.load('models/data_features.npy')
        data_y = np.load('models/data_labels.npy')

        # Apply feature selection and combine the datasets
        X_reduced = X[:, selected_features]
        X_combined = np.vstack((X, data_X))  # shape: (6000, 1920)
        y_combined = np.hstack((y, data_y))  # shape: (6000,)

        # Split into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

        # Initialize and fit the model
        model = MWPNN(sigma=0.9)
        model.fit(X_train, y_train)

        # Predict the labels for the test data
        y_pred = model.predict(X_test)

        # Ensure predictions and ground truth are valid
        if len(y_pred) == 0 or len(y_test) == 0:
            raise ValueError("Model predictions or ground truth are empty!")

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Return the metrics as JSON
        return jsonify({
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)