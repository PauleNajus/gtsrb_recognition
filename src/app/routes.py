from flask import Blueprint, render_template, request, jsonify, current_app
from src.models.neural_network import ResNet, train_neural_model, TrafficSignDataset, evaluate_cnn
from src.models.hyperparameter_tuning import tune_random_forest
from src.data.data_processing import preprocess_image, extract_features, load_and_preprocess_data, load_test_data
from src.database.operations import create_session, add_traffic_sign, add_model_result, get_all_model_results, get_all_traffic_signs
from src.models.evaluation import evaluate_model
from sqlalchemy import create_engine
import torch
from config import Config
import time
import os
from torch.utils.data import DataLoader
import joblib
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename
import numpy as np

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            img = preprocess_image(file_path)
            
            model_type = request.form.get('model_type', 'cnn')
            prediction = predict_cnn(img) if model_type == 'cnn' else predict_random_forest(extract_features(img))
            
            class_name = Config.CLASS_NAMES[prediction]
            
            engine = create_engine(current_app.config['SQLALCHEMY_DATABASE_URI'])
            with create_session(engine) as session:
                add_traffic_sign(session, filename, img.shape[1], img.shape[0], prediction, class_name)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'prediction': class_name,
                'class_id': int(prediction)
            })
    except Exception as e:
        current_app.logger.error(f"Error in upload_image: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
    
def predict_cnn(img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(num_classes=43).to(device)
    model.load_state_dict(torch.load('cnn_model.pth', map_location=device))
    model.eval()
    
    with torch.no_grad():
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item()

def predict_random_forest(features):
    model = joblib.load('random_forest_model.joblib')
    expected_features = model.n_features_in_
    
    if len(features) != expected_features:
        features = np.pad(features, (0, max(0, expected_features - len(features))))[:expected_features]
    
    return model.predict([features])[0]
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@bp.route('/train', methods=['POST'])
def train_model():
    try:
        model_type = request.form.get('model_type', 'cnn')
        
        X, y = load_and_preprocess_data()
        X_test, y_test = load_test_data()
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        start_time = time.time()
        
        if model_type == 'cnn':
            model = ResNet(num_classes=43).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            train_dataset = TrafficSignDataset(X_train, y_train)
            val_dataset = TrafficSignDataset(X_val, y_val)
            test_dataset = TrafficSignDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=256)
            test_loader = DataLoader(test_dataset, batch_size=256)
            
            trained_model, history = train_neural_model(model, train_loader, val_loader, epochs=5)
            train_results = evaluate_cnn(trained_model, train_loader)
            val_results = evaluate_cnn(trained_model, val_loader)
            test_results = evaluate_cnn(trained_model, test_loader)

            torch.save(trained_model.state_dict(), 'cnn_model.pth')
            
        elif model_type == 'random_forest':
            X_reshaped = X.reshape(X.shape[0], -1)
            X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
            X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
            model, best_params = tune_random_forest(X_reshaped, y, n_trials=20)
            
            train_results = evaluate_model(y, model.predict(X_reshaped))
            val_results = evaluate_model(y_val, model.predict(X_val_reshaped))
            test_results = evaluate_model(y_test, model.predict(X_test_reshaped))

            joblib.dump(model, 'random_forest_model.joblib')
        
        else:
            return jsonify({'error': 'Invalid model type'}), 400
        
        training_time = time.time() - start_time

        engine = create_engine(current_app.config['SQLALCHEMY_DATABASE_URI'])
        with create_session(engine) as session:
            add_model_result(
                session, 
                model_type, 
                train_results['accuracy'], 
                train_results['f1_score'],
                val_results['accuracy'],
                val_results['f1_score'],
                test_results['accuracy'],
                test_results['f1_score'],
                training_time
            )
        
        return jsonify({
            'success': 'Model trained and tested successfully',
            'train_accuracy': train_results['accuracy'],
            'train_f1_score': train_results['f1_score'],
            'validation_accuracy': val_results['accuracy'],
            'validation_f1_score': val_results['f1_score'],
            'test_accuracy': test_results['accuracy'],
            'test_f1_score': test_results['f1_score'],
            'training_time': training_time
        })
    except Exception as e:
        current_app.logger.error(f"Error in train_model: {str(e)}", exc_info=True)
        return jsonify({'error': f'An error occurred during model training: {str(e)}'}), 500

@bp.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            img = preprocess_image(file_path)
            features = extract_features(img)
            
            model_type = request.form.get('model_type', 'cnn')
            
            if model_type == 'cnn':
                model = ResNet(num_classes=43)
                model.load_state_dict(torch.load('cnn_model.pth'))
            elif model_type == 'random_forest':
                model = joblib.load('random_forest_model.joblib')
            else:
                return jsonify({'error': 'Invalid model type'})
            
            if model_type in ['cnn', 'vit']:
                model.eval()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                with torch.no_grad():
                    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
                    output = model(features_tensor)
                    _, predicted = torch.max(output, 1)
                class_index = predicted.item()
            else:
                class_index = model.predict([features])[0]
            
            class_name = Config.CLASS_NAMES[class_index]
            
            return jsonify({
                'prediction': class_name,
                'class_index': int(class_index)
            })
    except Exception as e:
        current_app.logger.error(f"Error in predict: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction'})

@bp.route('/model-results')
def view_model_results():
    try:
        engine = create_engine(current_app.config['SQLALCHEMY_DATABASE_URI'])
        with create_session(engine) as session:
            results = get_all_model_results(session)
        return render_template('model_results.html', results=results)
    except Exception as e:
        current_app.logger.error(f"Error in view_model_results: {str(e)}", exc_info=True)
        return jsonify({'error': f'An error occurred while fetching model results: {str(e)}'}), 500

@bp.route('/traffic-sign-results')
def view_traffic_sign_results():
    try:
        engine = create_engine(current_app.config['SQLALCHEMY_DATABASE_URI'])
        with create_session(engine) as session:
            traffic_signs = get_all_traffic_signs(session)
        return render_template('traffic_sign_results.html', traffic_signs=traffic_signs)
    except Exception as e:
        current_app.logger.error(f"Error in view_traffic_sign_results: {str(e)}", exc_info=True)
        return jsonify({'error': f'An error occurred while fetching traffic sign results: {str(e)}'}), 500

@bp.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@bp.errorhandler(500)
def internal_error(error):
    current_app.logger.error(f"Server Error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500