from flask import Blueprint, render_template, request, jsonify, current_app
from src.models.neural_network import ResNet, train_neural_model, TrafficSignDataset, evaluate_cnn
from src.models.vision_transformer import create_vit
from src.models.hyperparameter_tuning import tune_random_forest
from src.data.data_processing import preprocess_image, extract_features, load_and_preprocess_data, load_test_data
from src.database.operations import create_session, add_traffic_sign, add_model_result, get_all_model_results
from src.models.evaluation import evaluate_model, evaluate_cnn
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
            
            current_app.logger.info(f"File saved: {file_path}")
            
            img = preprocess_image(file_path)
            current_app.logger.info(f"Image preprocessed successfully. Shape: {img.shape}")
            
            model_type = request.form.get('model_type', 'cnn')
            if model_type == 'cnn':
                prediction = predict_cnn(img)
            elif model_type == 'random_forest':
                features = extract_features(img)
                prediction = predict_random_forest(features)
            else:
                return jsonify({'success': False, 'error': 'Invalid model type'}), 400
            
            current_app.logger.info(f"Prediction made successfully: {prediction}")
            
            class_name = Config.CLASS_NAMES[prediction]
            
            engine = create_engine(current_app.config['SQLALCHEMY_DATABASE_URI'])
            session = create_session(engine)
            add_traffic_sign(session, filename, img.shape[1], img.shape[0], prediction, class_name)
            current_app.logger.info(f"Traffic sign added to database: {filename}")
            
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
    model = ResNet(num_classes=43)
    model_path = 'cnn_model.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
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
        print(f"Warning: Feature mismatch. Expected {expected_features}, got {len(features)}")
        if len(features) < expected_features:
            features = np.pad(features, (0, expected_features - len(features)))
        else:
            features = features[:expected_features]
    
    prediction = model.predict([features])[0]
    return prediction
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@bp.route('/train', methods=['POST'])
def train_model():
    try:
        model_type = request.form.get('model_type', 'cnn')
        
        X, y = load_and_preprocess_data()
        X_test, y_test = load_test_data()
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        start_time = time.time()
        
        if model_type == 'cnn':
            model = ResNet(num_classes=43)
            train_dataset = TrafficSignDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_dataset = TrafficSignDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=32)
            test_dataset = TrafficSignDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=32)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            trained_model, history = train_neural_model(model, train_loader, val_loader, epochs=10)
            train_results = evaluate_cnn(trained_model, train_loader)
            val_results = evaluate_cnn(trained_model, val_loader)
            test_results = evaluate_cnn(trained_model, test_loader)

            torch.save(trained_model.state_dict(), 'cnn_model.pth')
            
        elif model_type == 'random_forest':
            X_reshaped = X.reshape(X.shape[0], -1)
            X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
            X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
            model, best_params = tune_random_forest(X_reshaped, y, n_trials=20)
            
            y_train_pred = model.predict(X_reshaped)
            train_results = evaluate_model(y, y_train_pred)
            
            y_val_pred = model.predict(X_val_reshaped)
            val_results = evaluate_model(y_val, y_val_pred)
            
            y_test_pred = model.predict(X_test_reshaped)
            test_results = evaluate_model(y_test, y_test_pred)

            joblib.dump(model, 'random_forest_model.joblib')
        
        else:
            return jsonify({'error': 'Invalid model type'}), 400
        
        training_time = time.time() - start_time

        engine = create_engine(current_app.config['SQLALCHEMY_DATABASE_URI'])
        session = create_session(engine)
        try:
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
        finally:
            session.close()
        
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
                import joblib
                model = joblib.load('random_forest_model.joblib')
            elif model_type == 'vit':
                model = create_vit()
                model.load_state_dict(torch.load('vit_model.pth'))
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

@bp.route('/results')
def view_results():
    try:
        engine = create_engine(current_app.config['SQLALCHEMY_DATABASE_URI'])
        session = create_session(engine)
        results = get_all_model_results(session)
        session.close()
        return render_template('results.html', results=results)
    except Exception as e:
        current_app.logger.error(f"Error in view_results: {str(e)}", exc_info=True)
        return jsonify({'error': f'An error occurred while fetching results: {str(e)}'}), 500

@bp.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@bp.errorhandler(500)
def internal_error(error):
    current_app.logger.error(f"Server Error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500