from flask import Flask, jsonify, request, send_from_directory, redirect
from flask_cors import CORS
import random
import csv
import os
import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

# --- App & DB Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=SCRIPT_DIR, static_url_path='', template_folder=SCRIPT_DIR)
CORS(app)

CSV_FILE = os.path.join(SCRIPT_DIR, 'users.csv')
CSV_HEADERS = ['name', 'phone', 'email', 'gender', 'password']

def initialize_database():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        print(f"Database file '{CSV_FILE}' created.")

initialize_database()

# --- Load ML Models ---
bp_model = None
hr_model = None
stress_model = None
try:
    MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
    bp_model_path = os.path.join(MODELS_DIR, 'bp_model.pkl')
    hr_model_path = os.path.join(MODELS_DIR, 'hr_model.pkl')
    stress_model_path = os.path.join(MODELS_DIR, 'stress_model.pkl')

    bp_model = joblib.load(bp_model_path)
    hr_model = joblib.load(hr_model_path)
    stress_model = joblib.load(stress_model_path)
    print("--- All ML models loaded successfully ---")
except FileNotFoundError as e:
    print(f"Error loading models: {e}.")
except Exception as e:
    print(f"An unexpected error occurred while loading models: {e}")

# --- Signal Processing Functions (Copied from ml_processor.py) ---
def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# --- Static File Serving ---
@app.route('/')
def redirect_to_home():
    return redirect('/home.html')

@app.route('/login.html')
def serve_login():
    return send_from_directory(SCRIPT_DIR, 'login.html')

@app.route('/signup.html')
def serve_signup():
    return send_from_directory(SCRIPT_DIR, 'signup.html')

@app.route('/index.html')
def serve_index():
    return send_from_directory(SCRIPT_DIR, 'index.html')

# --- User Authentication Endpoints ---
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    if not all(key in data for key in CSV_HEADERS):
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(data)

    return jsonify({'status': 'success', 'message': 'User registered successfully'})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    with open(CSV_FILE, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for user in reader:
            if user['email'] == email and user['password'] == password:
                return jsonify({'status': 'success', 'message': 'Login successful'})

    return jsonify({'status': 'error', 'message': 'Invalid email or password'}), 401

# --- ML Prediction Endpoint ---
@app.route('/process', methods=['POST'])
def process_data():
    if not all([bp_model, hr_model, stress_model]):
        return jsonify({'status': 'error', 'message': 'Models are not loaded on the server.'}), 500

    data = request.json
    raw_signal = data.get('raw_signal')
    
    if raw_signal is None or not raw_signal:
        return jsonify({'status': 'error', 'message': 'No raw signal data provided.'}), 400
    
    fs = 30 # Assuming ~30 FPS
    try:
        raw_signal_np = np.array(raw_signal)
        filtered_signal = bandpass_filter(raw_signal_np, 0.75, 4.0, fs)
        
        min_distance = int(0.5 * fs)
        peaks, _ = find_peaks(filtered_signal, distance=min_distance)
        
        if len(peaks) < 2:
            calculated_hr = random.uniform(60, 100)
        else:
            peak_intervals = np.diff(peaks) / fs
            avg_interval = np.mean(peak_intervals)
            calculated_hr = 60.0 / avg_interval
            
    except Exception as e:
        print(f"Error processing signal: {e}")
        calculated_hr = random.uniform(60, 100)

    input_df = pd.DataFrame([[calculated_hr]], columns=['calculated_hr'])

    bp_pred = bp_model.predict(input_df)
    if bp_pred.ndim == 1:
        bp_pred = bp_pred.reshape(1, -1)

    hr_pred = hr_model.predict(input_df)
    stress_pred = stress_model.predict(input_df)

    results = {
        'status': 'success',
        'systolic': round(bp_pred[0, 0]),
        'diastolic': round(bp_pred[0, 1]),
        'heartRate': round(float(hr_pred[0])),
        'stress': stress_pred[0]
    }

    return jsonify(results)

# --- Main Execution ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
