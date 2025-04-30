# Flask application to support Treatment and Medicine Identification

from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

# Load Excel data
EXCEL_FILE = "disease_treatment_dataset.xlsx"
data = pd.read_excel(EXCEL_FILE)

# Load trained model
MODEL_PATH = "trademed_mobilenet.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define Class Indices
class_indices = {
    "Arive-Dantu": 0,
    "Basale": 1,
    "Betel": 2,
    "Crape_Jasmine": 3,
    "Curry": 4,
    "Drumstick": 5,
    "Fenugreek": 6,
    "Guava": 7,
    "Hibiscus": 8,
    "Indian_Beech": 9,
    "Indian_Mustard": 10,
    "Jackfruit": 11,
    "Jamaica_Cherry-Gasagase": 12,
    "Jamun": 13,
    "Jasmine": 14,
    "Karanda": 15,
    "Lemon": 16,
    "Mango": 17,
    "Mexican_Mint": 18,
    "Mint": 19,
    "Neem": 20,
    "Oleander": 21,
    "Parijata": 22,
    "Peepal": 23,
    "Pomegranate": 24,
    "Rasna": 25,
    "Rose_apple": 26,
    "Roxburgh_fig": 27,
    "Sandalwood": 28,
    "Tulsi": 29
}
idx_to_class = {v: k for k, v in class_indices.items()}

# Image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the uploaded image for prediction.
    Args:
        image: PIL Image object
        target_size: Tuple specifying target size for the model
    Returns:
        Numpy array suitable for model prediction
    """
    image = image.resize(target_size)  # Resize to the target size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/treatment', methods=['GET', 'POST'])
def treatment():
    if request.method == 'POST':
        try:
            disease = request.form.get('disease').strip()
            age = int(request.form.get('age'))
            gender = request.form.get('gender').strip().lower()
            level = request.form.get('disease_level').strip().lower()
            prescription_input = request.form.get('prescription', '').strip().lower()
            season_input = request.form.get('season', '').strip().lower()

            # Disease level hierarchy
            level_hierarchy = {'low': 1, 'normal': 2, 'high': 3}

            def parse_age_range(age_str):
                separators = ['-', '–', '—', '−', 'to']
                for sep in separators:
                    if sep in age_str:
                        parts = age_str.split(sep)
                        if len(parts) == 2:
                            try:
                                return int(parts[0].strip()), int(parts[1].strip())
                            except:
                                return None
                return None

            # Filtering logic
            def match_row(row):
                if str(row['Disease']).strip().lower() != disease.lower():
                    return False

                age_range = parse_age_range(str(row['Age']))
                if not age_range:
                    return False
                min_age, max_age = age_range
                if not (min_age <= age <= max_age):
                    return False

                if str(row['Gender']).strip().lower() not in ['any', gender]:
                    return False

                if season_input and row['Season'].strip().lower() not in ['any', season_input]:
                    return False

                if level_hierarchy.get(row['Level of Disease'].strip().lower(), 0) > level_hierarchy.get(level, 0):
                    return False

                if prescription_input:
                    row_presc = row.get('Prescription', '').strip().lower()
                    if prescription_input not in row_presc and row_presc != 'any':
                        return False

                return True

            filtered_data = data[data.apply(match_row, axis=1)]

            if not filtered_data.empty:
                result_row = filtered_data.iloc[0]
                return jsonify({
                    'remedy': result_row['Remedy'],
                    'how_to_use': result_row['How to use'],
                    'prescription': result_row.get('Prescription', ''),
                    'matched_criteria': {
                        'disease': result_row['Disease'],
                        'gender': result_row['Gender'],
                        'season': result_row['Season'],
                        'level': result_row['Level of Disease']
                    }
                })

            return jsonify({'remedy': None})
        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('treatment.html')


@app.route('/medicine-identification', methods=['GET', 'POST'])
def medicine_identification():
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'image' in request.files:
            try:
                uploaded_image = request.files['image']
                image = Image.open(uploaded_image)

                # Preprocess the image and predict using the model
                preprocessed_image = preprocess_image(image)
                predictions = model.predict(preprocessed_image)
                class_idx = np.argmax(predictions)
                class_label = idx_to_class[class_idx]
                confidence = np.max(predictions) * 100

                return jsonify({
                    'result': class_label,
                    'confidence': f"{confidence:.2f}%"
                })
            except Exception as e:
                return jsonify({'error': str(e)})

        # If a text query is provided, handle it as before
        query = request.form.get('query')
        if query:
            medicines = data[data['Medicine'].str.contains(query, case=False)]
            response = [{'name': med, 'image': "placeholder.jpg"} for med in medicines['Medicine']]
            return jsonify(response)
    return render_template('medicine_identification.html')

@app.route('/disease-autocomplete', methods=['GET'])
def disease_autocomplete():
    try:
        term = request.args.get('term', '').strip().lower()

        # Filter diseases from the dataset based on the term
        unique_diseases = data['Disease'].dropna().drop_duplicates()
        filtered_diseases = [
            disease for disease in unique_diseases 
            if term in disease.lower()
        ]

        return jsonify(filtered_diseases)
    except Exception as e:
        return jsonify({'error': f'Disease autocomplete failed: {str(e)}'})


@app.route('/medicine-autocomplete', methods=['GET'])
def medicine_autocomplete():
    try:
        json_file_path = os.path.join(app.root_path, 'medicine_data.json')
        with open(json_file_path, 'r') as file:
            medicines = json.load(file)

        term = request.args.get('term', '').lower()
        filtered_medicines = [
            med for med in medicines if term in med['name'].lower()
        ]

        return jsonify(filtered_medicines)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
