import os
import sqlite3
import ollama
import easyocr
import cv2
import re
import json

from flask import *
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

reader = easyocr.Reader(['en', 'nl'])

@app.route('/')
def home_page():
    return render_template('home.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print(f"Bestand opgeslagen: {file_path}")
            if not os.path.exists(file_path):
                return {"error": "Bestand niet gevonden na opslaan"}, 500

            try:
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError(f"Failed to load image: {file_path}")

                # Get original dimensions
                h, w = image.shape[:2]

                # Preprocess image for better OCR
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Increase contrast slightly
                alpha = 1.2  # Mild contrast enhancement
                beta = 0  # No brightness change
                contrast = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
                # Resize while maintaining aspect ratio
                aspect_ratio = w / h
                new_width = 600  # Slightly higher resolution for better OCR
                new_height = int(new_width / aspect_ratio)
                resized_image = cv2.resize(contrast, (new_width, new_height), interpolation=cv2.INTER_AREA)

                # Extract text with details
                extracted_text = reader.readtext(resized_image, detail=0, paragraph=True)
                extracted_text = ' '.join(extracted_text)  # Combine into single string
                print("Raw extracted text:", extracted_text)

                # Check if any text was extracted
                if not extracted_text.strip():
                    return {"error": "No text extracted from the image"}, 500

                # Clean up extracted text with Ollama
                clean_prompt = f"""
                Clean the following OCR-extracted text from a receipt to improve readability and fix errors. Return **only** the cleaned text as output, without any additional explanations, comments, or markdown symbols. Perform the following:
                - Fix common OCR mistakes (e.g., 'JKG' to '1KG', 'Tota]' to 'Totaal', '1B ,01' to '18.01').
                - Format prices correctly (e.g., '8,99' to '8.99', '18 ,01' to '18.01').
                - Format dates to a consistent format (e.g., '18/09/2025' or '18-09-2025' to '2025-09-18').
                - Remove meaningless strings (e.g., 'JPDF7DS?OSEIKCKJG', 'oken').
                - Normalize spaces and remove extra whitespace.
                - Correct spelling based on context (e.g., 'BROCCOL' to 'Broccoli', 'MERMEDESTRAAT' to 'Merwedestraat').
                - Preserve all relevant details like product names, prices, quantities, dates, addresses, and store names.

                Text:
                {extracted_text}

                Example input:
                HOOFDSTRAAT 3913 Cs ROTTERDAM BROCCOL BLOEMKDOL KiFPENBOUTEN Ca , KIPFILET IKG 8,99 Te BeTalEN 1B ,01 Tota] 18,01 EUR 18/09/2025 14:29

                Example output:
                Merwedestraat 3913 CS Dordrecht Broccoli Bloemkool Kippenbouten Kipfilet 1KG 8.99 Te Betalen 18.01 Totaal 18.01 EUR 2025-09-18 14:29
                """
                clean_response = ollama.chat(
                    model="gemma3",
                    messages=[{"role": "user", "content": clean_prompt}],
                )
                cleaned_text = clean_response["message"]['content'].strip()
                print("Cleaned extracted text:", cleaned_text)

                # Check if cleaned text is valid
                if not cleaned_text.strip():
                    return {"error": "No cleaned text produced"}, 500

                json_prompt = f"""
                Convert the following cleaned text from a receipt into a strictly JSON format. Return **only** valid JSON as output, without any additional text, comments, explanations, or markdown symbols (such as ```). If a field is not clearly readable, use an empty string ("") for strings, 0.0 for floats, and 1 for quantities. Identify the location if present (e.g., address or store name) and determine the receipt type (e.g., "fuel station", "restaurant", "supermarket", or "" if not identifiable) based on keywords like 'ALDI' (supermarket), 'Tango' (fuel station), 'restaurant', or other context. Extract all identifiable items (products or services) with their names, prices, and quantities. For supermarkets, treat each product line as an item and associate prices found near product names (e.g., 'Kipfilet 1KG 8.99' is one item with price 8.99). For fuel receipts, treat fuel type and volume as an item (e.g., 'Euro 95 39.90' is 39 liters). For restaurants, extract menu items or services. Recognize dates in formats like 'DD/MM/YYYY', 'DD-MM-YYYY', or 'YYYY-MM-DD'. Follow the format below exactly.

                Required fields:
                - date: Date of the receipt (format: YYYY-MM-DD, or "" if not readable)
                - total_amount: Total amount (as a float, or 0.0 if not readable)
                - location: Location of the receipt (e.g., address or store name, or "" if not readable)
                - receipt_type: Type of receipt (e.g., "fuel station", "restaurant", "supermarket", or "" if not identifiable)
                - items: List of items, each with:
                  - name: Product or service name (e.g., "Broccoli", "Diesel", "Pizza", or "" if not readable)
                  - price: Price (as a float, or 0.0 if not readable)
                  - quantity: Quantity (as an integer, default 1 if not specified, use volume for fuel)

                Text:
                {cleaned_text}

                Example output:
                {{
                  "date": "2025-09-16",
                  "total_amount": 18.01,
                  "location": "123 Main St, City",
                  "receipt_type": "supermarket",
                  "items": [
                    {{"name": "Broccoli", "price": 2.50, "quantity": 1}},
                    {{"name": "Kipfilet 1KG", "price": 8.99, "quantity": 1}}
                  ]
                }}
                """

                json_response = ollama.chat(
                    model="gemma3",
                    messages=[{"role": "user", "content": json_prompt}],
                )
                content = json_response["message"]['content']
                # Remove markdown symbols
                content = content.replace('```json', '').replace('```', '').strip()
                print("Raw JSON response:", content)


                # Validate JSON
                try:
                    json_data = json.loads(content)
                    return json_data, 200
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON in response", "content": content,
                            "extracted_text": extracted_text, "cleaned_text": cleaned_text}, 500

            except Exception as e:

                return {"error": f"Error processing the image: {str(e)}"}, 500
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)