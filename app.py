import os
from flask import Flask, render_template, request, jsonify
from google.cloud import vision
from google.oauth2 import service_account

app = Flask(__name__)

# Configure Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials/google_cloud_key.json'



# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/ocr', methods=['GET', 'POST'])
def ocr():
  if request.method == 'POST':
      if 'file' not in request.files:
          return jsonify({'error': 'No file part'})
      
      file = request.files['file']
      
      if file.filename == '':
          return jsonify({'error': 'No selected file'})
      
      if file:
          filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
          file.save(filename)
          extracted_text = perform_ocr(filename)
          return jsonify({'text': extracted_text, 'image_path': filename})
  
  return render_template('ocr.html')
def perform_ocr(image_path):
  client = vision.ImageAnnotatorClient()

  with open(image_path, 'rb') as image_file:
      content = image_file.read()

  image = vision.Image(content=content)
  response = client.text_detection(image=image)
  texts = response.text_annotations

  if texts:
      return texts[0].description
  else:
      return "No text detected"


@app.route('/image_recognition', methods=['GET', 'POST'])
def image_recognition():
  if request.method == 'POST':
      if 'file' not in request.files:
          return jsonify({'error': 'No file part'})
      
      file = request.files['file']
      
      if file.filename == '':
          return jsonify({'error': 'No selected file'})
      
      if file:
          filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
          file.save(filename)
          labels = perform_image_recognition(filename)
          return jsonify({'labels': labels, 'image_path': filename})
  
  return render_template('image_recognition.html')

def perform_image_recognition(image_path):
  client = vision.ImageAnnotatorClient()

  with open(image_path, 'rb') as image_file:
      content = image_file.read()

  image = vision.Image(content=content)
  response = client.label_detection(image=image)
  labels = response.label_annotations

  return [{'description': label.description, 'score': label.score} for label in labels]

@app.route('/face_detection', methods=['GET', 'POST'])
def face_detection():
  if request.method == 'POST':
      if 'file' not in request.files:
          return jsonify({'error': 'No file part'})
      
      file = request.files['file']
      
      if file.filename == '':
          return jsonify({'error': 'No selected file'})
      
      if file:
          filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
          file.save(filename)
          face_data = perform_face_detection(filename)
          return jsonify({'face_data': face_data, 'image_path': filename})
  
  return render_template('face_detection.html')

def perform_face_detection(image_path):
  client = vision.ImageAnnotatorClient()

  with open(image_path, 'rb') as image_file:
      content = image_file.read()

  image = vision.Image(content=content)
  response = client.face_detection(image=image)
  faces = response.face_annotations

  face_data = []
  for face in faces:
      face_info = {
          'joy': face.joy_likelihood.name,
          'sorrow': face.sorrow_likelihood.name,
          'anger': face.anger_likelihood.name,
          'surprise': face.surprise_likelihood.name,
          'detection_confidence': face.detection_confidence,
      }
      face_data.append(face_info)

  return face_data

if __name__ == '__main__':
  app.run(debug=True)

# Created/Modified files during execution:
print("app.py")