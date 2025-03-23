import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import face_recognition
from torch import nn
import os
from PIL import Image as pImage
from flask import Flask, render_template_string, request, jsonify
import tempfile

# Define global variables
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)
inv_normalize = transforms.Normalize(mean=-1*np.divide(mean,std), std=np.divide([1,1,1],std))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define transformations
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Define Model
class Model(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(hidden_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, -1)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# Define Dataset
class ValidationDataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length
    
    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        for frame in self.frame_extract(video_path):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)[:self.count]
        return frames.unsqueeze(0)
    
    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success, image = vidObj.read()
        while success:
            yield image
            success, image = vidObj.read()

# Prediction functions
def predict(model, img):
    fmap, logits = model(img.to(device))
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    return int(prediction.item()), confidence

def load_model(model_path, sequence_length):
    model = Model(2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def run_prediction(video_path, model_path, sequence_length=60):
    model = load_model(model_path, sequence_length)
    dataset = ValidationDataset([video_path], sequence_length=sequence_length, transform=train_transforms)
    prediction, confidence = predict(model, dataset[0])
    return "REAL" if prediction == 1 else "FAKE", confidence

# Flask application
app = Flask(__name__)

# HTML template (Your existing HTML_TEMPLATE string goes here)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Powered Deep Fake Detection</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary: #2A2A72;
            --secondary: #009FFD;
            --success: #3BB143;
            --danger: #FF4162;
            --light: #F4F4F8;
            --dark: #232528;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', sans-serif;
        }

        body {
            background: linear-gradient(135deg, var(--light) 0%, #e6e6e6 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .navbar {
            background: var(--primary);
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .nav-container {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .brand {
            color: white;
            font-size: 1.5rem;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .brand i {
            font-size: 1.8rem;
        }

        .nav-links a {
            color: var(--light);
            text-decoration: none;
            margin-left: 1rem;
        }

        .container {
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 1rem;
            flex: 1;
        }

        .upload-section {
            background: white;
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 2rem 0;
            text-align: center;
        }

        .upload-zone {
            border: 2px dashed var(--secondary);
            border-radius: 10px;
            padding: 2rem;
            margin: 1rem 0;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .upload-zone:hover {
            background: rgba(0, 159, 253, 0.05);
            border-color: var(--primary);
        }

        #file-input {
            display: none;
        }

        .btn {
            background: var(--secondary);
            color: white;
            border: none;
            padding: 0.8rem 2rem;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 3px 8px rgba(0,0,0,0.2);
        }

        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }

        .result-card {
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 3px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }

        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.8);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }

        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 5px solid var(--light);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        footer {
            background: var(--primary);
            color: white;
            text-align: center;
            padding: 1rem;
            margin-top: auto;
        }

        .mb-2 { margin-bottom: 0.5rem; }
        .mb-3 { margin-bottom: 1rem; }
        .mt-2 { margin-top: 0.5rem; }
        .mt-3 { margin-top: 1rem; }
        .text-muted { color: #6c757d; }
        .text-secondary { color: #6c757d; }
        .text-primary { color: var(--primary); }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="nav-container">
            <a href="/" class="brand">
                <i class="fas fa-shield-alt"></i>
                AI-Powered Deep Fake Detection
            </a>
            <div class="nav-links">
                <a href="#" class="text-light">About</a>
                <a href="#" class="text-light">Contact</a>
            </div>
        </div>
    </nav>

    <div class="loading-overlay">
        <div class="loading-spinner"></div>
    </div>

    <main class="container">
        <div class="upload-section">
            <h1 class="mb-2">AI Video Authenticator</h1>
            <p class="text-muted mb-3">Upload a video to verify its authenticity</p>
            
            <form id="upload-form" class="mb-4" method="POST" enctype="multipart/form-data">
                <div class="upload-zone">
                    <i class="fas fa-cloud-upload-alt fa-3x mb-3 text-primary"></i>
                    <p class="mb-1">Drag and drop video file here</p>
                    <p class="text-secondary">or</p>
                    <label for="file-input" class="btn">
                        <i class="fas fa-file-upload"></i>
                        Choose File
                    </label>
                    <input type="file" id="file-input" name="video" accept="video/*">
                </div>
                <div id="file-name" class="text-secondary mt-2"></div>
                <button type="submit" class="btn btn-lg mt-3">
                    <i class="fas fa-search"></i>
                    Analyze Video
                </button>
            </form>

            <div class="supported-formats">
                <p class="text-muted">Supported formats: MP4, AVI, MOV</p>
            </div>
        </div>

        <div id="results-section" style="display: none;">
            <!-- Results will be dynamically inserted here -->
        </div>
    </main>

    <footer>
        <p>Developed by Abhinay, Bharath, Umesh | AI-Powered Deep Fake Detection © 2025</p>
    </footer>

    <script>
        // Show loading spinner on form submission
        document.getElementById('upload-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            
            // Show loading spinner
            document.querySelector('.loading-overlay').style.display = 'flex';
            
            try {
                const response = await fetch('/', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    // Create and show results
                    const resultsSection = document.getElementById('results-section');
                    resultsSection.style.display = 'block';
                    resultsSection.innerHTML = `
                        <div class="upload-section">
                            <h2 class="mb-3">Analysis Results</h2>
                            <div class="result-card" style="padding: 2rem;">
                                <h3 class="mb-2">Video: ${result.filename}</h3>
                                <p class="mb-2" style="font-size: 1.2rem; color: ${result.prediction === 'REAL' ? 'green' : 'red'};">
                                    <strong>Verdict: ${result.prediction}</strong>
                                </p>
                                <p class="text-secondary">Confidence: ${result.confidence.toFixed(2)}%</p>
                            </div>
                        </div>
                    `;
                } else {
                    alert('Error: ' + (result.error || 'Unknown error occurred'));
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                // Hide loading spinner
                document.querySelector('.loading-overlay').style.display = 'none';
            }
        });

        // Drag and drop functionality
        const dropZone = document.querySelector('.upload-zone');
        const fileInput = document.getElementById('file-input');

        dropZone.addEventListener('click', () => fileInput.click());
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = 'var(--primary)';
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.style.borderColor = 'var(--secondary)';
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            fileInput.files = e.dataTransfer.files;
            updateFileName();
        });

        fileInput.addEventListener('change', updateFileName);

        function updateFileName() {
            if (fileInput.files.length > 0) {
                document.getElementById('file-name').textContent = fileInput.files[0].name;
            }
        }
    </script>
</body>
</html>
'''


@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    if video:
        try:
            # Create a temporary file to store the uploaded video
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, video.filename)
            video.save(video_path)
            
            # Path to your trained model
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")
            
            # Run the prediction
            prediction, confidence = run_prediction(video_path, model_path)
            
            # Clean up the temporary file
            os.remove(video_path)
            os.rmdir(temp_dir)
            
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'filename': video.filename
            }
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)