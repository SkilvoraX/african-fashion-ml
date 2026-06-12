import io
import os
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from PIL import Image
from torchvision import models, transforms
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# 1. Define the exact same model architecture used during training
def load_textile_model(checkpoint_path):
    # Load base ResNet18
    model = models.resnet18(weights=None) 
    
    # Recreate your custom classifier head
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 5) # 5 categories
    )
    
    # Load your saved weights (mapping to CPU since local laptops usually run Flask on CPU)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval() # Set to evaluation mode (turns off dropout)
    return model

# 2. Define the exact same image preprocessing steps
transform = transforms.Compose([
    transforms.Resize((224, 224)), # ImageFolder automatically handles sizing, but API needs explicit resize
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. CRITICAL: Update this list to match your exact 5 folders in alphabetical order!
# Example: ['adire', 'aso_oke', 'bogolanfini', 'kente', 'korhogo']
class_names = ['adire', 'aso_oke', 'bogolanfini', 'kente', 'korhogo']

# Initialize model
MODEL_PATH = 'optimized_textile_model.pth'
if os.path.exists(MODEL_PATH):
    model = load_textile_model(MODEL_PATH)
    print("✅ Model loaded successfully and ready for API calls!")
else:
    print(f"❌ Error: Could not find '{MODEL_PATH}' in this directory. Please download it from Drive.")

# --- API ROUTES ---

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Textile Classification API is running. Send a POST request to /predict with an image."})

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file was sent in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided in the request'}), 400
        
    file = request.files['image']
    
    try:
        # Read the image byte stream and convert to RGB
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Preprocess and prepare batch dimension (1, C, H, W)
        tensor = transform(image).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        # Format the response
        response = {
            'prediction': class_names[predicted_idx.item()],
            'confidence': f"{confidence.item() * 100:.2f}%"
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f"Failed to process image: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv('PORT') , debug=os.getenv("DEBUG"))