import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
from fruit_classifier import FruitCNN

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Load the best model
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model = FruitCNN(len(checkpoint['classes'])).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['classes']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image, model_path):
    model, classes = load_model(model_path)

    if image is None:
        return "Please upload an image"

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return f"Predicted: {classes[predicted_class]}\nConfidence: {confidence:.2%}"

def create_interface():
    model_paths = [
        "best_model_adam_cross_entropy.pth",
        "best_model_sgd_cross_entropy.pth",
        "best_model_rmsprop_cross_entropy.pth",
    ]

    interface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="numpy", label="Upload Fruit Image"),
            gr.Dropdown(choices=model_paths, label="Select Model")
        ],
        outputs=gr.Textbox(label="Prediction"),
        title="Fruit Classifier",
        description="Upload an image of a fruit to classify it using our trained CNN model.",
        examples=[
            ["examples/apple.png", "best_model_adam_cross_entropy.pth"],
            ["examples/banana.png", "best_model_adam_cross_entropy.pth"],
            ["examples/orange.png", "best_model_adam_cross_entropy.pth"]
        ]
    )
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch() 