import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from fruit_classifier import FruitCNN, device

def plot_confusion_matrix(model_path, test_loader):
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = FruitCNN(len(checkpoint['classes'])).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get predictions
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=checkpoint['classes'], 
                yticklabels=checkpoint['classes'])
    plt.title(f'Confusion Matrix - {model_path}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_path.replace(".pth", "")}.png')
    plt.close()

def main():
    # Load test data
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import os
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dir = os.path.join('./fruits-360', 'Test')
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)  # Set num_workers=0 to avoid multiprocessing issues
    
    # Plot confusion matrices for all saved models
    model_paths = [
        "best_model_adam_cross_entropy.pth",
        "best_model_sgd_cross_entropy.pth",
        "best_model_rmsprop_cross_entropy.pth"
    ]
    
    for model_path in model_paths:
        try:
            print(f"Plotting confusion matrix for {model_path}...")
            plot_confusion_matrix(model_path, test_loader)
        except Exception as e:
            print(f"Error plotting {model_path}: {str(e)}")

if __name__ == '__main__':
    main() 