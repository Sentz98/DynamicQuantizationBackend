import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import copy
import os
from quant_fn import Qtensor, quantize_tensor, dequantize_tensor
from convert import QuantizedNetwork


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x, collect_activations=False):
        activations = []
        for layer in self.network:
            x = layer(x)
            if collect_activations:
                activations.append(x)
        # return (x, activations) if collect_activations else x
        return x


# Train the model
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))


# Evaluate the model
def evaluate_model(model, test_loader, device, quantized=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if quantized:
                inputs = quantize_tensor(inputs, per_channel=True, cast=False)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # break
    print(f"Accuracy: {100 * correct / total:.2f}%, {correct}/{total}")


# Save the trained model
def save_model(model, path="simple_cnn.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# Load the trained model if it exists
def load_model(path="simple_cnn.pth", device=torch.device('cpu')):
    model = SimpleCNN().to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        print(f"Model loaded from {path}")
    else:
        print(f"No saved model found at {path}. Training a new model.")
    return model

def test_models_single_input(model, cmodel, cmodel2, input_image, device):
    # Ensure evaluation mode
    model.eval()
    cmodel.eval()
    cmodel2.eval()

    # Move input to the appropriate device
    input_image = input_image.to(device)

    # Forward pass through the FP model
    with torch.no_grad():
        fp_output = model(input_image.unsqueeze(0))  # Add batch dimension
        fp_pred = torch.argmax(fp_output, dim=1)

    print(f"Floating-point model output: {fp_output}")
    print(f"Floating-point model predicted class: {fp_pred.item()}")

    # Forward pass through the regular quantized model
    with torch.no_grad():
        q_output = cmodel(input_image.unsqueeze(0))  # Add batch dimension
        q_pred = torch.argmax(q_output, dim=1)

    print(f"Regular quantized model output: {q_output}")
    print(f"Regular quantized model predicted class: {q_pred.item()}")

    # Forward pass through the estimated quantized model
    with torch.no_grad():
        q_est_output = cmodel2(input_image.unsqueeze(0))  # Add batch dimension
        q_est_pred = torch.argmax(q_est_output, dim=1)

    print(f"Estimated quantized model output: {q_est_output}")
    print(f"Estimated quantized model predicted class: {q_est_pred.item()}")



# Main function
def main(model_path):
    # Data Preparation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=2)

    # Training Parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load or initialize model (Only train if not already saved)
    model = load_model(path=model_path, device=device)

    if not os.path.exists(model_path):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_model(model, train_loader, criterion, optimizer, device, epochs=10)
        save_model(model, path=model_path)

    # Evaluate model
    print("FP model")
    evaluate_model(model, test_loader, device)

    #print(model)

    # Quantize model
    cmodel = QuantizedNetwork(model.network, symmetric=False, per_channel=False, estimate=False)
    
    print("Regular Dynamic Quantized model")
    evaluate_model(cmodel,test_loader,device)

    # #print(quantized_network)

    cmodel2 = QuantizedNetwork(model.network, symmetric=True, per_channel=False, estimate=True)
    print("Estimated Dynamic Quantized model")
    evaluate_model(cmodel2,test_loader,device)

    #print(quantized_network2)


if __name__ == "__main__":
    model_path = "simple_cnn.pth"
    main(model_path)
