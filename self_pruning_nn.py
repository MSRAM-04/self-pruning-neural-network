import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Prunable Linear Layer
# ---------------------------
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

# ---------------------------
# Model
# ---------------------------
class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32*32*3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ---------------------------
# Loss Function
# ---------------------------
def compute_loss(output, target, model, lambda_val):
    ce_loss = F.cross_entropy(output, target)
    
    sparsity_loss = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            sparsity_loss += gates.sum()
    
    return ce_loss + lambda_val * sparsity_loss

# ---------------------------
# Sparsity Calculation
# ---------------------------
def calculate_sparsity(model, threshold=1e-2):
    total, pruned = 0, 0
    
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.numel()
            pruned += (gates < threshold).sum().item()
    
    return 100 * pruned / total

# ---------------------------
# Data Loaders (CIFAR-10)
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# ---------------------------
# Training Function
# ---------------------------
def train_model(lambda_val):
    model = PrunableNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5  # increase to 10+ for better accuracy

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = compute_loss(outputs, labels, model, lambda_val)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Lambda {lambda_val} | Epoch {epoch+1} | Loss: {total_loss:.2f}")

    return model

# ---------------------------
# Evaluation
# ---------------------------
def evaluate(model):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# ---------------------------
# Gate Distribution Plot
# ---------------------------
def plot_gates(model):
    all_gates = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy().flatten()
            all_gates.extend(gates)

    plt.hist(all_gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.show()

# ---------------------------
# Run Experiments
# ---------------------------
lambdas = [0.0001, 0.001, 0.01]

for lam in lambdas:
    print("\n==============================")
    print(f"Training with lambda = {lam}")

    model = train_model(lam)
    
    acc = evaluate(model)
    sparsity = calculate_sparsity(model)

    print(f"Final Accuracy: {acc:.2f}%")
    print(f"Sparsity: {sparsity:.2f}%")

    plot_gates(model)
