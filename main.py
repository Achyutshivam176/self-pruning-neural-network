import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)

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


class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(784, 128)
        self.fc2 = PrunableLinear(128, 64)
        self.fc3 = PrunableLinear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def sparsity_loss(model):
    loss = 0
    for name, param in model.named_parameters():
        if "gate_scores" in name:
            gates = torch.sigmoid(param)
            loss += torch.sum(gates)
    return loss


def calculate_sparsity(model, threshold=1e-2):
    total, pruned = 0, 0
    for name, param in model.named_parameters():
        if "gate_scores" in name:
            gates = torch.sigmoid(param)
            total += gates.numel()
            pruned += (gates < threshold).sum().item()
    return (pruned / total) * 100


def train_step(model, optimizer, data, target, lambda_reg):
    optimizer.zero_grad()
    output = model(data)
    classification_loss = F.cross_entropy(output, target)
    reg_loss = sparsity_loss(model)
    total_loss = classification_loss + lambda_reg * reg_loss
    total_loss.backward()
    optimizer.step()
    return total_loss.item()


def run_experiment():
    lambda_values = [0.0001, 0.001, 0.01]
    results = []

    for lam in lambda_values:
        model = PrunableNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for _ in range(5):
            data = torch.randn(32, 1, 28, 28)
            target = torch.randint(0, 10, (32,))
            train_step(model, optimizer, data, target, lam)

        sparsity = calculate_sparsity(model)
        results.append((lam, sparsity))

    return results


def plot_gate_distribution(model):
    values = []
    for name, param in model.named_parameters():
        if "gate_scores" in name:
            gates = torch.sigmoid(param).detach().numpy().flatten()
            values.extend(gates)

    plt.hist(values, bins=50)
    plt.title("Gate Value Distribution")
    plt.savefig("gate_distribution.png")


if __name__ == "__main__":
    model = PrunableNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    data = torch.randn(32, 1, 28, 28)
    target = torch.randint(0, 10, (32,))

    loss = train_step(model, optimizer, data, target, 0.001)
    print("Loss:", loss)

    results = run_experiment()
    print("\nLambda\tSparsity(%)")
    for lam, sp in results:
        print(f"{lam}\t{sp:.2f}")

    plot_gate_distribution(model)
