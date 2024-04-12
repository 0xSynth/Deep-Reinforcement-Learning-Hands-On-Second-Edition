import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, num_inputs, num_classes, dropout_prob=0.1):
        super().__init__()

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, 5),
            nn.ReLU(),
            nn.Linear(5, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Dropout(p=dropout_prob),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)

if __name__ == "__main__":
    print("Cuda's availability is %s" % torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = Network(num_inputs=2, num_classes=3).to(device)
    v = torch.Tensor([[2, 3]]).to(device)

    out = net(v)
    print(out)