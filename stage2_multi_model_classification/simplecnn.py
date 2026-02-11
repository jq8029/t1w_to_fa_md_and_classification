import torch
import torch.nn as nn

class SimpleCNN(nn.Module): 
    def __init__(self, in_channel=1, num_class=3, n_kernel=16, drop_out=0.5):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channel, n_kernel, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_kernel),
            nn.ReLU(),
            nn.MaxPool2d(2),        # 140

            nn.Conv2d( n_kernel, n_kernel*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_kernel*2),
            nn.ReLU(),
            nn.MaxPool2d(2),        # 70

            nn.Conv2d(n_kernel*2, n_kernel*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_kernel*4),
            nn.ReLU(),
            nn.MaxPool2d(2),        # 35

            nn.Conv2d(n_kernel*4, n_kernel*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_kernel*8),
            nn.ReLU(),
            nn.MaxPool2d(2),        # 17

            nn.Conv2d(n_kernel*8, n_kernel*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_kernel*16),
            nn.ReLU(),
            nn.MaxPool2d(2),        # 8
        )

        # shape (batch, n_kernel*16, 8, 8)
        self.gap = nn.AdaptiveAvgPool2d((1,1))  # (batch, n_kernel*16, 1, 1)

        self.fc = nn.Sequential(
            nn.Linear(n_kernel*16, n_kernel*8),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(n_kernel*8, n_kernel*4),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(n_kernel*4, num_class)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)          # (B, n_kernel*16, 1, 1)
        x = torch.flatten(x, 1)  # (B, n_kernel*16)
        x = self.fc(x)

        return x