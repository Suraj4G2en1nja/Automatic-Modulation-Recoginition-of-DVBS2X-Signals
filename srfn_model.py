import torch
import torch.nn as nn
import torch.nn.functional as F

class StatisticalResidualFusionNet(nn.Module):
    def __init__(self, num_classes, num_expert_features):
        super().__init__()

        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc_expert = nn.Sequential(
            nn.Linear(num_expert_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        self.fc_fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_raw, x_exp):
        x = F.relu(self.bn1(self.conv1(x_raw)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        x_exp = self.fc_expert(x_exp)

        x = torch.cat((x, x_exp), dim=1)
        return self.fc_fusion(x)