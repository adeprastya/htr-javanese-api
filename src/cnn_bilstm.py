"""
CNN-BiLSTM Architecture for Javanese Script Recognition.
"""

import copy

import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """
    CNN-BiLSTM model for sequence recognition with CTC loss.

    Architecture:
        1. CNN: Feature extraction + downsampling (width/4, height→1)
        2. BiLSTM: Sequential context modeling
        3. Linear: Character classification

    Args:
        num_classes: Number of output classes (characters + blank)
        cnn_layers: CNN depth (3-7 layers)
        rnn_layers: BiLSTM depth (1-3 layers)
    """

    def __init__(self, num_classes: int, cnn_layers: int, rnn_layers: int):
        super().__init__()

        # Validate architecture parameters
        if cnn_layers not in [3, 4, 5, 6, 7]:
            raise ValueError(f"cnn_layers must be 3-7, got {cnn_layers}")
        if rnn_layers not in [1, 2, 3]:
            raise ValueError(f"rnn_layers must be 1-3, got {rnn_layers}")

        self.num_classes = num_classes
        self.cnn_layers = cnn_layers
        self.rnn_layers = rnn_layers

        # ========================================
        # Feature Extraction & Downsampling (CNN)
        # ========================================
        cnn_blocks = {
            # Stage 1 Transition (1 -> 64 channels) & Downsampling (48 -> 24 height)
            "stage1_transition": [
                nn.Conv2d(1, 64, kernel_size=3, padding=1, padding_mode="zeros"),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),
            ],
            # Stage 1 Unit (64 -> 64 channels)
            "stage1_unit": [
                nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode="zeros"),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ],
            # Stage 2 Transition (64 -> 128 channels) & Downsampling (24 -> 12 height)
            "stage2_transition": [
                nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode="zeros"),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),
            ],
            # Stage 2 Unit (128 -> 128 channels)
            "stage2_unit": [
                nn.Conv2d(128, 128, kernel_size=3, padding=1, padding_mode="zeros"),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ],
            # Stage 3 Transition (128 -> 256 channels) & Downsampling (12 -> 6 height)
            "stage3_transition": [
                nn.Conv2d(128, 256, kernel_size=3, padding=1, padding_mode="zeros"),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),
            ],
            # Global Vertical Pooling (6 -> 1 height)
            "global_pool": [nn.AdaptiveAvgPool2d((1, None))],
        }

        cnn_structure_map = {
            3: [
                "stage1_transition",
                "stage2_transition",
                "stage3_transition",
                "global_pool",
            ],
            4: [
                "stage1_transition",
                "stage2_transition",
                "stage2_unit",
                "stage3_transition",
                "global_pool",
            ],
            5: [
                "stage1_transition",
                "stage1_unit",
                "stage2_transition",
                "stage2_unit",
                "stage3_transition",
                "global_pool",
            ],
            6: [
                "stage1_transition",
                "stage1_unit",
                "stage2_transition",
                "stage2_unit",
                "stage2_unit",
                "stage3_transition",
                "global_pool",
            ],
            7: [
                "stage1_transition",
                "stage1_unit",
                "stage1_unit",
                "stage2_transition",
                "stage2_unit",
                "stage2_unit",
                "stage3_transition",
                "global_pool",
            ],
        }

        # Build CNN
        layers = []  # 3 / 4 / 5 / 6 / 7
        for block_name in cnn_structure_map[cnn_layers]:
            block_layers = copy.deepcopy(cnn_blocks[block_name])
            layers.extend(block_layers)
        self.cnn = nn.Sequential(*layers)

        # ========================================
        # Contextual Sequence Modeling (BiLSTM)
        # ========================================
        self.rnn = nn.LSTM(
            input_size=256,  # H' * C
            hidden_size=256,  # H' * C
            num_layers=rnn_layers,  # 1 / 2 / 3
            bidirectional=True,
            batch_first=True,
        )

        # ========================================
        # Class Distribution Probabilities (Linear)
        # ========================================
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        # Feature Extraction & Downsampling
        x = self.cnn(x)  # CNN: [B, 1, H, W] -> [B, 256, 1, W/4]

        # Reshape for RNN: [B, 256, 1, W/4] -> [B, W/4, 256]
        x = x.squeeze(2)  # [B, 256, W/4]
        x = x.permute(0, 2, 1)  # [B, W/4, 256]

        # Contextual Sequence Modeling
        x, _ = self.rnn(x)  # BiLSTM: [B, W/4, 256] → [B, W/4, 512]

        # Class Distribution Probabilities
        x = self.fc(x)  # Linear: [B, W/4, 512] → [B, W/4, num_classes]

        return x

    def get_model_info(self) -> dict:
        """Get model architecture information."""

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "num_classes": self.num_classes,
            "cnn_layers": self.cnn_layers,
            "rnn_layers": self.rnn_layers,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }
