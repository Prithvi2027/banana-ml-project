"""
Model architecture for banana ripeness classification
"""
import torch
import torch.nn as nn
from torchvision import models
import config


class BananaRipenessModel(nn.Module):
    """CNN model for banana ripeness classification"""
    
    def __init__(self, num_classes=config.NUM_CLASSES, model_name='resnet50', pretrained=True):
        super(BananaRipenessModel, self).__init__()
        
        self.model_name = model_name
        
        # Load pretrained model
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            
        elif model_name == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        return self.backbone(x)


def create_model(device='cuda'):
    """Create and initialize the model"""
    model = BananaRipenessModel(
        num_classes=config.NUM_CLASSES,
        model_name=config.MODEL_NAME,
        pretrained=config.PRETRAINED
    )
    model = model.to(device)
    
    print(f"\nModel: {config.MODEL_NAME}")
    print(f"Number of classes: {config.NUM_CLASSES}")
    print(f"Pretrained: {config.PRETRAINED}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
