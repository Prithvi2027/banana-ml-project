"""
Training script for banana ripeness classification
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

import config
from dataset import create_dataloaders
from model import create_model


class Trainer:
    """Trainer class for model training and evaluation"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create dataloaders
        self.train_loader, self.valid_loader, self.test_loader = create_dataloaders()
        
        # Create model
        self.model = create_model(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Tracking
        self.train_losses = []
        self.valid_losses = []
        self.train_accs = []
        self.valid_accs = []
        self.best_valid_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.valid_loader, desc='Validation')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.valid_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def train(self):
        """Full training loop"""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        
        for epoch in range(config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            valid_loss, valid_acc = self.validate_epoch()
            self.valid_losses.append(valid_loss)
            self.valid_accs.append(valid_acc)
            
            # Learning rate scheduling
            self.scheduler.step(valid_loss)
            
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}%")
            
            # Save best model
            if valid_loss < self.best_valid_loss - config.MIN_DELTA:
                self.best_valid_loss = valid_loss
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'valid_acc': valid_acc,
                }, config.BEST_MODEL_PATH)
                print(f"✓ Model saved! (Valid Loss: {valid_loss:.4f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Save final model
        torch.save(self.model.state_dict(), config.FINAL_MODEL_PATH)
        print(f"\nFinal model saved to {config.FINAL_MODEL_PATH}")
        
        # Plot training history
        self.plot_history()
    
    def plot_history(self):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.valid_losses, label='Valid Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accs, label='Train Accuracy')
        ax2.plot(self.valid_accs, label='Valid Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.MODEL_DIR, 'training_history.png'))
        print(f"Training history saved to {config.MODEL_DIR}/training_history.png")
        plt.close()
    
    def evaluate(self):
        """Evaluate on test set"""
        print("\n" + "="*50)
        print("Evaluating on Test Set")
        print("="*50)
        
        # Load best model
        checkpoint = torch.load(config.BEST_MODEL_PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        all_preds = []
        all_labels = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        print(f"\nTest Accuracy: {test_acc:.2f}%")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        self.plot_confusion_matrix(cm)
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(config.CLASS_NAMES))
        plt.xticks(tick_marks, config.CLASS_NAMES, rotation=45)
        plt.yticks(tick_marks, config.CLASS_NAMES)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(config.MODEL_DIR, 'confusion_matrix.png'))
        print(f"\nConfusion matrix saved to {config.MODEL_DIR}/confusion_matrix.png")
        plt.close()


if __name__ == "__main__":
    # Create trainer and train
    trainer = Trainer()
    trainer.train()
    trainer.evaluate()
