"""
Training utilities for lung nodule detection models.
"""
import os
import numpy as np
from typing import List, Optional, Union, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    Dataset = None
    DataLoader = None


class CTScanDataset(Dataset):
    """Dataset for CT scan slices."""
    
    def __init__(self, images: List[np.ndarray], labels: Optional[List[float]] = None):
        """
        Args:
            images: List of image arrays
            labels: Optional list of labels (for supervised learning)
        """
        self.images = images
        self.labels = labels
        self.has_labels = labels is not None
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        
        # Ensure 2D
        if len(img.shape) == 3:
            img = img[0]  # Take first channel if 3D
        
        # Normalize to [0, 1] if needed
        if img.max() > 1.0:
            img = img / 255.0
        
        # Convert to tensor
        if TORCH_AVAILABLE:
            img_tensor = torch.FloatTensor(img).unsqueeze(0)  # Add channel dimension
            
            if self.has_labels:
                label = self.labels[idx]
                return img_tensor, torch.FloatTensor([label])
            else:
                return img_tensor
        else:
            return img


class ImprovedCNN(nn.Module):
    """Improved CNN for binary classification with batch normalization and dropout."""
    
    def __init__(self, input_size: int = 256):
        super(ImprovedCNN, self).__init__()
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Use adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(256 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # First block
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Third block
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Fourth block
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        # Adaptive pooling and flatten
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        
        return x


class SimpleCNN(nn.Module):
    """Simple CNN for binary classification (kept for backward compatibility)."""
    
    def __init__(self, input_size: int = 256):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * (input_size // 8) * (input_size // 8), 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def calculate_metrics(outputs, targets, threshold=0.5):
    """
    Calculate accuracy, precision, recall, and F1 score.
    
    Args:
        outputs: Model predictions (probabilities)
        targets: Ground truth labels
        threshold: Classification threshold
    
    Returns:
        Dictionary with metrics
    """
    if not TORCH_AVAILABLE:
        return {}
    
    # Convert to binary predictions
    predictions = (outputs >= threshold).float()
    targets = targets.float()
    
    # Calculate metrics
    correct = (predictions == targets).float()
    accuracy = correct.mean().item()
    
    # True positives, false positives, false negatives
    tp = ((predictions == 1) & (targets == 1)).float().sum().item()
    fp = ((predictions == 1) & (targets == 0)).float().sum().item()
    fn = ((predictions == 0) & (targets == 1)).float().sum().item()
    tn = ((predictions == 0) & (targets == 0)).float().sum().item()
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def plot_training_history(history: dict, model_save_path: Optional[str] = None):
    """
    Plot training history including loss and accuracy curves.
    
    Args:
        history: Dictionary with training history
        model_save_path: Optional path to save plots near model file
    """
    try:
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        if history['val_loss']:
            axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        axes[0, 1].plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        if history['val_accuracy']:
            axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        # Plot 3: Precision and Recall
        axes[1, 0].plot(epochs, history['train_precision'], 'g-', label='Training Precision', linewidth=2)
        axes[1, 0].plot(epochs, history['train_recall'], 'orange', label='Training Recall', linewidth=2)
        if history['val_precision']:
            axes[1, 0].plot(epochs, history['val_precision'], 'g--', label='Validation Precision', linewidth=2)
            axes[1, 0].plot(epochs, history['val_recall'], 'orange', linestyle='--', label='Validation Recall', linewidth=2)
        axes[1, 0].set_title('Precision and Recall', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        # Plot 4: F1 Score
        axes[1, 1].plot(epochs, history['train_f1'], 'b-', label='Training F1', linewidth=2)
        if history['val_f1']:
            axes[1, 1].plot(epochs, history['val_f1'], 'r-', label='Validation F1', linewidth=2)
        axes[1, 1].set_title('F1 Score', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save plot
        if model_save_path:
            plot_path = model_save_path.replace('.pth', '_training_history.png')
        else:
            plot_path = 'training_history.png'
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to: {plot_path}")
        plt.close()
        
    except ImportError:
        print("Matplotlib not available. Skipping plot generation.")
    except Exception as e:
        print(f"Error generating plots: {e}")


def train(
    images: Union[List[np.ndarray], np.ndarray],
    labels: Optional[List[float]] = None,
    val_images: Optional[List[np.ndarray]] = None,
    val_labels: Optional[List[float]] = None,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    model_save_path: Optional[str] = None,
    use_improved_model: bool = True,
    plot_history: bool = True
) -> Tuple[Optional[nn.Module], dict]:
    """
    Train a model on CT scan images with accuracy tracking and visualization.
    
    Args:
        images: List of image arrays or single array
        labels: Optional labels for supervised learning
        val_images: Optional validation images
        val_labels: Optional validation labels
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        model_save_path: Path to save trained model
        use_improved_model: Whether to use ImprovedCNN (default) or SimpleCNN
        plot_history: Whether to generate and save training plots
    
    Returns:
        Tuple of (trained model, training history dictionary)
    """
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping training.")
        return None, {}
    
    # Convert to list if needed
    if isinstance(images, np.ndarray):
        if len(images.shape) == 3:
            images = [images[i] for i in range(images.shape[0])]
        else:
            images = [images]
    
    # Create dataset
    dataset = CTScanDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    if use_improved_model:
        model = ImprovedCNN(input_size=256)
        print("Using ImprovedCNN with batch normalization and dropout")
    else:
        model = SimpleCNN(input_size=256)
        print("Using SimpleCNN")
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    except TypeError:
        # Fallback for older PyTorch versions
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Validation setup
    val_dataloader = None
    if val_images is not None and val_labels is not None:
        val_dataset = CTScanDataset(val_images, val_labels)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # History tracking
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    print(f"Training on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_accuracy = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        
        for batch_idx, batch in enumerate(dataloader):
            if labels is not None:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
            else:
                inputs = batch.to(device) if isinstance(batch, torch.Tensor) else batch
                targets = None
            
            optimizer.zero_grad()
            
            if targets is not None:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                # Collect predictions for metrics
                all_outputs.append(outputs.detach().cpu())
                all_targets.append(targets.detach().cpu())
            else:
                print("Warning: Unsupervised training not fully implemented. Skipping batch.")
                continue
        
        # Calculate training metrics
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        train_metrics = {}
        if all_outputs and all_targets:
            train_outputs = torch.cat(all_outputs)
            train_targets = torch.cat(all_targets)
            train_metrics = calculate_metrics(train_outputs, train_targets)
        
        history['train_loss'].append(avg_loss)
        history['train_accuracy'].append(train_metrics.get('accuracy', 0.0))
        history['train_precision'].append(train_metrics.get('precision', 0.0))
        history['train_recall'].append(train_metrics.get('recall', 0.0))
        history['train_f1'].append(train_metrics.get('f1', 0.0))
        
        # Validation phase
        val_metrics = {}
        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            val_outputs = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_dataloader:
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    val_outputs.append(outputs.cpu())
                    val_targets.append(targets.cpu())
            
            avg_val_loss = val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0.0
            
            if val_outputs and val_targets:
                val_outputs_tensor = torch.cat(val_outputs)
                val_targets_tensor = torch.cat(val_targets)
                val_metrics = calculate_metrics(val_outputs_tensor, val_targets_tensor)
            
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_metrics.get('accuracy', 0.0))
            history['val_precision'].append(val_metrics.get('precision', 0.0))
            history['val_recall'].append(val_metrics.get('recall', 0.0))
            history['val_f1'].append(val_metrics.get('f1', 0.0))
            
            # Update learning rate scheduler
            scheduler.step(avg_val_loss)
            
            # Save best model
            val_acc = val_metrics.get('accuracy', 0.0)
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_state = model.state_dict().copy()
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {avg_loss:.4f} | Acc: {train_metrics.get('accuracy', 0.0):.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Val F1: {val_metrics.get('f1', 0.0):.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {avg_loss:.4f} | Acc: {train_metrics.get('accuracy', 0.0):.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")
    
    # Save model
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.', exist_ok=True)
        torch.save(model, model_save_path)
        print(f"Model saved to {model_save_path}")
    
    # Generate plots
    if plot_history and val_dataloader is not None:
        try:
            plot_training_history(history, model_save_path)
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
    
    return model, history
