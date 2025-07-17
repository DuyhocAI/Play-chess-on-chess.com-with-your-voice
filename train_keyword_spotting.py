import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchaudio.functional import pitch_shift
from tqdm import tqdm
import random
import torch.multiprocessing as mp
from pathlib import Path
import gc
import psutil
import time
import numpy as np

# Set multiprocessing start method for CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

DATA_DIR     = ".\data_edge_tts_concurrent_fixed"
BATCH_SIZE   = 16  # Reduced from 32 to prevent memory overload
SR           = 16000
N_MELS       = 64
LR           = 2e-3
EPOCHS       = 15
LABELS       = [f"{c}{n}" for c in "abcdefgh" for n in range(1,9)]  # a1...h8
NUM_CLASSES  = len(LABELS)
OUTPUT_PLOTS = "plots_complex"
os.makedirs(OUTPUT_PLOTS, exist_ok=True)

# CPU monitoring configuration
MAX_CPU_USAGE = 80  # Maximum CPU usage percentage before throttling
MAX_MEMORY_USAGE = 85  # Maximum memory usage percentage

# GPU Setup with better memory management
def setup_device():
    if not torch.cuda.is_available():
        print("CUDA không khả dụng, sử dụng CPU")
        return torch.device("cpu"), 1, BATCH_SIZE // 2  # Smaller batch for CPU
    
    num_gpus = torch.cuda.device_count()
    print(f"Phát hiện {num_gpus} GPU(s)")
    
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # Optimize CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    
    device = torch.device("cuda")
    # Conservative batch size scaling
    effective_batch_size = BATCH_SIZE * min(2, max(1, num_gpus))  # Cap at 2x scaling
    
    return device, num_gpus, effective_batch_size

DEVICE, NUM_GPUS, EFFECTIVE_BATCH_SIZE = setup_device()
print(f"Sử dụng device: {DEVICE}")
print(f"Effective batch size: {EFFECTIVE_BATCH_SIZE}")

def monitor_system_resources():
    """Monitor CPU and memory usage"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_percent = psutil.virtual_memory().percent
    
    if cpu_percent > MAX_CPU_USAGE:
        print(f"High CPU usage: {cpu_percent:.1f}% - Adding delay...")
        time.sleep(0.5)
    
    if memory_percent > MAX_MEMORY_USAGE:
        print(f"High memory usage: {memory_percent:.1f}% - Running garbage collection...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return cpu_percent, memory_percent

# --- MAPPING LABELS → IDX ---
label2idx = {lab: i for i, lab in enumerate(LABELS)}

# --- OPTIMIZED AUGMENTATION ---
class AudioAugment:
    def __init__(self):
        self.freq_mask = FrequencyMasking(freq_mask_param=10)  # Reduced from 15
        self.time_mask = TimeMasking(time_mask_param=20)       # Reduced from 30

    def __call__(self, spec):
        # Reduce augmentation probability to save CPU
        if random.random() < 0.3:  # Reduced from 0.5
            spec = self.freq_mask(spec)
        if random.random() < 0.3:  # Reduced from 0.5
            spec = self.time_mask(spec)
        return spec

# --- MEMORY-EFFICIENT DATASET ---
class KeywordDataset(Dataset):
    def __init__(self, data_dir, augment=False, max_samples_per_class=None):
        self.samples = []
        self.data_dir = Path(data_dir)
        self.max_samples_per_class = max_samples_per_class
        
        # Check if data directory exists
        if not self.data_dir.exists():
            print(f"ERROR: Data directory does not exist: {data_dir}")
            print(f"Please check the path and make sure it exists.")
            return
        
        print(f"Scanning directory: {self.data_dir}")
        print(f"Looking for subdirectories with labels: {LABELS}")
        
        # Scan for samples with detailed logging
        found_folders = []
        for lab in LABELS:
            folder = self.data_dir / lab
            if folder.exists() and folder.is_dir():
                found_folders.append(lab)
                wav_files = list(folder.glob('*.wav'))
                
                # Limit samples per class to prevent memory overflow
                if self.max_samples_per_class and len(wav_files) > self.max_samples_per_class:
                    wav_files = random.sample(wav_files, self.max_samples_per_class)
                    print(f"  {lab}: limited to {len(wav_files)} .wav files (from {len(list(folder.glob('*.wav')))})")
                else:
                    print(f"  {lab}: found {len(wav_files)} .wav files")
                
                for wav_file in wav_files:
                    self.samples.append((str(wav_file), label2idx[lab]))
            else:
                print(f"  {lab}: folder not found or not a directory")
        
        print(f"\nSummary:")
        print(f"  Found folders: {found_folders}")
        print(f"  Total samples: {len(self.samples)}")
        
        # Initialize transforms with optimized settings
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            n_mels=N_MELS,
            n_fft=512,  # Smaller FFT for efficiency
            hop_length=160,
            f_min=20,
            f_max=SR//2
        )
        
        self.augment = augment
        if augment:
            self.aug = AudioAugment()
        self.sr = SR

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label = self.samples[idx]
        
        try:
            # Load audio with memory optimization
            wav, original_sr = torchaudio.load(wav_path)
            
            # Early memory cleanup
            if original_sr != self.sr:
                wav = torchaudio.functional.resample(wav, original_sr, self.sr)
            
            wav = wav.mean(dim=0, keepdim=True)  # Convert to mono
            
            # Fixed length processing
            target_length = SR  # 1 second
            if wav.shape[-1] < target_length:
                wav = F.pad(wav, (0, target_length - wav.shape[-1]), 'constant', 0.0)
            else:
                wav = wav[:, :target_length]
            
            # Lighter augmentations
            if self.augment:
                # Reduced probability and intensity
                if random.random() < 0.2:  # Reduced from 0.3
                    n_steps = random.uniform(-1.0, 1.0)  # Reduced from -2.0, 2.0
                    wav = pitch_shift(wav, self.sr, n_steps)
                
                if random.random() < 0.2:  # Reduced from 0.3
                    wav = wav + 0.003 * torch.randn_like(wav)  # Reduced noise
            
            # Convert to spectrogram
            spec = self.mel_transform(wav)
            spec = torch.log1p(spec)
            
            # Spec augmentation
            if self.augment:
                spec = self.aug(spec)
            
            # Normalize efficiently
            mean_val = spec.mean()
            std_val = spec.std()
            spec = (spec - mean_val) / (std_val + 1e-6)
            
            return spec, label
            
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            return torch.zeros(1, N_MELS, SR//160), label

# --- OPTIMIZED COLLATE FUNCTION ---
def collate_fn(batch):
    specs, labels = zip(*batch)
    
    # Find max length efficiently
    max_len = max(s.shape[-1] for s in specs)
    padded_specs = []
    
    for s in specs:
        if s.shape[-1] < max_len:
            padded = F.pad(s, (0, max_len - s.shape[-1]))
        else:
            padded = s
        padded_specs.append(padded)
    
    # Stack tensors
    specs_tensor = torch.stack(padded_specs)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return specs_tensor, labels_tensor

# --- LIGHTWEIGHT RESNET MODEL ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.05):  # Reduced dropout
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out, inplace=True)
        
        return out

class ResNetAudio(nn.Module):
    def __init__(self, num_classes, dropout=0.3):  # Reduced dropout
        super().__init__()
        
        # Smaller initial convolution
        self.conv1 = nn.Conv2d(1, 32, 5, 2, 2, bias=False)  # Reduced from 64 channels, 7x7 kernel
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Smaller residual blocks
        self.layer1 = self._make_layer(32, 32, 2, stride=1)    # Reduced channels
        self.layer2 = self._make_layer(32, 64, 2, stride=2)    # Reduced channels
        self.layer3 = self._make_layer(64, 128, 2, stride=2)   # Reduced channels
        self.layer4 = self._make_layer(128, 256, 2, stride=2)  # Reduced channels
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)  # Adjusted for smaller model
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# --- OPTIMIZED DATA LOADERS ---
def create_data_loaders():
    print("Creating dataset...")
    
    # Limit samples to prevent memory overflow
    max_samples = 1000  # Limit total samples per class
    full_dataset = KeywordDataset(DATA_DIR, augment=True, max_samples_per_class=max_samples)
    
    if len(full_dataset) == 0:
        print("\nERROR: No samples found in dataset!")
        return None, None
    
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(full_dataset)}")
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    print(f"\nSplit:")
    print(f"  Train samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create validation dataset without augmentation
    val_dataset_clean = KeywordDataset(DATA_DIR, augment=False, max_samples_per_class=max_samples)
    _, val_dataset_clean = torch.utils.data.random_split(
        val_dataset_clean, [train_size, val_size]
    )
    
    # Optimized data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=EFFECTIVE_BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # Reduced from 0 to allow some parallelism but not overload
        pin_memory=DEVICE.type == 'cuda',
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=2  # Reduce prefetching
    )
    
    val_loader = DataLoader(
        val_dataset_clean,
        batch_size=EFFECTIVE_BATCH_SIZE,
        shuffle=False,
        num_workers=1,  # Minimal workers for validation
        pin_memory=DEVICE.type == 'cuda',
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=True
    )
    
    return train_loader, val_loader

# --- OPTIMIZED TRAINING FUNCTIONS ---
def train_epoch(model, train_loader, optimizer, criterion, scaler, scheduler, device):
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc='Training')
    
    for batch_idx, (specs, labels) in enumerate(progress_bar):
        # Monitor system resources periodically
        if batch_idx % 20 == 0:
            cpu_usage, mem_usage = monitor_system_resources()
        
        # Move to device
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(specs)
                loss = criterion(outputs, labels)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Track loss
        total_loss += loss.item()
        
        # Update progress bar less frequently
        if batch_idx % 10 == 0:
            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{current_lr:.6f}',
                'CPU': f'{psutil.cpu_percent():.1f}%'
            })
        
        # Periodic memory cleanup
        if batch_idx % 50 == 0:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
    
    return total_loss / num_batches

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        
        for batch_idx, (specs, labels) in enumerate(progress_bar):
            specs = specs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(specs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(specs)
                loss = criterion(outputs, labels)
            
            # Calculate accuracy
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            total_loss += loss.item()
            
            # Update progress bar less frequently
            if batch_idx % 5 == 0:
                current_acc = correct / total if total > 0 else 0
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

# --- MAIN TRAINING LOOP WITH RESOURCE MONITORING ---
def main():
    print("=== OPTIMIZED AUDIO CLASSIFICATION TRAINING ===")
    print(f"Device: {DEVICE}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Batch size: {EFFECTIVE_BATCH_SIZE}")
    
    # System info
    print(f"\nSystem Info:")
    print(f"  CPU cores: {psutil.cpu_count()}")
    print(f"  Memory: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    print(f"  Available memory: {psutil.virtual_memory().available / 1024**3:.1f}GB")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders()
    
    if train_loader is None or val_loader is None:
        print("Exiting due to data loading issues.")
        return None, None
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating lightweight model...")
    model = ResNetAudio(NUM_CLASSES)
    model = model.to(DEVICE)
    
    # Multi-GPU support (conservative)
    if NUM_GPUS > 1 and torch.cuda.device_count() > 1:
        print(f"Using {min(2, NUM_GPUS)} GPUs with DataParallel")  # Limit to 2 GPUs max
        model = nn.DataParallel(model, device_ids=list(range(min(2, NUM_GPUS))))
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Conservative optimizer settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Reduced label smoothing
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': [],
        'cpu_usage': [],
        'memory_usage': []
    }
    
    best_acc = 0.0
    best_model_path = 'spotting_word_best_model_optimized.pth'
    
    print("\n=== STARTING OPTIMIZED TRAINING ===")
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 50)
        
        # Pre-epoch resource check
        cpu_usage, mem_usage = monitor_system_resources()
        
        # Training
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scaler, scheduler, DEVICE
        )
        
        # Validation
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, DEVICE)
        
        # Post-epoch cleanup
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]
        
        # Monitor resources
        cpu_usage, mem_usage = monitor_system_resources()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)
        history['cpu_usage'].append(cpu_usage)
        history['memory_usage'].append(mem_usage)
        
        # Print results
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"CPU Usage: {cpu_usage:.1f}%")
        print(f"Memory Usage: {mem_usage:.1f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'history': history
            }, best_model_path)
            
            print(f"✓ New best model saved! Accuracy: {best_acc:.4f}")
        
        # Emergency break if resources are too high
        if cpu_usage > 95 or mem_usage > 95:
            print("EMERGENCY STOP: System resources critically high!")
            print("Saving current state and exiting...")
            break
    
    print(f"\n=== TRAINING COMPLETED ===")
    print(f"Best validation accuracy: {best_acc:.4f}")
    
    # Save training plots
    save_training_plots(history)
    
    return model, history

def save_training_plots(history):
    """Save enhanced training plots with resource monitoring"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['val_acc'], 'g-', label='Val Accuracy', linewidth=2)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Resource usage plot
    if 'cpu_usage' in history and history['cpu_usage']:
        ax3.plot(epochs, history['cpu_usage'], 'orange', label='CPU Usage (%)', linewidth=2)
        ax3.plot(epochs, history['memory_usage'], 'purple', label='Memory Usage (%)', linewidth=2)
        ax3.axhline(y=MAX_CPU_USAGE, color='red', linestyle='--', alpha=0.7, label='CPU Limit')
        ax3.axhline(y=MAX_MEMORY_USAGE, color='red', linestyle='--', alpha=0.7, label='Memory Limit')
        ax3.set_title('System Resource Usage')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Usage (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        # Learning rate plot as fallback
        ax3.plot(epochs, history['learning_rate'], 'orange', label='Learning Rate', linewidth=2)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    
    # Summary text
    max_acc = max(history['val_acc']) if history['val_acc'] else 0
    min_loss = min(history['val_loss']) if history['val_loss'] else 0

    # Calculate model parameters safely
    try:
        actual_model = model.module if hasattr(model, 'module') else model
        total_params = sum(p.numel() for p in actual_model.parameters())
        param_text = f'{total_params:,}'
    except:
        param_text = 'N/A'

    ax4.text(0.1, 0.9, f'Max Validation Accuracy: {max_acc:.4f}', fontsize=12, weight='bold')
    ax4.text(0.1, 0.8, f'Min Validation Loss: {min_loss:.4f}', fontsize=12)
    ax4.text(0.1, 0.7, f'Total Epochs: {len(epochs)}', fontsize=12)
    ax4.text(0.1, 0.6, f'Device: {DEVICE}', fontsize=12)
    ax4.text(0.1, 0.5, f'GPUs Used: {NUM_GPUS}', fontsize=12)
    ax4.text(0.1, 0.4, f'Batch Size: {EFFECTIVE_BATCH_SIZE}', fontsize=12)
    ax4.text(0.1, 0.3, f'Model Parameters: {param_text}', fontsize=10)
    ax4.text(0.1, 0.2, 'Optimized for Resource Efficiency', fontsize=12, color='green', weight='bold')
    ax4.set_title('Training Summary')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS, 'optimized_training_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training plots saved to {OUTPUT_PLOTS}/optimized_training_results.png")

# --- UTILITY FUNCTION TO CHECK DATA STRUCTURE ---
def check_data_structure(data_dir):
    """Utility function to check and display the data directory structure"""
    data_path = Path(data_dir)
    
    print(f"=== DATA STRUCTURE CHECK ===")
    print(f"Checking directory: {data_path.absolute()}")
    
    if not data_path.exists():
        print(f"ERROR: Directory does not exist!")
        print(f"Expected path: {data_path.absolute()}")
        return False
    
    print(f"Directory exists")
    
    # Check for expected subdirectories
    found_labels = []
    missing_labels = []
    total_files = 0
    
    print(f"\nChecking for chess position subdirectories:")
    for label in LABELS:
        label_dir = data_path / label
        if label_dir.exists() and label_dir.is_dir():
            wav_files = list(label_dir.glob('*.wav'))
            found_labels.append(label)
            total_files += len(wav_files)
            print(f"{label}: {len(wav_files)} .wav files")
        else:
            missing_labels.append(label)
            print(f"{label}: directory not found")
    
    print(f"\nSummary:")
    print(f"  Found directories: {len(found_labels)}/{len(LABELS)}")
    print(f"  Missing directories: {missing_labels}")
    print(f"  Total audio files: {total_files}")
    
    if len(found_labels) == 0:
        print(f"\nCRITICAL: No valid subdirectories found!")
        print(f"Please ensure your data directory contains subdirectories named: {LABELS[:10]}...")
        return False
    elif len(found_labels) < len(LABELS):
        print(f"\n WARNING: Some directories are missing, but training can continue with {len(found_labels)} classes")
    else:
        print(f"\nAll directories found! Ready for training.")
    
    return True

def analyze_sample_distribution(data_dir):
    """Analyze the distribution of samples across classes"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return
    
    print(f"\n=== SAMPLE DISTRIBUTION ANALYSIS ===")
    
    class_counts = {}
    for label in LABELS:
        label_dir = data_path / label
        if label_dir.exists():
            wav_files = list(label_dir.glob('*.wav'))
            class_counts[label] = len(wav_files)
    
    if not class_counts:
        print("No samples found!")
        return
    
    # Statistics
    counts = list(class_counts.values())
    total_samples = sum(counts)
    mean_samples = np.mean(counts)
    std_samples = np.std(counts)
    min_samples = min(counts)
    max_samples = max(counts)
    
    print(f"Total samples: {total_samples}")
    print(f"Average samples per class: {mean_samples:.1f}")
    print(f"Standard deviation: {std_samples:.1f}")
    print(f"Min samples in a class: {min_samples}")
    print(f"Max samples in a class: {max_samples}")
    
    # Show classes with significantly different sample counts
    threshold = mean_samples * 0.5  # Classes with less than 50% of average
    low_sample_classes = [label for label, count in class_counts.items() if count < threshold]
    
    if low_sample_classes:
        print(f"\n⚠️  Classes with low sample counts (< {threshold:.0f}):")
        for label in low_sample_classes:
            print(f"  {label}: {class_counts[label]} samples")
    
    return class_counts

def test_data_loading():
    """Test data loading with a small sample"""
    print(f"\n=== TESTING DATA LOADING ===")
    
    try:
        # Create small test dataset
        test_dataset = KeywordDataset(DATA_DIR, augment=False, max_samples_per_class=5)
        
        if len(test_dataset) == 0:
            print("No samples loaded!")
            return False
        
        print(f"Successfully created dataset with {len(test_dataset)} samples")
        
        # Test data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # No multiprocessing for test
            collate_fn=collate_fn
        )
        
        # Load one batch
        for batch_idx, (specs, labels) in enumerate(test_loader):
            print(f" Successfully loaded batch {batch_idx + 1}")
            print(f"   Spec shape: {specs.shape}")
            print(f"   Labels shape: {labels.shape}")
            print(f"   Labels: {labels.tolist()}")
            
            if batch_idx >= 2:  # Test first 3 batches
                break
        
        print(" Data loading test passed!")
        return True
        
    except Exception as e:
        print(f" Data loading test failed: {e}")
        return False

def quick_model_test():
    """Quick test of model forward pass"""
    print(f"\n=== TESTING MODEL ===")
    
    try:
        # Create model
        model = ResNetAudio(NUM_CLASSES)
        model = model.to(DEVICE)
        
        # Create dummy input
        dummy_input = torch.randn(2, 1, N_MELS, 100).to(DEVICE)  # Batch size 2
        
        print(f" Model created successfully")
        print(f"   Input shape: {dummy_input.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f" Forward pass successful")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected shape: ({dummy_input.shape[0]}, {NUM_CLASSES})")
        
        # Test loss calculation
        dummy_labels = torch.randint(0, NUM_CLASSES, (2,)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, dummy_labels)
        
        print(f" Loss calculation successful: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f" Model test failed: {e}")
        return False

def estimate_memory_requirements():
    """Estimate memory requirements for training"""
    print(f"\n=== MEMORY ESTIMATION ===")
    
    # Model parameters
    model = ResNetAudio(NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters())
    model_memory_mb = total_params * 4 / (1024**2)  # 4 bytes per float32
    
    # Batch memory (approximate)
    sample_size = N_MELS * SR // 160  # Approximate spectrogram size
    batch_memory_mb = EFFECTIVE_BATCH_SIZE * sample_size * 4 / (1024**2)
    
    # Total estimated memory
    optimizer_memory_mb = model_memory_mb * 2  # Optimizer states
    total_estimated_mb = model_memory_mb + batch_memory_mb + optimizer_memory_mb + 500  # 500MB buffer
    
    print(f"Model parameters: {total_params:,}")
    print(f"Model memory: {model_memory_mb:.1f} MB")
    print(f"Batch memory: {batch_memory_mb:.1f} MB")
    print(f"Optimizer memory: {optimizer_memory_mb:.1f} MB")
    print(f"Total estimated: {total_estimated_mb:.1f} MB ({total_estimated_mb/1024:.1f} GB)")
    
    # System memory
    system_memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    print(f"\nSystem memory: {system_memory_gb:.1f} GB")
    print(f"Available memory: {available_memory_gb:.1f} GB")
    
    if total_estimated_mb / 1024 > available_memory_gb * 0.8:
        print(f"  WARNING: Estimated memory usage is high!")
        print(f"Consider reducing batch size or model size.")
    else:
        print(f" Memory requirements look reasonable.")

def run_diagnostics():
    """Run comprehensive diagnostics before training"""
    print("="*60)
    print(" RUNNING PRE-TRAINING DIAGNOSTICS")
    print("="*60)
    
    # System info
    print(f"\n SYSTEM INFORMATION:")
    print(f"   Python version: {os.sys.version}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Device: {DEVICE}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check data structure
    data_ok = check_data_structure(DATA_DIR)
    if not data_ok:
        return False
    
    # Analyze sample distribution
    analyze_sample_distribution(DATA_DIR)
    
    # Estimate memory
    estimate_memory_requirements()
    
    # Test data loading
    data_loading_ok = test_data_loading()
    if not data_loading_ok:
        return False
    
    # Test model
    model_ok = quick_model_test()
    if not model_ok:
        return False
    
    print("\n" + "="*60)
    print(" ALL DIAGNOSTICS PASSED! Ready to start training.")
    print("="*60)
    
    return True

if __name__ == "__main__":
    print(" OPTIMIZED CHESS AUDIO CLASSIFICATION TRAINING")
    print("="*60)
    
    # Run diagnostics first
    if not run_diagnostics():
        print("\n DIAGNOSTICS FAILED! Please fix the issues above before training.")
        exit(1)
    
    # Ask user for confirmation
    print("\n" + " READY TO START TRAINING!")
    print(f"Configuration:")
    print(f"  • Data directory: {DATA_DIR}")
    print(f"  • Device: {DEVICE}")
    print(f"  • Batch size: {EFFECTIVE_BATCH_SIZE}")
    print(f"  • Epochs: {EPOCHS}")
    print(f"  • Learning rate: {LR}")
    print(f"  • Classes: {NUM_CLASSES}")
    
    response = input("\nDo you want to proceed with training? (y/n): ").lower().strip()
    
    if response == 'y' or response == 'yes':
        print("\n Starting training...")
        
        try:
            model, history = main()
            
            if model is not None and history is not None:
                print("\n" + "="*60)
                print(" TRAINING COMPLETED SUCCESSFULLY!")
                print("="*60)
                print(f" Best validation accuracy: {max(history['val_acc']):.4f}")
                print(f" Model saved to: best_model_optimized.pth")
                print(f" Plots saved to: {OUTPUT_PLOTS}/")
            else:
                print("\n Training failed or was interrupted.")
                
        except KeyboardInterrupt:
            print("\n\n  Training interrupted by user.")
            print("Progress has been saved where possible.")
        except Exception as e:
            print(f"\n Training failed with error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n Training cancelled by user.")
    
    print("\nThank you for using the optimized chess audio classification trainer!")