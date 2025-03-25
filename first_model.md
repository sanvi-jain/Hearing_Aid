import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader


# Define the CNN model
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 16 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.fc_layers(x)
        return x


class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        for label, class_name in enumerate(["cat", "dog"]):
            class_path = os.path.join(root_dir, class_name)
            for file in os.listdir(class_path):
                if file.endswith(".wav"):
                    self.data.append((os.path.join(class_path, file), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        spectrogram = self.load_audio(file_path)

        return spectrogram, torch.tensor(label, dtype=torch.float32)

    def load_audio(self, file_path):
        waveform, sr = torchaudio.load(file_path)
        spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sr)(waveform)
        spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)

        # Pad or trim the spectrogram to a fixed length
        desired_length = 256  # Choose a suitable length

        # Use torch.nn.functional.pad instead of torchaudio.functional.pad
        # Calculate padding needed for each dimension
        pad_left = (desired_length - spectrogram.shape[-1]) // 2
        pad_right = desired_length - spectrogram.shape[-1] - pad_left

        # Apply padding using torch.nn.functional.pad
        spectrogram = torch.nn.functional.pad(spectrogram, (pad_left, pad_right))

        return spectrogram

# Hyperparameters
batch_size = 32
learning_rate = 0.0005
num_epochs = 15

dataset = AudioDataset("data/train")  # Update with correct path
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioCNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    for spectrograms, labels in dataloader:
        spectrograms, labels = spectrograms.to(device), labels.view(-1, 1).to(device)

        outputs = model(spectrograms)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")

# Testing the model
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for spectrograms, labels in dataloader:
        spectrograms, labels = spectrograms.to(device), labels.view(-1, 1).to(device)
        outputs = model(spectrograms)
        predictions = (outputs >= 0.5).float()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

