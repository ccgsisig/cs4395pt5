# Sentiment Analysis using PyTorch Transformer from Scratch
# ---------------------------------------------------------------
# Instructions:
# - Fill in the code where you see "Your code here".
# - Follow the instructions and comments provided to complete each part.
# - Do not change the structure of the code unless necessary.
# - Ensure that your code is well-documented and readable.

# Import necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from collections import Counter

# Ensure reproducibility
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Download NLTK data (if not already downloaded)
nltk.download('stopwords')

# ------------------------------------------------------
# Part 1: Data Loading and Preprocessing
# ------------------------------------------------------

# Step 1: Load the Dataset
try:
    # Load datasets using pandas.read_csv
    df1 = pd.read_csv('amazon_cells_labelled.txt', delimiter='\t', header=None, names=['sentence', 'label'])
    df2 = pd.read_csv('imdb_labelled.txt', delimiter='\t', header=None, names=['sentence', 'label'])
    df3 = pd.read_csv('yelp_labelled.txt', delimiter='\t', header=None, names=['sentence', 'label'])
except FileNotFoundError:
    print("Dataset files not found. Please ensure the dataset files are in the current directory.")
    exit()

# Step 2: Combine and Shuffle the Dataset
# Combine the datasets
df = pd.concat([df1, df2, df3], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 3: Data Cleaning and Preprocessing
# Initialize stop words
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Split text into tokens
    tokens = text.split()
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Apply preprocessing
df['tokens'] = df['sentence'].apply(preprocess_text)

# Step 4: Build Vocabulary and Tokenize Text
# Build vocabulary
vocab_size = 10000  # Set vocabulary size
all_tokens = [token for tokens in df['tokens'] for token in tokens]
most_common_tokens = Counter(all_tokens).most_common(vocab_size - 2)  # Reserve 2 indices for special tokens

# Create mappings
word_to_idx = {word: idx + 2 for idx, (word, _) in enumerate(most_common_tokens)}
word_to_idx['<PAD>'] = 0
word_to_idx['<UNK>'] = 1
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

def tokens_to_indices(tokens):
    # Convert tokens to indices, use '<UNK>' for unknown words
    indices = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]
    return indices

# Convert tokens to indices
df['indices'] = df['tokens'].apply(tokens_to_indices)

# Step 5: Pad Sequences
max_seq_length = 50  # Set maximum sequence length

def pad_sequence(seq, max_len):
    # Pad or truncate sequences
    if len(seq) < max_len:
        seq = seq + [word_to_idx['<PAD>']] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

# Apply padding
df['padded_indices'] = df['indices'].apply(lambda x: pad_sequence(x, max_seq_length))

# Prepare feature and label arrays
X = np.array(df['padded_indices'].tolist())
y = df['label'].values

# Step 6: Split the Dataset Manually
# Shuffle indices
indices = np.arange(len(df))
np.random.shuffle(indices)

# Calculate split index
split_ratio = 0.8
split_index = int(len(df) * split_ratio)

# Split data
train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

# ------------------------------------------------------
# Part 2: Transformer Model Implementation
# ------------------------------------------------------

# Instructions:
# - Implement the Transformer model from scratch using PyTorch.
# - Define classes for Positional Encoding, Multi-Head Self-Attention, Position-wise Feedforward Network, Transformer Encoder Layer, and the Transformer Classifier.
# - Use the provided method signatures and fill in the code as per the instructions.

import torch.nn.functional as F

# Define Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Instructions:
        # - Initialize a positional encoding matrix 'pe' of shape [max_len, embedding_dim].
        # - Compute the positional encodings using sine and cosine functions.
        # - Register 'pe' as a buffer.

        # Your code here
        pass

    def forward(self, x):
        # Instructions:
        # - Add positional encoding to the input embeddings 'x'.
        # - Ensure the positional encoding is added correctly considering the sequence length.

        # Your code here
        pass

# Multi-Head Self-Attention Layer
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        # Instructions:
        # - Ensure 'embedding_dim' is divisible by 'num_heads'.
        # - Define linear layers for query, key, and value projections.
        # - Define a linear layer for the output.

        # Your code here
        pass

    def forward(self, x):
        # Instructions:
        # - Compute queries, keys, and values using the linear layers.
        # - Split the embeddings into multiple heads.
        # - Compute scaled dot-product attention for each head.
        # - Concatenate the attention outputs from all heads.
        # - Apply the final linear layer.

        # Your code here
        pass

# Position-wise Feedforward Network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(PositionwiseFeedForward, self).__init__()
        # Instructions:
        # - Define two linear layers with a ReLU activation in between.

        # Your code here
        pass

    def forward(self, x):
        # Instructions:
        # - Apply the first linear layer, followed by ReLU activation.
        # - Apply the second linear layer.

        # Your code here
        pass

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim):
        super(TransformerEncoderLayer, self).__init__()
        # Instructions:
        # - Initialize the multi-head self-attention layer.
        # - Initialize the position-wise feedforward network.
        # - Define layer normalization and dropout layers as needed.

        # Your code here

    def forward(self, x):
        # Instructions:
        # - Apply self-attention with residual connection and layer normalization.
        # - Apply feedforward network with residual connection and layer normalization.

        # Your code here
        pass

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, num_layers):
        super(TransformerEncoder, self).__init__()
        # Instructions:
        # - Create a list of TransformerEncoderLayer instances use nn.ModuleList().

        # Your code here

    def forward(self, x):
        # Instructions:
        # - Pass the input through each layer in the encoder.

        # Your code here
        pass

# Define the Transformer-based classifier
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_heads=8, num_layers=2, num_classes=1, hidden_dim=512, max_seq_length=50):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=word_to_idx['<PAD>'])
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len=max_seq_length)
        self.transformer_encoder = TransformerEncoder(embedding_dim, num_heads, hidden_dim, num_layers)
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.embedding_dim = embedding_dim

    def forward(self, src):
        embedded = self.embedding(src) * np.sqrt(self.embedding_dim)
        embedded = self.pos_encoder(embedded)
        transformer_output = self.transformer_encoder(embedded)
        pooled_output = transformer_output.mean(dim=1)
        output = self.fc(pooled_output)
        output = self.sigmoid(output)
        return output

# Instantiate the model
vocab_size = len(word_to_idx)
model = TransformerClassifier(vocab_size)

# ------------------------------------------------------
# Part 3: Training Loop
# ------------------------------------------------------

# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).long()
y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
X_test_tensor = torch.from_numpy(X_test).long()
y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop parameters
epochs = 10  # Adjust as needed
batch_size = 32
num_batches = int(np.ceil(len(X_train_tensor) / batch_size))

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for i in range(num_batches):
        optimizer.zero_grad()
        # Get batch data
        batch_X = X_train_tensor[i * batch_size:(i + 1) * batch_size]
        batch_y = y_train_tensor[i * batch_size:(i + 1) * batch_size]
        # Forward pass
        outputs = model(batch_X)
        # Compute loss
        loss = criterion(outputs, batch_y)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / num_batches
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

# ------------------------------------------------------
# Part 4: Evaluation
# ------------------------------------------------------

model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (outputs > 0.5).float()

# Convert tensors to numpy arrays
y_test_np = y_test_tensor.numpy()
predicted_np = predicted.numpy()

def calculate_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    correct = np.sum(y_true == y_pred)
    accuracy = correct / len(y_true)
    return accuracy

# Calculate and print accuracy
accuracy = calculate_metrics(y_test_np, predicted_np)
print(f'\nEvaluation Metrics:')
print(f'Accuracy  : {accuracy * 100:.2f}%')

# ------------------------------------------------------
# Optional: Analyzing Misclassifications
# ------------------------------------------------------

# Identify misclassified samples
misclassified_idx = np.where(y_test_np.flatten() != predicted_np.flatten())[0]

# Display some misclassified examples
print("\nSome misclassified examples:")
for idx in misclassified_idx[:5]:
    original_idx = test_indices[idx]
    print(f"Sentence: {df['sentence'].iloc[original_idx]}")
    print(f"Actual Label: {int(y_test_np[idx])}, Predicted Label: {int(predicted_np[idx][0])}\n")