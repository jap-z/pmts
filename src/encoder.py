import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import argparse
import time
from tqdm import tqdm

class Time2Vec(nn.Module):
    """
    Time2Vec implementation to provide temporal structural awareness to the model.
    Learns periodic and non-periodic representations of the sequential steps.
    """
    def __init__(self, seq_len, num_features=4):
        super(Time2Vec, self).__init__()
        self.seq_len = seq_len
        self.w0 = nn.Parameter(torch.randn(seq_len, 1))
        self.b0 = nn.Parameter(torch.randn(seq_len, 1))
        self.w = nn.Parameter(torch.randn(seq_len, num_features - 1))
        self.b = nn.Parameter(torch.randn(seq_len, num_features - 1))

    def forward(self, x):
        batch_size = x.size(0)
        # Create a time vector [0, 1, 2, ... seq_len-1]
        tau = torch.arange(self.seq_len, dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        
        # Linear component
        v1 = tau * self.w0 + self.b0
        # Periodic components
        v2 = torch.sin(tau * self.w + self.b)
        
        t2v = torch.cat([v1, v2], dim=-1)
        # Concatenate time features to original features
        return torch.cat([x, t2v], dim=-1)

class Encoder(nn.Module):
    def __init__(self, input_dim=5, t2v_dim=4, hidden_dim=128, emb_dim=256, seq_len=28):
        super(Encoder, self).__init__()
        self.t2v = Time2Vec(seq_len=seq_len, num_features=t2v_dim)
        # Bidirectional LSTM for better context capture with 3 layers and dropout
        self.lstm = nn.LSTM(input_size=input_dim + t2v_dim, hidden_size=hidden_dim, num_layers=3, batch_first=True, bidirectional=True, dropout=0.2)
        # Since it's bidirectional, hidden state is 2 * hidden_dim
        self.fc = nn.Linear(hidden_dim * 2, emb_dim)
        
    def forward(self, x):
        x = self.t2v(x)
        out, (hn, cn) = self.lstm(x)
        # Extract the last hidden states from both directions of the top layer
        hidden_fwd = hn[-2]
        hidden_bwd = hn[-1]
        last_hidden = torch.cat((hidden_fwd, hidden_bwd), dim=1)
        
        emb = self.fc(last_hidden)
        return emb

class Decoder(nn.Module):
    def __init__(self, emb_dim=256, hidden_dim=128, output_dim=5, seq_len=28):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(emb_dim, hidden_dim * 2)
        # Unidirectional LSTM for reconstruction
        self.lstm = nn.LSTM(input_size=hidden_dim * 2, hidden_size=hidden_dim * 2, num_layers=3, batch_first=True, dropout=0.2)
        self.out = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, emb):
        x = self.fc(emb)
        # Repeat the embedding for the entire sequence length
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(x)
        res = self.out(out)
        return res

class TSAutoencoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, emb_dim=256, seq_len=28):
        super(TSAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, emb_dim=emb_dim, seq_len=seq_len)
        self.decoder = Decoder(emb_dim=emb_dim, hidden_dim=hidden_dim, output_dim=input_dim, seq_len=seq_len)
        
    def forward(self, x):
        emb = self.encoder(x)
        reconstruction = self.decoder(emb)
        return reconstruction, emb

def train_and_extract_embeddings(data_path='data/btc_6h_features.npz', epochs=500, batch_size=128, emb_dim=256, checkpoint_dir='data/checkpoints'):
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    X_raw = data['X']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_raw, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, X_tensor) # Autoencoder targets itself
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model
    _, seq_len, num_features = X_tensor.shape
    model = TSAutoencoder(input_dim=num_features, hidden_dim=128, emb_dim=emb_dim, seq_len=seq_len).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'ts2vec_checkpoint.pth')
    
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}...")
    
    print(f"Training Time Series Autoencoder for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_Y in dataloader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            optimizer.zero_grad()
            reconstructed, _ = model(batch_X)
            loss = criterion(reconstructed, batch_Y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            epochs_run = epoch - start_epoch + 1
            if epochs_run > 0:
                est_total = (elapsed / epochs_run) * (epochs - start_epoch)
                eta = est_total - elapsed
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f} | ETA: {eta/60:.1f} min")
            
            # Save Checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
            
    # Extract Embeddings for all data
    print("Extracting semantic embeddings for the entire dataset...")
    model.eval()
    all_embeddings = []
    
    # Use un-shuffled data to keep ordering
    eval_loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for (batch_X,) in tqdm(eval_loader, desc="Generating Vectors"):
            batch_X = batch_X.to(device)
            _, embs = model(batch_X)
            all_embeddings.append(embs.cpu().numpy())
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Embeddings shape: {all_embeddings.shape}")
    
    # Save the updated dataset
    out_file = os.path.join(os.path.dirname(data_path), 'btc_6h_embeddings.npz')
    np.savez_compressed(
        out_file,
        X=X_raw,
        embeddings=all_embeddings,
        y_close_ret=data['y_close_ret'],
        y_max_ret=data['y_max_ret'],
        y_min_ret=data['y_min_ret'],
        timestamps=data['timestamps']
    )
    print(f"Embeddings successfully saved to {out_file}")
    
    # Save the model
    torch.save(model.state_dict(), os.path.join(os.path.dirname(data_path), 'ts2vec_autoencoder.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/btc_6h_features.npz')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--emb_dim', type=int, default=128)
    args = parser.parse_args()
    
    train_and_extract_embeddings(args.input, args.epochs, emb_dim=args.emb_dim)