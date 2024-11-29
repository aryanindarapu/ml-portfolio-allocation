import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import TransformerModel
from dataset import FinancialDataset

class SharpeRatioLoss(nn.Module):
    def __init__(self, C=0.0002):  # C = 2 basis points (0.0002)
        super(SharpeRatioLoss, self).__init__()
        self.C = C  # transaction cost rate

    def forward(self, weights, returns):
        """
        Parameters:
        - weights: Tensor of shape (tau, N) where tau is the number of time steps and N is the number of assets.
                   Each row represents the asset weights for a given day, which should sum to 1.
        - returns: Tensor of shape (tau, N) where each element represents the return of an asset on a given day.
        
        Returns:
        - Sharpe Ratio-based loss (negative Sharpe Ratio to minimize)
        """

        # Portfolio returns (with transaction costs)
        portfolio_returns = torch.sum(weights[:-1] * returns[1:], dim=1) - \
                            self.C * torch.sum(torch.abs(weights[1:] - weights[:-1]), dim=1)

        # Mean and variance of portfolio returns over the trading period
        expected_return = portfolio_returns.mean()
        expected_square_return = (portfolio_returns ** 2).mean()
        variance_return = expected_square_return - expected_return ** 2

        # Sharpe Ratio
        sharpe_ratio = expected_return / torch.sqrt(variance_return + 1e-8)  # Adding epsilon for stability

        # Loss is negative Sharpe Ratio (to maximize Sharpe Ratio)
        loss = -sharpe_ratio
        return loss

if __name__ == "__main__":
    # Set parameters for training
    sequence_length = 5
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 100

    # Instantiate Dataset and DataLoader
    csv_file = "etf_returns_2006_2023.csv"  # Replace with the actual path to your CSV file
    dataset = FinancialDataset(csv_file, sequence_length=sequence_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    input_dim = dataset.returns.shape[1]  # Number of assets
    embedding_dim = 64
    num_heads = 4
    hidden_dim = 128
    num_layers = 4
    output_dim = input_dim

    model = TransformerModel(input_dim, embedding_dim, num_heads, hidden_dim, num_layers, output_dim)
    loss_fn = SharpeRatioLoss(C=0.0002)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in data_loader:
            # Transpose to fit model input shape (sequence_length, batch_size, num_assets)
            batch_x = batch_x.transpose(0, 1)
            batch_y = batch_y.transpose(0, 1)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predicted_weights = model(batch_x, batch_x)
            
            # Compute Sharpe Ratio-based loss
            loss = loss_fn(predicted_weights, batch_y)
            
            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()
            
            # Accumulate epoch loss
            epoch_loss += loss.item() * batch_x.size(1)  # Multiply by batch size

        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

    print("Training complete.")
