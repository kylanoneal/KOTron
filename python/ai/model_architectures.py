import torch
from torch import nn
import torch.nn.functional as F


# Stride == 1
class FastNet(nn.Module):

    def __init__(self, grid_dim=10):

        super(FastNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=0)

        output_dim1 = self.conv_output_size(
            grid_dim, kernel_size=3, padding=0, stride=1
        )
        output_dim2 = self.conv_output_size(
            output_dim1, kernel_size=3, padding=0, stride=1
        )

        fc1_input_neurons = 10 * output_dim2**2

        #print(f"Conv1 output size: {output_dim1}")
        #print(f"Conv2 output size: {output_dim2}")
        #print(f"FC1 input size: {fc1_input_neurons}")

        self.fc1 = nn.Linear(fc1_input_neurons, 50)  # input features for fc1
        self.fc_value = nn.Linear(50, 1)

    @staticmethod
    def conv_output_size(input_size, kernel_size, padding, stride):
        return ((input_size - kernel_size + 2 * padding) // stride) + 1

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))

        x = self.fc_value(x).squeeze(1)

        return x


# Stride == 1
class EvaluationNetConv3OneStride(nn.Module):
    PADDING = 2

    def __init__(self, grid_dim=40):
        # FIX PADDING TO BE IMPLICIT HERE
        super(EvaluationNetConv3OneStride, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 10, kernel_size=5, stride=1, padding=0
        )  # changed stride to 2
        self.conv2 = nn.Conv2d(
            10, 20, kernel_size=3, stride=1, padding=1
        )  # changed stride to 2
        output_dim1 = self.conv_output_size(grid_dim, 5, 2, 1)  # after first conv layer
        output_dim2 = self.conv_output_size(
            output_dim1, 3, 1, 1
        )  # after second conv layer
        self.fc1 = nn.Linear(20 * output_dim2**2, 250)  # input features for fc1
        self.fc_value = nn.Linear(250, 1)

    @staticmethod
    def conv_output_size(input_size, kernel_size, padding, stride):
        return ((input_size - kernel_size + 2 * padding) // stride) + 1

    def forward(self, x):
        padded_game_grid = F.pad(
            x[:, 0:1], pad=(2, 2, 2, 2), mode="constant", value=1.0
        )
        padded_heads = F.pad(x[:, 1:], pad=(2, 2, 2, 2), mode="constant", value=0.0)
        concat = torch.concat((padded_game_grid, padded_heads), dim=1)


        x = F.relu(self.conv1(concat))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten the tensor

        x = F.relu(self.fc1(x))
        
        value_estimate = self.fc_value(x).squeeze(1)

        return value_estimate

# Stride == 1
class LeakyReLU(nn.Module):
    PADDING = 2

    def __init__(self, grid_dim=40):
        # FIX PADDING TO BE IMPLICIT HERE
        super(LeakyReLU, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 10, kernel_size=5, stride=1, padding=0
        )  # changed stride to 2
        self.conv2 = nn.Conv2d(
            10, 20, kernel_size=3, stride=1, padding=1
        )  # changed stride to 2
        output_dim1 = self.conv_output_size(grid_dim, 5, 2, 1)  # after first conv layer
        output_dim2 = self.conv_output_size(
            output_dim1, 3, 1, 1
        )  # after second conv layer
        self.fc1 = nn.Linear(20 * output_dim2**2, 250)  # input features for fc1
        self.fc_value = nn.Linear(250, 1)

    @staticmethod
    def conv_output_size(input_size, kernel_size, padding, stride):
        return ((input_size - kernel_size + 2 * padding) // stride) + 1

    def forward(self, x):
        padded_game_grid = F.pad(
            x[:, 0:1], pad=(2, 2, 2, 2), mode="constant", value=1.0
        )
        padded_heads = F.pad(x[:, 1:], pad=(2, 2, 2, 2), mode="constant", value=0.0)
        concat = torch.concat((padded_game_grid, padded_heads), dim=1)


        x = F.relu(self.conv1(concat))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten the tensor

        x = F.leaky_relu(self.fc1(x))
        
        value_estimate = self.fc_value(x).squeeze(1)

        return value_estimate
    

import torch
import torch.nn as nn

class TransformerGameEvaluator(nn.Module):
    def __init__(self, grid_size, num_classes=1):
        super(TransformerGameEvaluator, self).__init__()
        
        self.grid_size = grid_size
        self.num_tokens = grid_size ** 2  # Flattened grid size
        
        # # If you want one-hot encoding, this layer is not necessary, just use the grid values directly
        # self.embedding = nn.Embedding(4, 4)  # 4 categories (4 possible values)
        
        # Positional encoding (grid positions)
        self.positional_encoding = nn.Parameter(torch.rand(1, self.num_tokens, 4))  # 4 features for each token
        
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=4, nhead=1),  # Use d_model=4 for one-hot size
            num_layers=10  # You can increase this for a deeper model
        )
        
        # Final output layer (to predict evaluation score)
        self.fc = nn.Linear(4, num_classes)  # Scalar output for evaluation
    
    def forward(self, x):
        # Step 1: Embed the grid cells (tokens)
#        embedded = self.embedding(x)  # (batch_size, num_tokens, 4)
        
        # Step 2: Add positional encoding to the token embeddings
        x += self.positional_encoding
        
        # Step 3: Reshape the embeddings for the transformer (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # (batch_size, num_tokens, embed_dim) -> (num_tokens, batch_size, embed_dim)
        
        # Step 4: Pass through the transformer encoder
        transformer_out = self.transformer_encoder(x)


        # # Step 5: Pool the transformer output (we use the last token or mean)

        # transformer_pool = transformer_out.mean(dim=0)  # [batch, embed_dim]
        transformer_pool = transformer_out[-1, :, :]  # Last token's output
        
        # # Step 6: Feed through the classification head (fully connected layer)
        logits = self.fc(transformer_pool).squeeze(-1)  # [batch]
        return logits
    

class EmbeddingTransformerGameEvaluator(nn.Module):
    def __init__(self, grid_size, embed_dim):
        super(EmbeddingTransformerGameEvaluator, self).__init__()
        
        self.grid_size = grid_size
        self.num_tokens = grid_size ** 2  # Flattened grid size
        
        # # If you want one-hot encoding, this layer is not necessary, just use the grid values directly
        self.embedding = nn.Embedding(4, embed_dim)  # 4 categories (4 possible values)
        
        # Positional encoding (grid positions)
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))  # 4 features for each token
        
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8),  # Use d_model=4 for one-hot size
            num_layers=10  # You can increase this for a deeper model
        )
        
        # Final output layer (to predict evaluation score)
        self.fc = nn.Linear(embed_dim, 1)  # Scalar output for evaluation
    
    def forward(self, x):
        # Step 1: Embed the grid cells (tokens)
        x = self.embedding(x)  # (batch_size, num_tokens, 4)

        #print(f"Sum x after embedding: {x.sum()}")
        
        # Step 2: Add positional encoding to the token embeddings
        x += self.positional_encoding

        #print(f"Sum positional encoding: {self.positional_encoding.sum()}")
        
        # Step 3: Reshape the embeddings for the transformer (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # (batch_size, num_tokens, embed_dim) -> (num_tokens, batch_size, embed_dim)
        
        # Step 4: Pass through the transformer encoder
        x = self.transformer_encoder(x)

        x = x.mean(dim=0)  # [batch, embed_dim]
        x = self.fc(x).squeeze(-1)  # [batch]
        
        # # Step 5: Pool the transformer output (we use the last token)
        # # output = transformer_out[-1, :, :]  # Last token's output
        
        # # Step 6: Feed through the classification head (fully connected layer)
        # evaluation_score = self.fc(output).squeeze(1)
        return x



