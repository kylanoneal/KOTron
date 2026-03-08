import torch
import torch.nn as nn

class TransformerImageRegressor(nn.Module):
    def __init__(
        self,
        image_height: int,
        image_width: int,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        """
        Args:
            image_height (int): Height of the input image (pixels).
            image_width (int): Width of the input image (pixels).
            embed_dim (int): Dimensionality of token embeddings (and Transformer embeddings).
            num_heads (int): Number of attention heads in each Transformer layer.
            num_layers (int): Number of TransformerEncoder layers.
            dim_feedforward (int): Hidden size of the feedforward network inside each Transformer layer.
            dropout (float): Dropout probability (applied inside Transformer layers).
        """
        super().__init__()

        self.H = image_height
        self.W = image_width
        self.seq_len = self.H * self.W  # number of pixel‐tokens
        self.embed_dim = embed_dim

        # 1) Pixel embedding: map {0,1,2,3} → R^embed_dim
        self.token_embed = nn.Embedding(num_embeddings=4, embedding_dim=embed_dim)

        # 2) Learnable [CLS] token
        #    We create one vector of size (1, D) and expand it per batch in forward.
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 3) Learnable positional embedding for (seq_len + 1) positions
        #    Index 0 will correspond to [CLS], and indices 1..seq_len correspond to each pixel position.
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len + 1, embed_dim))

        # 4) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,  # we’ll transpose to (S+1, B, D) below
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5) Regression head: from embed_dim → 1 scalar
        #    You can make this deeper if you wish.
        self.regress_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape (B, H, W), dtype long or int, with values in {0,1,2,3}.
        Returns:
            Tensor of shape (B,) with values in [-1, +1].
        """
        B, num_tokens = x.shape
        assert num_tokens == 100


        token_embeddings = self.token_embed(x)  # long → (B, S, D)

        # 2) Prepare the CLS token: replicate for each batch
        #    cls_token: originally (1, 1, D); expand to (1, B, D), then will concat
        cls_tokens = self.cls_token.expand(-1, B, -1)  # (1, B, D)

        # 3) Rearrange token_embeddings to (S, B, D) for Transformer
        #    and prepend the cls_tokens so the final shape is (S+1, B, D)
        token_embeddings = token_embeddings.transpose(0, 1)   # (S, B, D)
        tokens_with_cls = torch.cat([cls_tokens, token_embeddings], dim=0)  # (S+1, B, D)

        # 4) Add positional embeddings: pos_embed is (1, S+1, D), so transpose to (S+1, 1, D)
        #    Then we broadcast over the batch dimension and add.
        pos = self.pos_embed.transpose(0, 1)  # (S+1, 1, D)
        tokens_with_pos = tokens_with_cls + pos  # auto-broadcast over B

        # 5) Pass through Transformer Encoder: output shape is (S+1, B, D)
        transformer_out = self.transformer(tokens_with_pos)  # (S+1, B, D)

        # 6) Extract the CLS embedding (first token) → shape (B, D)
        #cls_out = transformer_out[0].transpose(0, 1)  # (B, D)
        cls_out = transformer_out[0]

        # 7) Regression head to a single scalar per image → shape (B, 1)
        reg_out = self.regress_head(cls_out)  # (B, 1)

        # 8) Squeeze to (B,) and apply tanh to bound to [-1, +1]
        reg_out = reg_out.squeeze(-1)       # (B,)
        reg_out = torch.tanh(reg_out)       # each in [-1, +1]

        return reg_out
