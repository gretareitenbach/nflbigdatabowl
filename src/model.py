# --- NFL Big Data Bowl 2026: Project Oracle ---
# Phase 2.2: Core Model Architecture
#
# This file defines the Transformer architecture for our Oracle model,
# including the main class, embedding layers, and a testing block.

import torch
import torch.nn as nn
import math

class OracleTransformer(nn.Module):
    """
    A Transformer-based model to predict player trajectories.
    It uses an Encoder-Decoder architecture to process the input sequence
    and generate the output sequence.
    """
    def __init__(self, input_dim: int, model_dim: int, nhead: int, num_encoder_layers: int,
                 num_decoder_layers: int, num_players_total: int, embedding_dim: int, dropout: float = 0.1):
        """
        Args:
            input_dim (int): The number of features for each player at each timestep (kinematics + embedding).
            model_dim (int): The dimensionality of the model's internal representations.
            nhead (int): The number of heads in the multiheadattention models.
            num_encoder_layers (int): The number of sub-encoder-layers in the encoder.
            num_decoder_layers (int): The number of sub-decoder-layers in the decoder.
            num_players_total (int): The total number of unique players in the dataset for the embedding layer.
            embedding_dim (int): The size of the player embedding vector.
            dropout (float): The dropout value.
        """
        super().__init__()
        self.model_dim = model_dim

        # Player Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=num_players_total, embedding_dim=embedding_dim)

        # Input Linear Projection
        self.input_projection = nn.Linear(input_dim, model_dim)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(model_dim, dropout)

        # The Core Transformer
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True # This simplifies tensor manipulation
        )

        # Output Linear Projection
        self.output_projection = nn.Linear(model_dim, 22 * 2)

    def forward(self, src: torch.Tensor, player_ids: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            src (torch.Tensor): The source sequence (pre-throw kinematic data).
                                Shape: (batch_size, seq_len_in, 22, 6)
            player_ids (torch.Tensor): The nflId for each player slot.
                                       Shape: (batch_size, seq_len_in, 22)
            tgt (torch.Tensor): The target sequence for the decoder (post-throw data).
                                Used for "teacher forcing" during training.
                                Shape: (batch_size, seq_len_out, 22, 2)
        Returns:
            torch.Tensor: The model's prediction.
                          Shape: (batch_size, seq_len_out, 22, 2)
        """
        batch_size, seq_len_in, _, _ = src.shape
        _, seq_len_out, _, _ = tgt.shape

        # Look up embeddings and combine with kinematic features
        embeds = self.embedding(player_ids)
        combined_features = torch.cat([src, embeds], dim=-1)

        # Flatten the player and feature dimensions for the linear layers
        src = combined_features.view(batch_size, seq_len_in, -1)

        # For the target, we create a simple dummy input for the decoder
        tgt_input = torch.zeros(batch_size, seq_len_out, self.model_dim, device=src.device)

        # Model Pass
        src = self.input_projection(src) * math.sqrt(self.model_dim)
        src = self.pos_encoder(src)

        output = self.transformer(src, tgt_input)

        output = self.output_projection(output)

        # Reshape the output to our desired format
        output = output.view(batch_size, seq_len_out, 22, 2)

        return output

class PositionalEncoding(nn.Module):
    """ Helper class for Positional Encoding """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

# --- Testing Block ---
if __name__ == "__main__":
    print("--- Testing OracleTransformer Architecture with Embeddings ---")

    # 1. Define Model Hyperparameters
    BATCH_SIZE = 4
    SEQ_LEN_IN = 70
    SEQ_LEN_OUT = 50
    NUM_PLAYERS = 22
    INPUT_FEATURES_PER_PLAYER = 6
    OUTPUT_FEATURES_PER_PLAYER = 2

    TOTAL_UNIQUE_PLAYERS = 6000
    EMBEDDING_DIM = 10

    MODEL_DIM = 128
    N_HEADS = 4
    NUM_ENCODER_LAYERS = 2
    NUM_DECODER_LAYERS = 2

    combined_player_features = INPUT_FEATURES_PER_PLAYER + EMBEDDING_DIM
    model_input_dim = NUM_PLAYERS * combined_player_features

    # 2. Instantiate the Model
    model = OracleTransformer(
        input_dim=model_input_dim,
        model_dim=MODEL_DIM,
        nhead=N_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        num_players_total=TOTAL_UNIQUE_PLAYERS,
        embedding_dim=EMBEDDING_DIM
    )
    print("Model instantiated successfully.")

    # 3. Create Dummy Input Data
    dummy_src = torch.randn(BATCH_SIZE, SEQ_LEN_IN, NUM_PLAYERS, INPUT_FEATURES_PER_PLAYER)
    dummy_ids = torch.randint(0, TOTAL_UNIQUE_PLAYERS, (BATCH_SIZE, SEQ_LEN_IN, NUM_PLAYERS))
    dummy_tgt = torch.randn(BATCH_SIZE, SEQ_LEN_OUT, NUM_PLAYERS, OUTPUT_FEATURES_PER_PLAYER)

    # 4. Perform a Forward Pass
    with torch.no_grad():
        prediction = model(dummy_src, dummy_ids, dummy_tgt)

    print(f"\nShape of model prediction: {prediction.shape}")

    # 5. Verify Output Shape
    expected_shape = (BATCH_SIZE, SEQ_LEN_OUT, NUM_PLAYERS, OUTPUT_FEATURES_PER_PLAYER)
    assert prediction.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {prediction.shape}"

    print("\n--- Model Architecture Test Complete ---")
    print("The model correctly incorporates embeddings and produces the correct output shape.")
