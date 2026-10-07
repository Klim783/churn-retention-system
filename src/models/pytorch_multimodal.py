import torch
import torch.nn as nn
from transformers import AutoModel


class ChurnMultiModalNet(nn.Module):
    def __init__(
        self,
        num_numeric_features: int,
        seq_input_size: int,
        seq_hidden_size: int = 64,
        transformer_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        # 1. Tabular Branch (MLP)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_numeric_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # 2. Activity Sequence Branch (Bi-GRU)
        self.sequence_gru = nn.GRU(
            input_size=seq_input_size,
            hidden_size=seq_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if seq_hidden_size > 1 else 0,
        )
        self.seq_fc = nn.Linear(seq_hidden_size * 2, 32)

        # 3. Text NLP Branch (Transformer)
        self.text_encoder = AutoModel.from_pretrained(transformer_model_name)
        # Freeze early transformer layers for speed
        for param in list(self.text_encoder.parameters())[:-12]:
            param.requires_grad = False

        text_hidden_size = self.text_encoder.config.hidden_size
        self.text_fc = nn.Linear(text_hidden_size, 32)

        # 4. Final Classification Head
        combined_dim = 32 + 32 + 32  # Tabular + Sequence + Text
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),  # Output raw logits
        )

    def forward(
        self,
        x_tab: torch.Tensor,
        x_seq: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Tabular pass
        out_tab = self.tabular_mlp(x_tab)

        # Sequence pass
        gru_out, _ = self.sequence_gru(x_seq)
        out_seq = self.seq_fc(gru_out[:, -1, :])

        # Text pass
        text_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        token_embeddings = text_outputs[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        text_emb = sum_embeddings / sum_mask
        out_text = self.text_fc(text_emb)

        # Combine modalities
        concat_features = torch.cat([out_tab, out_seq, out_text], dim=1)
        logits = self.classifier(concat_features)
        return logits