from enum import StrEnum

class AutoencoderType(StrEnum):
    """Selection of autoencoders"""
    GRU = "GRU"
    LSTM = "LSTM"
    TRANSFORMER = "Transformer"
