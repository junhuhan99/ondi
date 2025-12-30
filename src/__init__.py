"""
OnDi - On-Device AI Model
100% From Scratch | 100% Owned License
"""

from .model import OnDiModel, get_model_config
from .tokenizer import BPETokenizer
from .dataset import OnDiDataset, prepare_training_data

__version__ = "1.0.0"
__author__ = "Jun Hu Han"
__license__ = "Proprietary - 100% Owned"
