"""Model definitions."""

from .gcn import GCNClassifier
from .graphsage import GraphSageClassifier

__all__ = ['GCNClassifier', 'GraphSageClassifier']
