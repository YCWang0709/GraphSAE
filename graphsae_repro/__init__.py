"""Graph-grounded sparse autoencoders and source-node explanations."""

from .activations import (
    ActivationPack,
    extract_graph_activations,
    extract_node_activations,
)
from .alignment import (
    feature_alignment_table,
    run_graph_feature_ablation,
    run_node_feature_ablation,
)
from .backbones import MODEL_NAMES, build_backbone, train_full_batch_node
from .explainer import explain_graph, explain_node
from .masking import run_mask_experiment
from .sae import SAEArtifact, SAEConfig, train_sae
from .api import GraphSAE
from .adapters import FrozenModelAdapter
from .policies import ExtractionPolicy
from .tuning import fit_sae, tune_sae, tune_pipeline
from .synthetic import SyntheticDataset, SYNTHETIC_DATASETS, generate_synthetic

__all__ = [
    "ActivationPack",
    "SyntheticDataset",
    "SYNTHETIC_DATASETS",
    "generate_synthetic",
    "GraphSAE",
    "FrozenModelAdapter",
    "ExtractionPolicy",
    "fit_sae",
    "tune_sae",
    "tune_pipeline",
    "MODEL_NAMES",
    "SAEArtifact",
    "SAEConfig",
    "build_backbone",
    "explain_graph",
    "explain_node",
    "extract_graph_activations",
    "extract_node_activations",
    "feature_alignment_table",
    "run_graph_feature_ablation",
    "run_mask_experiment",
    "run_node_feature_ablation",
    "train_full_batch_node",
    "train_sae",
]
