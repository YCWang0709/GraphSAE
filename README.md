# GraphSAE Minimal Reproduction

This folder is a standalone, submission-oriented implementation of the core GraphSAE experiments. It deliberately excludes paper figures, intermediate caches, and the full experiment-management code. Cora is downloaded automatically; the graph-classification demo uses a public TUDataset.

## Included Components

- Last-layer activation extraction for `GCN`, `GAT`, `GIN`, and `GraphSAGE`.
- Top-k SAE training with standardized activations, reconstruction loss, L1 code penalty, and decoder-column normalization.
- SAE feature-label alignment and causal SAE feature ablation.
- SAE-GNNExplainer for node and graph classification.
- Original-graph node masking with feature, edge, or joint masking.
- A Cora notebook and command-line scripts for the full pipeline.

The code intentionally uses only the final message-passing layer. For node classification, this hidden state is classified directly. For graph classification, the same hidden state is mean pooled before classification; the graph explainer therefore allocates active SAE directions back to contributing node states.

## Model-Specific Last-Layer Handling

| Backbone | First layer | Final SAE input |
| --- | --- | --- |
| GCN | `GCNConv + ReLU` | `GCNConv + ReLU` output |
| GraphSAGE | `SAGEConv + ReLU` | `SAGEConv + ReLU` output |
| GIN | `GINConv(MLP) + ReLU` | `GINConv(MLP) + ReLU` output |
| GAT | Multi-head `GATConv + ELU` | One-head, `concat=False`, `GATConv + ELU` output |

All four models expose the same capture contract: `model(..., capture=True)` returns `logits` and a `LastLayerCapture`. This makes SAE training and explanation code model agnostic while retaining the correct GAT head and activation handling.

## Installation

Create an environment compatible with your PyTorch CUDA or CPU installation, then run:

```powershell
cd paper_reproduction
pip install -r requirements.txt
pip install -e .
```

For a CUDA installation, install the matching PyTorch and PyG wheels first, following the official PyTorch Geometric installation instructions, and then install the remaining requirements.

## Cora End-to-End Reproduction

Run the following commands from `paper_reproduction`. The default output location is `artifacts/cora`.

```powershell
python scripts/01_train_cora_backbone.py --model GCN --epochs 300
python scripts/02_extract_activations.py --model GCN --split train
python scripts/03_train_sae.py --model GCN --activation-split train --epochs 200
python scripts/04_feature_alignment.py --model GCN
python scripts/05_sae_gnnexplainer.py --model GCN --top-k 8
python scripts/06_mask_experiment.py --model GCN --top-k 8 --mask-mode both
```

Repeat the same six commands with `--model GAT`, `--model GIN`, or `--model GraphSAGE` to reproduce the cross-backbone workflow. Pass `--device cuda` when a CUDA device is available, or `--device cpu` for CPU-only execution.

Artifacts are intentionally explicit:

- `backbone_*.pt`: trained GNN and its architecture configuration.
- `activations_*_train.pt`: final-layer activation matrix and row metadata.
- `sae_*.pt`: encoder/decoder weights, activation normalizer, configuration, metrics, and history.
- `feature_alignment_*.csv`: feature purity, activity, and majority-class statistics.
- `feature_ablation_*.json`: causal target-logit/probability deltas from SAE feature removal.
- `explanation_*.json`: SAE+, SAE-, and SAE+/- node explanations plus all node scores.
- `mask_experiment_*.json`: prediction changes after masking each explanation group.

## Graph-Classification Path

The public graph demo exercises the graph-specific activation, alignment, explainer, and masking paths:

```powershell
python scripts/07_graph_classification_demo.py --dataset MUTAG --model GIN
```

For graph classification, `extract_graph_activations` trains the SAE on pooled final-layer graph vectors. `explain_graph` projects active SAE decoder directions onto final-layer node states and divides by the number of nodes, matching mean pooling. `run_graph_feature_ablation` patches the pooled graph representation, while `run_mask_experiment` masks the selected original-graph nodes.

## SAE-GNNExplainer Definition

For a target representation `h`, the SAE produces sparse codes `a = TopK(ReLU(W_enc h + b_enc))` and reconstruction `h_hat = W_dec a + b_dec`. The explainer selects target-active features and measures each feature's class-logit direction using `W_cls[c]^T W_dec[:, f]`.

- Node task: candidate nodes are restricted to the target's message-passing receptive field. Their signed score is the sum of their active SAE code values weighted by each selected feature's target-class effect.
- Graph task: candidate nodes contribute through their mean-pooled final-layer activation. The graph score projects each node state onto the selected decoder directions, weighted by the graph-active SAE feature effects.

`SAE+` contains the highest positive signed scores, `SAE-` contains the most negative scores, and `SAE+/-` is their union. The scripts report all three groups in masking experiments.

## Notebook

Open `notebooks/GraphSAE_Cora_Minimal_Reproduction.ipynb` from this directory. It invokes the same scripts in order and loads the resulting JSON/CSV artifacts for inspection.

## Scope

This folder is the minimal algorithmic reproduction package. The full paper-specific synthetic data generation, all 13-dataset batch manifests, and final plotting code remain in the parent project because they are not necessary to audit the SAE, alignment, explanation, and masking procedures. Exact numerical replication also requires matching the original dataset split, checkpoint seed, and hardware/software stack.
