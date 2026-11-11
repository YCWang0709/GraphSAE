# GraphSAE

Graph-grounded sparse autoencoders and SAE-GNNExplainer for frozen graph predictors. The package provides general SAE training and validation search, source-node contribution decomposition, explanation extraction, deletion evaluation, and controlled ablations.

## Cora with pretrained checkpoints

A complete inference example is available in [examples/cora_pretrained](examples/cora_pretrained/README.md). It bundles four frozen backbone checkpoints and 12 tuned SAE checkpoints, recomputes the Cora paper row, and renders one-hop explanation subgraphs. Run it without model or SAE fitting:

```powershell
python -m pip install -r examples/cora_pretrained/requirements.txt
python examples/cora_pretrained/run.py --check-paper
```

## Installation

Python 3.10 or later is required. Install a PyTorch build suitable for the available CPU or CUDA runtime, then install the package:

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
```

## Code layout

| Component | Implementation |
| --- | --- |
| SAE encoders, objectives, training, checkpoints | `graphsae_repro/sae.py`, `sparse_ops.py` |
| General tensor training and validation selection | `graphsae_repro/tuning.py` |
| GCN, GIN, GAT, mean-GraphSAGE | `graphsae_repro/backbones.py` |
| Source contribution vectors and reverse projections | `graphsae_repro/contributions.py` |
| SAE-GNNExplainer and extraction/refinement policies | `graphsae_repro/explainer.py`, `policies.py` |
| Custom and heterogeneous predictors | `graphsae_repro/adapters.py`, `api.py` |
| Representation, recovery, and deletion metrics | `graphsae_repro/evaluation.py`, `alignment.py`, `masking.py` |
| Regression groups and stable mean-GraphSAGE replay | `graphsae_repro/grounding.py`, `numerics.py` |
| Ablation settings and pure-path control | `graphsae_repro/ablations.py` |
| Synthetic dataset construction, annotations, storage | `graphsae_repro/synthetic.py`, `synthetic_node.py`, `synthetic_graph.py` |
| Molecular mask utilities and dataset compatibility functions | `graphsae_repro/datasets.py` |
| Command-line entry point | `graphsae_repro/cli.py` |

The supplementary protocol is mapped to the implementation in [METHOD.md](METHOD.md). The Cora inference example includes its frozen predictor and SAE checkpoints. The remaining repository provides source code, configuration templates, and examples; datasets, generated experiment outputs, and paper figures are not distributed.

## Synthetic dataset construction

`graphsae_repro.synthetic` provides the six tasks from the supplementary's **Synthetic Dataset Construction** section. Its default configuration is [configs/synthetic_datasets.json](configs/synthetic_datasets.json).

| Task | Default construction | Seed | Stratified train / validation / test |
| --- | --- | --- | --- |
| BATriadic | `powerlaw_cluster_graph(n=2000, m=3, p=0.35)`; degree/triangle rank weights 0.65/0.35 | 37 | 60% / 20% / 20% |
| BAPathRole | BA(750, 3) with 250 five-node paths; background/internal/endpoint labels | 43 | 60% / 20% / 20% |
| BABiFanRole | BA(750, 3) with 250 complete bipartite K2,3 motifs; background/two partition roles | 47 | 60% / 20% / 20% |
| BAHouse | BA(m=2, n uniform on 22..38) with 0/1/2 house motifs; 300 graphs per class | 101 | 70% / 15% / 15% |
| BAPath | Same BA graph distribution with 0/1/2 five-node paths; 300 graphs per class | 404 | 70% / 15% / 15% |
| BACycle | Same BA graph distribution with 0/1/2 six-cycles; 300 graphs per class | 202 | 70% / 15% / 15% |

Node motif anchors use the specified 0.55 degree-plus-one / 0.45 uniform mixture. Node features combine structural statistics and Gaussian coordinates. Graph tasks use six structural features: constant, normalized degree, clustering, normalized triangle count, normalized core number, and normalized PageRank. Integer rounding is applied within each class when allocating splits.

Generate all six datasets, or one task, without training a predictor:

```powershell
python -m graphsae_repro generate-data --dataset all --config configs/synthetic_datasets.json --output data/synthetic
python -m graphsae_repro generate-data --dataset BAPathRole --output data/synthetic
```

Each task writes `<output>/<name>/dataset.pt` and `metadata.json`. The bundle contains raw PyG data, labels, split indices, construction settings, and reference annotations; the metadata records construction parameters and dataset sizes. Generated files belong under the ignored `data/` directory. The repository distributes construction code and configuration only.

The Python API supports the same defaults, explicit overrides, saving, and loading:

```python
from graphsae_repro import SyntheticDataset, generate_synthetic

dataset = generate_synthetic("BAPathRole")
data = dataset.data
reference_nodes = dataset.reference(target=750, candidates=range(data.num_nodes))
dataset.save("data/synthetic/BAPathRole")
restored = SyntheticDataset.load("data/synthetic/BAPathRole")

graphs = generate_synthetic("BAHouse")
train_graphs = [graphs.graphs[i] for i in graphs.splits["train"]]
```

Role tasks store `motif_node_id`, the planted-node union mask, internal motif-edge masks, and a separate edge mask including attachments. Target-specific reference queries use stored motif membership. BATriadic references use structural-role agreement within the eligible field; validation queries restrict source class labels to training/validation nodes. Graph tasks retain their planted node and edge masks, including a separate attachment-inclusive mask. Reference annotations are not appended to `x`.

Development overrides such as `generate_synthetic("BAHouse", num_graphs=12)` are explicit changes from the supplementary setting. For a single-task CLI configuration, supply an options object such as `{"backbone_nodes": 30, "num_motifs": 6}`; the all-task configuration uses the `datasets` mapping shown in the supplied JSON. `examples.synthetic:load` accepts `dataset_path` to reuse a generated bundle and `generator_options` for explicit construction overrides.

## General SAE training

For an arbitrary activation matrix, use the tensor API. It requires neither a GNN nor semantic labels:

```python
import torch
from graphsae_repro import SAEConfig, fit_sae

activations = torch.randn(1000, 64)
config = SAEConfig(
    epochs=100, expansion=8, k_active=16,
    lambda_ground=0, lambda_faith=0, lambda_coh=0,
)
artifact = fit_sae(activations, config)
codes = artifact.encode(activations)
artifact.save("artifacts/general/sae.pt")
```

To use grounding, supply training group IDs. To use prediction faithfulness, supply the frozen downstream prediction head. `GraphSAE.fit` uses all four GraphSAE objectives by default:

```python
from graphsae_repro import GraphSAE, SAEConfig

graphsae = GraphSAE.fit(train_pack, frozen_model.classifier, SAEConfig())
```

`ActivationPack` stores raw representations in `z` and one metadata record per row. Declare `split="train"`, `"val"`, or `"test"` in both the metadata and `info`. Grounding uses the metadata `group` field, falling back to `label` for classification. Use stable `node_id`, `graph_id`, or `sample_id` values across splits.

The equivalent CLI accepts saved activation packs. A head file is a `torch.nn.Linear.state_dict()` with `weight` and optional `bias` tensors:

```powershell
graphsae train-sae --train train.pt --config configs/general_sae.json --output artifacts/general
graphsae train-sae --train train.pt --head head.pt --config configs/graphsae_fixed.json --output artifacts/fixed
graphsae tune-sae --train train.pt --validation val.pt --head head.pt --space configs/sae_search.json --criterion fidelity --output artifacts/sae_search
```

## Complete explanation selection

An experiment factory returns an `Experiment` containing the frozen predictor, training representations, and separate validation/test explanation targets. Factories are supplied for a small offline example, Cora, synthetic benchmarks, and existing PyG checkpoints. A custom factory can preserve a dataset's original preprocessing and split definitions.

Run the complete workflow without a dataset download:

```powershell
graphsae tune-explainer --factory examples.toy:load --space configs/smoke_search.json --policies configs/policies_fixed.json --output artifacts/toy
graphsae explain --factory examples.toy:load --sae artifacts/toy/sae_seed_0.pt --policy artifacts/toy/selected_policy.json --split test --output artifacts/toy_test
```

Use `configs/sae_search.json` for a larger reference search, and supply factory arguments through `--factory-options options.json`. The search varies dictionary width, active count, learning rate, duration, and auxiliary weights. The example grid is editable: the supplementary specifies these search dimensions, rather than a single universal candidate grid. No dataset-specific selected parameters or measured outcomes are embedded in it.

Fixed-budget selection uses one training/ranking policy across budgets 8, 12, and 16. Native-size selection uses `--native --policies configs/policies_native.json`. Native policies include positive support, relative thresholds, positive-score mass, ranked prefixes, and greedy additions/removals. Separate templates cover topology-based city-reviews masks and block refinement. Search uses seed 0, then refits the selected training configuration with seeds 0, 1, and 2. Test targets are evaluated only by the separate `explain` command.

The validation utility is explicit: mean probability decrease plus a configurable Jaccard weight, minus a configurable node-count penalty. Regression uses mean absolute prediction change. Set these weights before running selection; the selected utility, training configuration, policy, and trials are saved together. For node datasets whose reference masks depend on source class labels, validation references must exclude test-source labels; `node_reference` implements this restriction for BATriadic.

## Cora and graph classification

The numbered scripts retain a direct Cora workflow:

```powershell
python scripts/01_train_cora_backbone.py --model GCN
python scripts/02_extract_activations.py --model GCN --split train
python scripts/02_extract_activations.py --model GCN --split val
python scripts/02_extract_activations.py --model GCN --split test
python scripts/03_train_sae.py --model GCN
python scripts/04_feature_alignment.py --model GCN
python scripts/05_sae_gnnexplainer.py --model GCN --top-k 12
python scripts/06_mask_experiment.py --model GCN --top-k 12 --mask-mode both
```

Repeat with `GAT`, `GIN`, or `GraphSAGE`. Backbone training selects its checkpoint by validation loss. Existing checkpoints from the former two-layer implementation must be retrained for the current configurable backbone architecture.

For Cora explanation tuning, create a factory-options JSON pointing to the newly trained checkpoint:

```json
{"checkpoint": "artifacts/cora/backbone_gcn.pt", "targets_per_split": 25}
```

Then use `--factory examples.cora:load --factory-options options.json` with `tune-explainer` and `explain`. Target lists are deterministic and are not filtered by test prediction correctness.

The graph-classification workflow downloads MUTAG, creates stratified training/validation/test splits, saves a frozen predictor and activation packs, and runs SAE training and explanation:

```powershell
python scripts/07_graph_classification_demo.py --dataset MUTAG --model GIN
```

Use `examples.from_checkpoint:load` with its generated `backbone.pt` and `dataset.pt` for pipeline selection. For the six synthetic benchmarks, `examples.synthetic:load` supports `BATriadic`, `BAPathRole`, `BABiFanRole`, `BAHouse`, `BAPath`, and `BACycle`. Their default construction parameters follow the supplementary. Smaller sizes may be supplied for development.

## Ablations

All training ablations call the shared SAE trainer. Full SAE-GNNExplainer is not duplicated in the ablation module.

```powershell
graphsae ablate --train train.pt --evaluation val.pt --head head.pt --config configs/graphsae_fixed.json --study losses --output artifacts/loss_ablations
graphsae ablate --train train.pt --evaluation test.pt --head head.pt --config configs/graphsae_fixed.json --study sensitivity --output artifacts/sensitivity
graphsae ablate --train train.pt --evaluation val.pt --head head.pt --config configs/graphsae_fixed.json --study sensitivity --sensitivity-scope joint --output artifacts/joint_sensitivity
graphsae ablate --train train.pt --evaluation val.pt --head head.pt --config configs/graphsae_fixed.json --study encoders --output artifacts/encoders
graphsae tune-sae --train train.pt --validation val.pt --l1 --output artifacts/l1_path
graphsae ablate-explanations --factory examples.toy:load --sae artifacts/toy/sae_seed_0.pt --study pure-path --output artifacts/pure_path
graphsae ablate-explanations --factory examples.toy:load --sae artifacts/toy/sae_seed_0.pt --study signed --budgets 8 12 16 --output artifacts/signed
```

Loss variants remove reconstruction, grounding, faithfulness, or coherence independently, or retain reconstruction alone. Sensitivity changes individual weights, dictionary width, and the joint grounding/faithfulness grid. Encoder variants include BatchTopK, JumpReLU, Gated, and ReLU. The dedicated L1 selection follows the supplementary's four-coefficient search and active-feature preference; its selected coefficient can then be reused with `refit_seeds` and the common path mapping. Signed evidence uses matched total node budgets with deterministic shortage handling.

Training ablations share a deterministic population of at most 2,048 training representations. Evaluation is capped separately at 2,048 rows and never changes configurations. Run single-factor sensitivity on the test pack and joint-weight sensitivity on the validation pack; [METHOD.md](METHOD.md) specifies the configuration filters. The complete fixed reference can be trained once through `train-sae` on the same `subset_pack(train, 2048, seed=0)` population.

## Custom predictors and heterogeneous graphs

`FrozenModelAdapter` accepts differentiable target-representation and predictor functions, an eligible-source function, and optional encoded-input/deletion callbacks. It preserves typed node IDs and relation types. `examples/heterogeneous.py` is a runnable relation-aware example:

```powershell
python -m examples.heterogeneous
```

Included backbones use ReLU and mean graph pooling. Custom adapters preserve their predictor's own readout, activations, normalization, and typed structure. Standard homogeneous/bipartite GAT layers use frozen observed attention during attribution. Shared recurrent GAT modules and edge-conditioned/custom attention require an explicit attention context. Deletion always replays the ordinary predictor with its normalization, means, attention, and gates recomputed.

Reproducing a particular reported dataset/predictor case requires its original preprocessing, split, frozen architecture, and validation target cohort. The examples provide runnable workflows and adapters without distributing those experiment artifacts.
