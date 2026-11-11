# Cora with pretrained checkpoints

This example loads four frozen Cora predictors and their validation-selected SAE checkpoints, constructs explanations, evaluates source-node deletion, and renders one-hop explanation subgraphs.

## Run

From the repository root:

```powershell
python -m pip install -r examples/cora_pretrained/requirements.txt
python examples/cora_pretrained/run.py --check-paper
```

The first invocation downloads the public Planetoid Cora dataset through PyTorch Geometric. Subsequent invocations reuse it. `--data-root` selects its cache directory, and `--output` selects the output directory. The default output directory is `artifacts/cora_pretrained`.

The complete run recomputes the following published row:

| Method | Target-class logit decrease | Target-class probability decrease | Mean selected source nodes |
| --- | --- | --- | --- |
| SAE (Ours) | 7.08 | 0.51 | 8.07 |

`--check-paper` compares the rounded computed metrics with this row. A mismatch produces a nonzero exit status and retains the per-target results.

For a shorter installation check:

```powershell
python examples/cora_pretrained/run.py --models GCN --seeds 0 --target-limit 2
```

This command evaluates two GCN targets at seed 0. To reproduce the paper row, use all four backbones, all three seeds, and the full saved target lists with `--check-paper`.

## Checkpoints and evaluation population

`checkpoints/` contains four backbone state dictionaries and 12 SAE state dictionaries. Each SAE checkpoint contains its trained encoder/decoder, training mean and standard deviation, active-feature count, SAE layer, and seed.

`manifest.json` records checkpoint file/tensor checksums, the input-tensor checksum, architecture parameters, fixed validation/test target IDs, and the validation-selected extraction rule. Evaluation checks the normalized inputs and public split masks and uses the checkpoint's saved normalizer.

| Backbone | Hidden width | Message-passing layers | SAE active features | Selected extraction rule |
| --- | --- | --- | --- | --- |
| GCN | 128 | 2 | 16 | First nonpositive probability-gain stop |
| GIN | 128 | 2 | 16 | 12 greedy additions, capped by available sources |
| GAT | 256 | 3 | 4 | First nonpositive probability-gain stop |
| GraphSAGE | 256 | 3 | 32 | First nonpositive probability-gain stop |

Every backbone uses 40 saved test targets and SAE seeds 0, 1, and 2: 480 seed-target evaluations in total. The aggregate averages seeds within each target, targets within each backbone, and then the four backbones equally. The run loads the selected extraction rule and SAE configuration from the manifest and checkpoints.

## Explanation and deletion

For each target, the program encodes its final representation with the frozen SAE. Active decoder directions are weighted by activation and their alignment with the frozen predicted-class gradient. A transpose of the observed local message-passing map projects this direction onto source-node inputs. GAT attention is fixed at its original-forward values during attribution.

Candidates are the target's message-passing receptive field, excluding the target. The 32 highest ranked eligible sources form the greedy search pool; opposite-sign scores are clipped to zero for ranking, and ties use node ID. Each greedy step queries the frozen predictor after zeroing selected source inputs and removing incident edges. The saved extraction rule determines when extraction stops, producing a variable node count with mean 8.07.

First-layer affine projections are cached for repeated queries. GAT and GraphSAGE queries use incoming dependency fields, with near ties replayed through the full cached predictor. Final metrics use the full predictor with raw input deletion. Extraction stops when the selected greedy rule terminates.

## One-hop visualization

The full run writes a PNG and an SVG for the first declared target of each backbone, using the first requested SAE seed. To visualize a single explanation directly:

```powershell
python examples/cora_pretrained/visualize.py --model GCN --seed 0 --target-node 1711
```

Each figure has two panels: the original one-hop neighborhood and the target plus selected sources that lie within that neighborhood. The gold star marks the target, selected-source fills indicate class, and gray nodes provide neighborhood context. Original node IDs are retained. The title gives both the full explanation size and the number visible within one hop.

The one-hop restriction applies to the figure. Deletion metrics use the complete selected source set from the full message-passing field, including sources outside the displayed neighborhood.

## Generated files

- `metrics.csv`: freshly evaluated metrics for every model/seed/target.
- `explanations.jsonl`: selected nodes, source scores, extraction rules, and prediction-query counts.
- `summary.json`: per-backbone means, the aggregate, evaluation scope, and the paper comparison.
- `figures/*_1hop.png` and `figures/*_1hop.svg`: one-hop explanations.

Outputs are written to `--output` (default `artifacts/cora_pretrained`). Use `--skip-visualization` for metrics and explanation records. Evaluation uses CPU float32 with one PyTorch thread; the checkpoint environment uses PyTorch 2.12.0 and PyTorch Geometric 2.8.0.
