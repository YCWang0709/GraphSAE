# Cora with pretrained checkpoints

This example loads the four frozen Cora predictors and their validation-selected SAE checkpoints, constructs explanations, and evaluates source-node deletion. It also renders explanations within the target's one-hop neighborhood. The example contains inference and evaluation code; no model or SAE fitting is performed.

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

These values are the reference for `--check-paper`, which compares them to the rounded metrics computed by the run. They are not substituted for computed outputs. A mismatch produces a nonzero exit status and retains the per-target results for inspection.

For a shorter installation check:

```powershell
python examples/cora_pretrained/run.py --models GCN --seeds 0 --target-limit 2
```

This is labeled a partial cohort and does not reproduce the paper aggregate. `--check-paper` requires all four backbones, all three seeds, and the full archived target lists.

## Checkpoints and evaluation population

`checkpoints/` contains four backbone state dictionaries and 12 SAE state dictionaries. Each SAE checkpoint contains its trained encoder/decoder, training mean and standard deviation, active-feature count, SAE layer, and seed. Checkpoints contain tensors and the metadata needed for inference, without training histories or saved explanation masks.

`manifest.json` records checkpoint file/tensor checksums, the input-tensor checksum, architecture parameters, fixed validation/test target IDs, and the validation-selected extraction rule. The normalized inputs and public split masks are checked before evaluation. The program reads the checkpoint's serialized normalizer and does not fit a new one.

| Backbone | Hidden width | Message-passing layers | SAE active features | Selected extraction rule |
| --- | --- | --- | --- | --- |
| GCN | 128 | 2 | 16 | First nonpositive probability-gain stop |
| GIN | 128 | 2 | 16 | 12 greedy additions, capped by available sources |
| GAT | 256 | 3 | 4 | First nonpositive probability-gain stop |
| GraphSAGE | 256 | 3 | 32 | First nonpositive probability-gain stop |

Every backbone uses its original 40 test targets and SAE seeds 0, 1, and 2: 480 seed-target evaluations in total. The aggregate first averages seeds within each target, then targets within a backbone, and finally weights the four backbones equally. The selected rule and SAE configuration are fixed before test evaluation; there is no fitting or parameter search in this example.

## Explanation and deletion

For each target, the program encodes its final representation with the frozen SAE. Active decoder directions are weighted by activation and their alignment with the frozen predicted-class gradient. A transpose of the observed local message-passing map projects this direction onto source-node inputs. GAT attention is fixed at its original-forward values during attribution.

Candidates are the target's message-passing receptive field, excluding the target. The 32 highest ranked eligible sources form the greedy search pool; opposite-sign scores are clipped to zero for ranking, and ties use node ID. Each greedy step queries the same frozen predictor after zeroing selected source inputs and removing incident edges. The archived rule determines when extraction stops. Node count is variable, so 8.07 is a mean rather than a fixed budget of eight.

First-layer affine projections are cached for repeated queries. GAT and GraphSAGE queries use their exact incoming dependency fields, with near ties replayed through the full cached predictor. Every final mask is evaluated independently by the original full predictor using raw input deletion; these full-predictor values supply the reported metrics. The program can stop its computation once the selected greedy prefix is determined, without evaluating unused later prefixes.

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

Generated outputs are written outside the example's checkpoint directory and are ignored by Git. `--skip-visualization` runs evaluation without rendering figures. Evaluation uses CPU float32 with one PyTorch thread, matching the deterministic replay protocol. The checkpoint replay is verified with PyTorch 2.12.0 and PyTorch Geometric 2.8.0; low-order floating-point differences can depend on the software and CPU stack.
