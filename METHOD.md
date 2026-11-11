# Method and supplementary protocol

## Representation and training

The SAE input is a final node representation or the predictor's graph readout. Training statistics define `s_tilde = (s - mean) / std`; validation/test inputs reuse the same statistics. `ActivationPack` contains raw representations, not pre-normalized values.

`sae.py` implements nonnegative Top-K codes, linear decoding, and the objective

```text
L = lambda_rec * MSE(s_tilde, reconstructed_s_tilde)
  + ramp(epoch) * (lambda_ground * grounding_entropy
                + lambda_faith * prediction_faithfulness
                + lambda_coh * decoder_coherence)
```

Grounding sums each feature's activation mass over training groups, adds epsilon to group masses, and averages group entropy over active features. Minibatch training uses the vocabulary of all training groups, including groups absent from a minibatch. Classification groups are training labels. Regression groups are `floor(log2(1 + count))`: original stored incoming-edge counts for city-roads-M and total nodes across all graph node types for RelSC. Continuous regression responses are never converted into grounding classes.

Faithfulness is `KL(p(original) || p(reconstructed))` for classification and output MSE for regression. A scalar binary logit is represented as `[0, logit]`. The prediction head is copied, frozen, and evaluated on de-standardized reconstructions. Its reference branch is detached, while gradients through the reconstructed branch reach the SAE. Coherence is the mean squared off-diagonal cosine similarity of decoder columns; an equivalent input-width Gram calculation avoids allocating a dictionary-width square matrix.

The fixed controlled configuration in supplementary II-C/V uses Adam, learning rate 0.001, 100 epochs, expansion 8, K=16, batch size 512, unit decoder columns after every optimizer update, and SAE seeds 0/1/2. Reconstruction, grounding, faithfulness, and coherence weights are 1/0.01/0.1/0.01. The auxiliary ramp is `min(1, epoch/5)` for epochs numbered from one. Magnitude and stability weights are zero. The optional general-purpose stability penalty is disabled in these experiments.

## Source contributions and projection

`contributions.py` initializes each source with its input vector and propagates only that source term through the observed local linear map. GCN uses the original normalized adjacency; mean-GraphSAGE uses separate neighbor and root transforms; GIN uses the local MLP Jacobian, including internal gates; GAT retains original attention coefficients and head concatenation/averaging. Bias and local affine remainders are excluded from source terms. No deleted-graph quantities enter attribution.

For node targets, contributions end at the target node. For graph targets in the included backbones, contributions are averaged over final graph nodes. Candidate fields are directed incoming-message receptive fields with the node target excluded, or all nodes of the selected graph. Typed adapters retain their original readout and relation structure.

The default score is

```text
a_tau = sum(active feature k) z_tau[k] * normalized_decoder[:, k]
S_tau[u] = dot(c_u_tau / train_std, a_tau)
```

`source_scores` evaluates the same score through a reverse projection: if `L` is the frozen local map, `<L(e_u x_u), a>` equals `<x_u, [L^T a]_u>`. This avoids a source-by-node Jacobian. The explicit source-vector routines remain available for inspection and numerical verification. The full graph supplies normalization degrees and boundary messages.

Prediction-weighted scoring multiplies each active coefficient by the dot product between its normalized decoder direction in raw coordinates and the frozen head's original predicted-class gradient. Regression uses the scalar-output gradient. Source scores describe representation alignment; their sign does not guarantee a prediction change under deletion.

## Extraction and selection

`policies.py` implements fixed-budget positive, negative, and mixed evidence. Positive/negative rankings clip the opposite sign to zero and pad by node ID. Mixed evidence splits the budget between signs and fills shortages by remaining absolute score. All return exactly `min(B, eligible_count)` distinct sources. Decoder feature count K and input-node budget B are independent.

Native extraction supports strictly positive support, relative score thresholds, positive mass fractions, positive ranked prefixes, and topology-based one-hop masks. Greedy local refinement adds/removes scored sources and their one-hop neighbors. Block refinement uses score sign/quantiles, direct-neighbor membership, high-score prefixes, and degree strata; it accepts at most eight improving moves and stops at improvement at most 1e-6. RelSC-style local refinement uses up to 32 steps. The predictor and SAE remain frozen during these queries. `stable_sage_deletion` provides direct float64 surviving-neighbor accumulation followed by the original float32 modules for large mean-GraphSAGE replay.

Supplementary II-C/III-E distinguishes validation-selected pipelines from fixed-configuration ablations. `tune_pipeline` jointly selects training and extraction settings using only declared validation targets. Fixed-budget search averages the same selected ranking across B=8/12/16. The validation objective and its recovery/size coefficients are explicit inputs and are recorded with the selected configuration. One policy supplies all metrics for a target; individual metrics are not selected independently. After selection, `refit_seeds` trains the fixed configuration at seeds 0/1/2. Test inference is a separate operation.

The supplied search grid and extraction-policy lists are reference spaces, not saved dataset-specific selections. Dataset-specific target cohorts and search ranges can be supplied through experiment factories/configurations. `Experiment` checks train/validation/test target disjointness. Node factories must retain original normalization context; graph targets must carry stable graph IDs.

## Deletion, feature ablation, and metrics

Input deletion zeros selected feature vectors and removes every incident edge while retaining all vertex slots, original IDs, and batch assignments. Typed deletion applies the same operation across every relation. Predictor replay recomputes normalization, neighborhood means, attention, and activations. Classification reports the original predicted-class logit/probability decrease; regression reports absolute output change. Recovery uses node-set precision/recall/Jaccard with zero for undefined denominators.

Feature ablation zeros selected target SAE coefficients and patches only the target representation. The reported coefficient effect compares the ablated reconstruction to the unablated reconstruction, using the class predicted by the original frozen model. This separates a feature's removal from SAE reconstruction error.

Main-paper feature alignment uses top-50 class purity for nodes and majority class among active graphs for graphs. The 128 most active features are retained, including inactive features with zero purity if needed. Controlled sensitivity uses Purity@5 instead: at least three training activations and five evaluation activations above 1e-8 are required. `representation_metrics` also reports standardized MSE, NMSE divided by evaluation coordinate variance, prediction agreement, classification KL or regression output MSE, and decoder coherence.

## Ablation code

`ablations.py` contains configuration changes and the pure-path score only. Training and SAE-GNNExplainer are imported from the main implementation.

| Study | Configuration / implementation |
| --- | --- |
| Individual loss removal | Set one of the four coefficients to zero |
| Reconstruction only | Set grounding/faithfulness/coherence to zero |
| Pure path | `Q(u) = dot(c_u / std, (s - mean) / std)`; native support keeps Q>0 |
| Signed evidence | Shared scoring and matched budgets 8/12/16 |
| Individual weight sensitivity | 0.1, 0.3, 1, 3, 10 times each auxiliary default |
| Dictionary sensitivity | Expansions 2, 4, 8, 16 |
| Joint weights | 0.1, 1, 10 times grounding and faithfulness defaults |
| L1+Path | ReLU codes, reconstruction plus decoder-norm-weighted activation penalty |

`sensitivity_configurations()` returns changed settings without duplicating Full. Evaluate names beginning `joint_` on validation and other names on test. Reuse a separately trained Full configuration when forming comparisons. The deterministic 2,048-row training subset and train-fitted normalizer are identical across a matched training study. Validation/test populations are capped independently at 2,048; they never affect the training population or statistics.

Supplementary V-A L1+Path uses expansion 8, Adam 0.001, batch size 512, and 100 epochs. `tune_l1` tests coefficients 0.001/0.01/0.1/1 at seed 0. Candidates with 12--20 mean active features are preferred by validation NMSE; otherwise it minimizes absolute log-ratio to 16, with NMSE breaking ties. Refit the chosen coefficient at seeds 1/2 without reselection. Both L1 and Top-K explanations use the same source-to-decoder path projection and B=12 for the sparse-encoder comparison.

Additional encoder controls keep the shared GraphSAE losses configurable: BatchTopK uses a batch-wide active budget and a training-calibrated inference threshold; JumpReLU uses rectangle-kernel threshold/L0 estimators; Gated uses tied gate/magnitude directions, gate sparsity, and detached-decoder auxiliary reconstruction. Their mechanisms follow the [BatchTopK paper](https://arxiv.org/abs/2412.06410), [JumpReLU paper](https://arxiv.org/abs/2407.14435), and [Gated SAE paper](https://arxiv.org/abs/2404.16014). Encoder-specific sparsity coefficients remain configurable. The dedicated supplementary L1 baseline disables the GraphSAE auxiliary objectives.

## Data and predictor interfaces

Synthetic generators retain the specified seeds, structural features, motif attachments, and stratified splits. Graph datasets keep planted node/edge masks separate from inputs. B-XAIC reference masks should be read from the task's annotation field in `explanations.sdf` and passed through `sdf_reference_mask` with original atom indexing. MUTAG proxy rules select terminal nitrogen for NH2 and degree-three nitrogen with at least two terminal oxygen neighbors for NO2. A supplied structural mask and a constructed proxy are distinct reference sources.

Included backbones expose final node/mean-pooled graph representations and per-layer local-forward quantities. They are configurable reference predictors, not replacements for a dataset's selected frozen architecture. Custom predictor adapters expose their own differentiable representations, candidate fields, and native input-deletion functions. Encoded-input models can supply encoded source tensors before fixed biases. Heterogeneous inputs retain all node/relation types. The Cora pretrained example supplies the selected frozen architectures, four backbone weights, 12 SAE weights, and the fixed target/extraction manifest for its paper row. Its source scores, explanation masks, and deletion metrics are computed at runtime.
