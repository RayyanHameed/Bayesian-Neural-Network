# Bayesian Neural Networks for Image Classification

**Author:** Rayyan Hameed

## Overview

Standard neural networks learn a single point estimate for every weight, producing confident predictions with no indication of how certain the model actually is. This is a serious limitation in safety-critical settings — autonomous driving, medical imaging, financial forecasting — where knowing *how sure* a model is can matter as much as the prediction itself.

This project implements a **Bayesian Neural Network (BNN)** trained on FashionMNIST using **Variational Inference**. Instead of a single value per weight, each weight is represented by a Gaussian variational posterior `q(θ) = N(μ, σ²)`. The network is trained to make this posterior approximate the true (intractable) weight posterior by maximising the **Evidence Lower Bound (ELBO)**, which is used as the training loss. At test time, Monte Carlo sampling from this posterior yields both a prediction and a principled uncertainty estimate, decomposed into **epistemic** (model/weight uncertainty, reducible with more data) and **aleatoric** (irreducible data noise) components. Performance and calibration are benchmarked against a standard neural network of identical architecture trained with plain cross-entropy.

## Method

### Bayesian linear layer

Each `BayesianLinear` layer replaces a single weight matrix `W` and bias `b` with four learnable tensors of the same shape: `w_mu`, `w_rho`, `b_mu`, `b_rho`. The standard deviation is derived from the raw parameter via the softplus function, `σ = log(1 + e^ρ)`, keeping it strictly positive while maintaining smooth gradients.

At each forward pass, weights are sampled using the **reparameterisation trick**:

```
θ = μ + σ ⊙ ε,   ε ~ N(0, I)
```

which lets gradients flow through `μ` and `σ` despite the sampling step being non-differentiable. Each layer also computes the closed-form KL divergence between its variational posterior and the prior `p(θ) = N(0, σ_prior²)`, summed across all weights and biases.

### Training objective

The network minimises the negative ELBO:

```
L = CrossEntropy(y, ŷ) + β · (1/N) · KL(q(θ) ‖ p(θ))
```

where `N` is the dataset size (so the mini-batch loss is an unbiased estimator of the full-dataset ELBO) and `β` is a **KL annealing weight**, ramped linearly from 0 to 1 over the first 5 epochs. Without annealing, the KL term can collapse the posterior towards the prior before the network has learned anything useful from the data.

### Architecture

A 4-layer network shared by both models for a fair comparison:

```
Input (784) → Hidden 1 (200, ReLU) → Hidden 2 (80, ReLU) → Output (10, softmax)
```

- **BNN**: every layer is a `BayesianLinear` layer; prior `p(θ) = N(0, 4)` (σ_prior = 2.0).
- **Baseline NN**: identical shape, standard `nn.Linear` layers, trained with cross-entropy alone (no KL term).

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Batch size | 128 |
| Epochs | 10 |
| KL warm-up | 5 epochs |
| Prior σ | 2.0 |
| MC samples (test time) | 50 |

### Test-time inference and uncertainty

The posterior predictive distribution `p(y*|x*, D) = ∫ p(y*|x*, θ) p(θ|D) dθ` is intractable, so it's approximated with Monte Carlo sampling: 50 weight samples are drawn from `q(θ)`, each run through the network, and the resulting softmax outputs are averaged for the final prediction. The spread across these 50 predictions gives uncertainty estimates for free, decomposed as:

- **Predictive entropy** `H[p̄]` — total uncertainty in the averaged prediction.
- **Expected entropy** `E[H]` — average entropy of the individual samples (**aleatoric** uncertainty, irreducible).
- **Mutual information** `I = H[p̄] − E[H]` — the gap between the two (**epistemic** uncertainty, reducible with more data/training).

Calibration is measured with **Expected Calibration Error (ECE)**: the weighted average gap between a model's confidence and its actual accuracy across confidence bins, where 0 is perfect calibration.

## Results

Evaluated on the FashionMNIST test set (10,000 images):

| Model | Test Accuracy | ECE |
|---|---|---|
| Baseline NN | 88.2% | 0.016 |
| BNN (MC, 50 samples) | 86.0% | 0.019 |

- The **2.2 percentage-point accuracy gap** is the expected cost of the KL regularisation pulling the posterior towards the prior — a modest price for gaining principled uncertainty estimates.
- Both models are **well calibrated** (ECE close to 0), with the baseline slightly better — a somewhat surprising result, likely because FashionMNIST is clean enough that the baseline doesn't encounter enough ambiguous inputs to become overconfident.
- **Per-class accuracy correlates inversely with epistemic uncertainty**: visually similar/ambiguous classes like *Shirt* (57% accuracy, 0.087 epistemic uncertainty) and *Pullover* (71%, 0.070) are both harder to classify and where the BNN is most uncertain about its weights, while structurally distinct classes like *Trouser* (96%, 0.012) and *Bag* (95%, 0.025) show both high accuracy and low uncertainty — the model is correctly identifying where it doesn't know.
- **Aleatoric uncertainty dominates epistemic uncertainty** across the test set, indicating that most of the difficulty on FashionMNIST comes from genuinely ambiguous images (e.g. Shirt vs. Pullover vs. Coat) rather than from the model being unsure of its own weights — this is an irreducible source of error that more training data would not fix.

Full derivations, figures (per-class breakdown, reliability diagrams, uncertainty distributions), and discussion are in the accompanying report PDF.

### Pros and cons of the Bayesian approach

**Advantages**
- Principled, decomposed uncertainty (epistemic vs. aleatoric) rather than a single confidence score.
- Comparable accuracy to the baseline (within ~2%) despite the added complexity.
- The KL term acts as a theoretically grounded regulariser, without needing dropout or explicit weight decay.

**Disadvantages**
- **2× the parameters** of an equivalent standard network (a mean and a scale per weight/bias).
- **~50× the inference cost** — 50 forward passes per prediction instead of one, which may be prohibitive for real-time use.
- **Mean-field approximation**: the variational posterior assumes all weights are independent Gaussians, which the true posterior is unlikely to satisfy — this can lead to underestimated uncertainty.
- Sensitive to the choice of prior; in this project the KL term ended up dominating the likelihood term during training, suggesting the regularisation may be stronger than ideal relative to the data fit.

## Repository Structure

```
├── NN_and_DL_project_code_M_R_H.ipynb              # Full implementation: data loading, BayesianLinear
│                                                      # layer, BNN & baseline models, training loops,
│                                                      # Monte Carlo prediction, calibration & uncertainty analysis
└── Neural_Networks_and_Deep_Learning_projectMRH.pdf  # Full written report
```

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch, torchvision
- numpy, matplotlib, seaborn, tqdm

```bash
pip install torch torchvision numpy matplotlib seaborn tqdm
```

### Running

```bash
git clone <this-repo-url>
cd <repo-directory>
jupyter notebook NN_and_DL_project_code_M_R_H.ipynb
```

FashionMNIST is downloaded automatically via `torchvision.datasets.FashionMNIST` the first time the notebook is run (no manual dataset setup required). Run the notebook top to bottom to train both the baseline and Bayesian models and reproduce all results, plots, and the calibration/uncertainty analysis.

**Note on reproducibility:** a fixed seed (`seed=42`) is used, but due to the stochastic nature of SGD and Monte Carlo sampling, exact accuracy/ECE figures may vary slightly between runs. The numbers in the report reflect one representative run.

## References

- Bishop, C. M. *Pattern Recognition and Machine Learning.* Springer, 2006.
- Blundell, C. et al. *Weight Uncertainty in Neural Networks.* arXiv:1505.05424, 2015.
- Gal, Y. *Uncertainty in Deep Learning.* PhD thesis, University of Cambridge, 2016.
- Xiao, H., Rasul, K., Vollgraf, R. *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* arXiv:1708.07747, 2017.
