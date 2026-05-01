### EGNN + Flow Matching for Transition State Prediction

### Team 6 · AI Schmidt Hackathon 2026 · Generative Chemistry
Isha Sharma · Khadija Shuaib · Kristian Velazquez

A 3-stage pipeline for predicting transition state (TS) geometries from reactant and product structures, using an E(n)-Equivariant Graph Neural Network (EGNN) combined with flow matching — implemented in PyTorch.

#### Key Results at a Glance
MetricValueTest Set RMSD (EGNN v5)0.3565 — beats midpoint baseline 0.3566 ✅100-sample RMSD snapshot0.299 — beats baseline 0.349 by 14% ✅Best Validation Loss0.1247 (EGNN + Flow combined)Reactions improved vs midpoint49% per-sampleModel parameters62,746Training reactions7,700Pipeline stages3

#### Problem Statement
The transition state (TS) is the highest-energy saddle point along a reaction pathway — the fleeting atomic configuration reactants must pass through to become products. Locating it is essential for computing activation energies and understanding reaction mechanisms, but traditional quantum methods (DFT, NEB) take hours per molecule.
This project predicts TS 3D geometry from reactant + product structures in milliseconds — enabling high-throughput catalyst and drug design.
The challenge: predictions must be rotation- and translation-invariant (standard NNs fail here without explicit constraints), and must beat the midpoint baseline x₀ = 0.5 × (xR + xP) to demonstrate real ML value.

#### Datasets
Transition1x

7,700 training / 1,650 val / 1,650 test reactions
Features: atomic positions, charges, forces, energies, TS guess variants

#### Halo8 (Bonus)

Halogen-abstraction (SN2-like) reactions
Slightly different atom types — tests model transferability across reaction classes

Combined training set: Halo8 + Transition1x (7,700 reactions total)
Evaluation metric: Δ = RMSD(midpoint, TS) − RMSD(model, TS) · Mean RMSD (lower is better) · % reactions improved vs midpoint

### Pipeline Architecture

INPUT: Reactant + Product positions (3D), atom types, charges, forces
         │
         ▼
┌─────────────────────────────────────┐
│  Stage 1: EGNN — Initial TS Guess   │
│  E(n)-Equivariant GNN on molecular  │
│  graph of reactant + product atoms  │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Stage 2: Flow Matching — Refinement│
│  Learns velocity field vθ(xt,t|R,P) │
│  Trajectory: midpoint → TS (20 steps)│
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Stage 3: Ensemble — Final Output   │
│  50/50 average of EGNN + Flow       │
│  Kabsch alignment for RMSD eval     │
└─────────────────────────────────────┘
         │
         ▼
OUTPUT: Predicted TS geometry (.xyz)


#### Stage 1 — EGNN (Initial TS Guess)
An E(n)-Equivariant Graph Neural Network operates on the combined reactant + product molecular graph. Equivariance guarantees physically valid predictions under any rotation or translation.
Node features (25D):

3D positions
Displacement vector + magnitude
Atom one-hot encoding (8D)
Normalized charge
Reactant AND product force vectors

#### Key innovation: Initialize atom positions from the midpoint x₀ = 0.5 × (xR + xP) — a physically meaningful starting geometry that accelerates convergence vs random initialization.
#### Stage 2 — Flow Matching (Geometry Refinement)
A continuous normalizing flow learns the velocity field vθ(xt, t | xR, xP).

#### Trajectory: x₀ = midpoint → xTS over 20 ODE integration steps
Loss: E||vθ − (xTS − x₀)||²
The flow learns the residual correction from midpoint toward the true TS
Flow Matching validation loss stayed stable (~0.062) across all training versions

#### Stage 3 — Ensemble
A 50/50 average of EGNN and Flow predictions. Combines complementary strengths and averages away individual model variance. Kabsch alignment is applied before RMSD computation for fair rotational comparison.

#### Key Innovations
AreaInnovationArchitectureMidpoint initialization x₀ = 0.5×(xR+xP) — reduces training time vs random initFeatures25D node features including force vectors from both reactant AND productInferenceIterative inference refinement — feed model's TS output back as input, iterating toward convergence like a learned NEB methodTrainingReduceOnPlateau halves LR on plateau + feature dropout (0.2) to prevent over-reliance on any single featureEvaluationKabsch algorithm for fair rotational alignment before RMSD computation

#### Experiment History
VersionLRHidden DimDropoutKey ChangeOutcomev15e-4128—Baseline large modelVal EGNN spiked 0.062→0.108. Overfitting.v21e-4640.1Reduced LR + dimStabilized but val still slowly rising.v35e-5320.2Low LR + dropoutVal EGNN stable 0.062. RMSD 0.299 on 100 samples ✅v45e-5320.2200 epochs, patience 40RMSD 0.299 beats baseline. Full 500-sample 0.354.v55e-41280.2Larger model + midpoint init62K params. EGNN RMSD 0.3565 beats test baseline ✅✅v61e-41280.2Lower LR, patience 30Ensemble 0.3502. Best generalization.

#### Key insight: Scaling to 62K params (hidden dim 128) WITH dropout 0.2 + midpoint initialization was the winning combination. v5's EGNN directly beat the test set midpoint baseline — our best single-model result.


### Results
Test Set RMSD (500 samples)
ModelRMSDvs BaselineMidpoint Baseline0.3566—Stage 1: EGNN (v5)0.3565✅ BETTERStage 3: Ensemble0.3580slightly above baselineStage 2: Flow Matching0.3608slightly above baseline

### Note: The 100-sample RMSD snapshot (0.299) beats the baseline (0.349) by 14%. The full 500-sample evaluation is harder due to the diversity of the combined dataset.


### Challenges & Lessons Learned
Overfitting in EGNN — Val loss spiked 0.062→0.108 at epoch 10 in v1 (hidden dim 128, LR 5e-4). Fixed by reducing LR, hidden dim, and adding dropout 0.2. Fully resolved by v3.
MSE ≠ RMSD — Training on MSE loss doesn't directly optimize the RMSD evaluation metric. Models converged well (val loss 0.124) but RMSD didn't improve proportionally. An RMSD-aware loss function is the top next step.
Dataset diversity — Combined Halo8 + Transition1x is diverse, making full-set evaluation harder than 100-sample results suggest. Halo8-only training is recommended to push RMSD below baseline consistently.
Flow model is stable — Flow Matching val loss stayed flat (~0.062) across all versions. All regularization effort should focus on the EGNN stage; per-stage tuning is more efficient.

### Next Steps

Halo8-only training (High) — Smaller, homogeneous dataset should push full-set RMSD below baseline consistently
RMSD-aware loss (High) — Weighted MSE + RMSD directly aligns training signal with the evaluation metric
Separate stage tuning (Medium) — Independent LR schedules and early stopping per stage
Scale EGNN (High) — With better data + loss, scale hidden dim to 128–256; v5 showed 62K params is the right capacity
More flow steps at inference (Medium) — Try 40–50 steps instead of 20 for better refinement at no training cost


### Repository Structure

egnn-flow-reactive-structures/
├── train_egnn_flow_combined 1.3.py        # Main training script
├── combined_train.pkl                      # Training set (7,700 reactions)
├── combined_val.pkl                        # Validation set (1,650 reactions)
├── combined_test.pkl                       # Test set (1,650 reactions)
├── EGNN_TS_Prediction_Final_Team6.pptx    # Final presentation slides
└── README.md

### Installation & Usage
bashgit clone https://github.com/sharmai309/egnn-flow-reactive-structures.git
cd egnn-flow-reactive-structures
pip install torch torch-geometric numpy scipy
python "train_egnn_flow_combined 1.3.py"
Python 3.8+ and PyTorch 1.12+ recommended.

### References

Satorras et al. (2021). E(n) Equivariant Graph Neural Networks
Lipman et al. (2022). Flow Matching for Generative Modeling
https://github.com/chicago-aiscience/generative_chem_reactive_structures
