# Advanced Learning Paradigms: Complete Guide
**Topics 2.3, 2.4, and 2.5: Semi-Supervised, Self-Supervised, and Other Learning Paradigms**  
*Comprehensive guide with detailed mathematical derivations and step-by-step explanations*

---

## Table of Contents
1. [Semi-Supervised Learning (2.3)](#1-semi-supervised-learning-23)
2. [Self-Supervised Learning (2.4)](#2-self-supervised-learning-24)
3. [Other Learning Paradigms (2.5)](#3-other-learning-paradigms-25)
   - [Multi-Model Learning](#31-multi-model-learning)
   - [Active Learning](#32-active-learning)
   - [Transfer Learning](#33-transfer-learning)
   - [Federated Learning](#34-federated-learning)
   - [Meta-Learning](#35-meta-learning)
4. [Comparison & Integration](#4-comparison--integration)
5. [Best Practices & Applications](#5-best-practices--applications)

---

## 1. Semi-Supervised Learning (2.3)

### 1.1 Intuition & Goal

**Goal**: Leverage both labeled and unlabeled data to improve learning when labeled data is scarce.

**Real-World Problem**:
- Labeling data is expensive and time-consuming
- Often have lots of unlabeled data but few labeled examples
- Example: Medical diagnosis (few labeled cases, many unlabeled patient records)

**Key Insight**: Unlabeled data contains valuable information about the data distribution that can guide learning!

### 1.2 Core Concepts

#### **The Data Landscape**

**Labeled Data (L)**:
- Small set: {(x₁, y₁), (x₂, y₂), ..., (xₗ, yₗ)}
- Expensive to obtain
- Provides direct supervision

**Unlabeled Data (U)**:
- Large set: {xₗ₊₁, xₗ₊₂, ..., xₗ₊ᵤ}
- Cheap and abundant
- Contains distributional information

### 1.3 Mathematical Foundation: Combined Loss Function

#### **Step 1: Define Supervised Loss**

For labeled data, we use standard supervised loss:

```
L_supervised = (1/|L|) × Σ_{(x,y) ∈ L} l(f(x), y)
```

**Explanation**:
- **|L|**: Number of labeled examples
- **l(f(x), y)**: Loss function (e.g., cross-entropy for classification, MSE for regression)
- **f(x)**: Model prediction
- **y**: True label
- We average the loss over all labeled examples

**For Classification (Cross-Entropy)**:
```
l(f(x), y) = -Σᵢ yᵢ × log(f(x)ᵢ)
```

**For Regression (MSE)**:
```
l(f(x), y) = (f(x) - y)²
```

#### **Step 2: Define Unsupervised Loss**

For unlabeled data, we need a loss that doesn't require labels. Common approach: **consistency regularization**.

**Consistency Regularization Principle**:
- Model should make consistent predictions for similar inputs
- If we perturb input slightly, prediction should remain similar

**Mathematical Formulation**:
```
L_unsupervised = (1/|U|) × Σ_{x ∈ U} D(f(x), f(x'))
```

**Explanation**:
- **|U|**: Number of unlabeled examples
- **x'**: Augmented or perturbed version of x
- **D(·, ·)**: Distance metric between predictions
- **f(x)**: Prediction on original input
- **f(x')**: Prediction on perturbed input
- We want these predictions to be similar

#### **Step 3: Distance Metric D**

**Option 1: Mean Squared Error (MSE)**:
```
D(f(x), f(x')) = ||f(x) - f(x')||²
```

**Step-by-step**:
1. **f(x)**: Vector of predictions for original input x
2. **f(x')**: Vector of predictions for perturbed input x'
3. **f(x) - f(x')**: Element-wise difference
4. **||·||²**: Squared L2 norm (sum of squared differences)
5. **Result**: Measures how different the predictions are

**For classification with probabilities**:
```
D(f(x), f(x')) = Σᵢ (P(yᵢ|x) - P(yᵢ|x'))²
```

**Option 2: KL Divergence**:
```
D(f(x), f(x')) = KL(P(y|x) || P(y|x'))
                = Σᵢ P(yᵢ|x) × log(P(yᵢ|x) / P(yᵢ|x'))
```

**Step-by-step derivation**:
1. **P(y|x)**: Probability distribution over classes for input x
2. **P(y|x')**: Probability distribution for perturbed input x'
3. **KL divergence formula**: Measures difference between two probability distributions
4. **P(yᵢ|x) × log(P(yᵢ|x) / P(yᵢ|x'))**: For each class i, compute this term
5. **Σᵢ**: Sum over all classes
6. **Result**: Measures how different the probability distributions are

**Why KL Divergence?**:
- Asymmetric measure (not symmetric like MSE)
- Penalizes when P(y|x) is high but P(y|x') is low
- Natural for probability distributions

#### **Step 4: Combine Both Losses**

**Combined Objective**:
```
L_total = L_supervised + λ × L_unsupervised
```

**Step-by-step explanation**:
1. **L_supervised**: Ensures model fits labeled data correctly
2. **L_unsupervised**: Ensures model makes consistent predictions on unlabeled data
3. **λ**: Regularization parameter (weight for unsupervised term)
4. **λ > 0**: Controls trade-off between fitting labeled data and consistency

**Expanded Form**:
```
L_total = (1/|L|) × Σ_{(x,y) ∈ L} l(f(x), y) + λ × (1/|U|) × Σ_{x ∈ U} D(f(x), f(x'))
```

**Why This Works**:
- **First term**: Minimizes error on labeled examples (supervised signal)
- **Second term**: Encourages smooth predictions on unlabeled examples (unsupervised signal)
- **Together**: Model learns from both labeled and unlabeled data

#### **Step 5: Optimization**

**Goal**: Find model parameters θ that minimize L_total

```
θ* = argmin_θ L_total(θ)
```

**Gradient Descent Update**:
```
θ^(t+1) = θ^(t) - η × ∇_θ L_total(θ^(t))
```

**Step-by-step gradient calculation**:

**Gradient of supervised term**:
```
∇_θ L_supervised = (1/|L|) × Σ_{(x,y) ∈ L} ∇_θ l(f(x; θ), y)
```

**Explanation**:
1. **l(f(x; θ), y)**: Loss depends on parameters θ through f(x; θ)
2. **∇_θ l(·)**: Gradient with respect to parameters
3. **Chain rule**: ∇_θ l = (∂l/∂f) × (∂f/∂θ)
4. **Average**: Sum over all labeled examples and divide by |L|

**Gradient of unsupervised term**:
```
∇_θ L_unsupervised = (1/|U|) × Σ_{x ∈ U} ∇_θ D(f(x; θ), f(x'; θ))
```

**Step-by-step**:
1. **D(f(x; θ), f(x'; θ))**: Distance depends on θ through both f(x) and f(x')
2. **∇_θ D**: Gradient with respect to θ
3. **Chain rule**: ∇_θ D = (∂D/∂f(x)) × (∂f(x)/∂θ) + (∂D/∂f(x')) × (∂f(x')/∂θ)
4. **Average**: Sum over all unlabeled examples and divide by |U|

**Complete gradient**:
```
∇_θ L_total = ∇_θ L_supervised + λ × ∇_θ L_unsupervised
```

**Final update rule**:
```
θ^(t+1) = θ^(t) - η × [∇_θ L_supervised + λ × ∇_θ L_unsupervised]
```

### 1.4 Co-Training Mathematical Formulation

#### **Step 1: Problem Setup**

**Given**:
- Labeled data: L = {(x₁, y₁), ..., (xₗ, yₗ)}
- Unlabeled data: U = {xₗ₊₁, ..., xₗ₊ᵤ}
- Two views: X = [X₁, X₂] where X₁ and X₂ are conditionally independent given Y

**Assumption**: 
```
P(X₁, X₂ | Y) = P(X₁ | Y) × P(X₂ | Y)
```

**Explanation**:
- Given the class Y, views X₁ and X₂ are independent
- This means: Knowing X₁ doesn't help predict X₂ if we already know Y
- Example: Text content and hyperlinks of web pages (independent given the topic)

#### **Step 2: Train Initial Models**

**Model 1 (View 1)**:
```
f₁* = argmin_{f₁} (1/|L|) × Σ_{(x,y) ∈ L} l(f₁(x₁), y)
```

**Step-by-step**:
1. **f₁**: Model that takes View 1 (X₁) as input
2. **x₁**: View 1 features of example x
3. **l(f₁(x₁), y)**: Loss on labeled data using only View 1
4. **argmin**: Find model parameters that minimize this loss
5. **Result**: Model trained only on View 1 of labeled data

**Model 2 (View 2)**:
```
f₂* = argmin_{f₂} (1/|L|) × Σ_{(x,y) ∈ L} l(f₂(x₂), y)
```

**Similar explanation for View 2**

#### **Step 3: Predict on Unlabeled Data**

**Model 1 predictions**:
```
For each x ∈ U:
  ŷ₁ = f₁*(x₁)
  confidence₁ = max_i P(yᵢ | x₁)
```

**Explanation**:
1. **f₁*(x₁)**: Model 1's prediction using View 1
2. **P(yᵢ | x₁)**: Probability of class i given View 1
3. **max_i**: Maximum probability (confidence in prediction)
4. **High confidence**: Model is sure about prediction

**Model 2 predictions**:
```
For each x ∈ U:
  ŷ₂ = f₂*(x₂)
  confidence₂ = max_i P(yᵢ | x₂)
```

#### **Step 4: Select High-Confidence Predictions**

**For Model 1**:
```
L₁ = {(x, ŷ₁) | x ∈ U, confidence₁ > threshold, ŷ₁ = argmax_i P(yᵢ | x₁)}
```

**Step-by-step**:
1. **Filter unlabeled examples**: Keep only those with confidence > threshold
2. **Get prediction**: ŷ₁ = most likely class according to Model 1
3. **Create pseudo-labeled set**: L₁ contains (x, ŷ₁) pairs
4. **These become labels for Model 2**

**For Model 2**:
```
L₂ = {(x, ŷ₂) | x ∈ U, confidence₂ > threshold, ŷ₂ = argmax_i P(yᵢ | x₂)}
```

#### **Step 5: Retrain Models**

**Update Model 1**:
```
f₁^(t+1) = argmin_{f₁} (1/|L ∪ L₂|) × Σ_{(x,y) ∈ L ∪ L₂} l(f₁(x₁), y)
```

**Step-by-step**:
1. **L ∪ L₂**: Combine original labeled data L with pseudo-labels from Model 2
2. **Train on combined set**: Model 1 now sees more training examples
3. **y comes from**: Original labels in L, pseudo-labels in L₂
4. **Result**: Model 1 improves by learning from Model 2's high-confidence predictions

**Update Model 2**:
```
f₂^(t+1) = argmin_{f₂} (1/|L ∪ L₁|) × Σ_{(x,y) ∈ L ∪ L₁} l(f₂(x₂), y)
```

**Similar process for Model 2**

#### **Step 6: Iterate**

**Repeat Steps 3-5 until convergence**:
- Models keep teaching each other
- Training set grows with each iteration
- Stop when no new high-confidence predictions or max iterations reached

**Convergence criterion**:
```
|L₁^(t+1) ∪ L₂^(t+1)| - |L₁^(t) ∪ L₂^(t)| < ε
```

**Explanation**:
- **|L₁^(t) ∪ L₂^(t)|**: Total pseudo-labeled examples at iteration t
- **|L₁^(t+1) ∪ L₂^(t+1)|**: Total at iteration t+1
- **Difference**: How many new examples were added
- **ε**: Small threshold (e.g., 0.01)
- **Convergence**: When very few new examples are added

### 1.5 Information Gain in Semi-Supervised Context

#### **Step 1: Entropy of Labeled Data**

**Initial entropy**:
```
H(Y | L) = -Σᵢ P(yᵢ | L) × log₂(P(yᵢ | L))
```

**Step-by-step derivation**:
1. **P(yᵢ | L)**: Proportion of class i in labeled set L
   - P(yᵢ | L) = (number of examples with class i) / |L|
2. **log₂(P(yᵢ | L))**: Logarithm (base 2) of probability
   - Measures information content
3. **P(yᵢ | L) × log₂(P(yᵢ | L))**: Weighted information
4. **-Σᵢ**: Negative sum over all classes
   - Negative because log of probability < 1 is negative
5. **Result**: Entropy measures uncertainty in labeled data

**Example calculation**:
- Labeled set: 6 examples of class A, 4 examples of class B
- P(A | L) = 6/10 = 0.6
- P(B | L) = 4/10 = 0.4
- H(Y | L) = -(0.6 × log₂(0.6) + 0.4 × log₂(0.4))
- H(Y | L) = -(0.6 × (-0.737) + 0.4 × (-1.322))
- H(Y | L) = -(-0.442 - 0.529) = 0.971 bits

#### **Step 2: Conditional Entropy After Adding Unlabeled Data**

**After pseudo-labeling unlabeled data U**:
```
H(Y | L ∪ U_pseudo) = -Σᵢ P(yᵢ | L ∪ U_pseudo) × log₂(P(yᵢ | L ∪ U_pseudo))
```

**Step-by-step**:
1. **U_pseudo**: Subset of U that received pseudo-labels
2. **L ∪ U_pseudo**: Combined labeled and pseudo-labeled set
3. **P(yᵢ | L ∪ U_pseudo)**: Updated class proportions
4. **Calculate entropy**: Same formula as Step 1, but with updated probabilities

**Information Gain**:
```
IG = H(Y | L) - H(Y | L ∪ U_pseudo)
```

**Explanation**:
- **H(Y | L)**: Uncertainty before adding pseudo-labels
- **H(Y | L ∪ U_pseudo)**: Uncertainty after adding pseudo-labels
- **IG**: Reduction in uncertainty (information gained)
- **IG > 0**: Pseudo-labels reduced uncertainty (good!)
- **IG = 0**: No change (pseudo-labels didn't help)
- **IG < 0**: Uncertainty increased (bad pseudo-labels)

### 1.6 When to Use Semi-Supervised Learning

**✅ Use When:**
- Labeled data is expensive or scarce
- Large amount of unlabeled data available
- Labeled and unlabeled data come from same distribution
- Data has clear structure (clusters, manifolds)
- Base classifier works reasonably well on labeled data

**❌ Don't Use When:**
- Labeled data is sufficient for good performance
- Unlabeled data is from different distribution (domain shift)
- Base classifier performs poorly even on labeled data
- Data is too noisy or unstructured

---

## 2. Self-Supervised Learning (2.4)

### 2.1 Intuition & Goal

**Goal**: Learn useful representations from data itself without external labels by creating supervision signals from the data structure.

**Key Insight**: Data contains inherent structure that can be used as supervision!

### 2.2 Masked Language Modeling: Mathematical Derivation

#### **Step 1: Problem Setup**

**Given**: Sentence x = [x₁, x₂, ..., xₙ] where each xᵢ is a word/token

**Masking**: Randomly mask some positions M = {i₁, i₂, ..., iₘ} where m < n

**Masked sentence**: x_masked where xᵢ is replaced with [MASK] for i ∈ M

**Task**: Predict original words at masked positions

#### **Step 2: Conditional Probability**

**Goal**: Learn P(xᵢ | x\ᵢ) for i ∈ M

**Where**:
- **xᵢ**: Word at position i (what we want to predict)
- **x\ᵢ**: All words except position i (context)
- **P(xᵢ | x\ᵢ)**: Probability of word xᵢ given context

**Step-by-step explanation**:
1. **x\ᵢ = [x₁, ..., xᵢ₋₁, [MASK], xᵢ₊₁, ..., xₙ]**: Context with masked position
2. **Model f**: Takes context x\ᵢ as input
3. **Output**: Probability distribution over vocabulary V
4. **P(xᵢ | x\ᵢ)**: Probability that word xᵢ appears at position i

#### **Step 3: Model Architecture**

**Transformer encoder**:
```
h = Encoder(x_masked)
```

**Step-by-step**:
1. **x_masked**: Input sequence with [MASK] tokens
2. **Embedding**: Convert tokens to vectors
   - E(xᵢ) for each position i
   - E([MASK]) for masked positions
3. **Encoder**: Multi-layer transformer
   - Self-attention: Each position attends to all positions
   - Feed-forward: Non-linear transformation
   - Layer normalization: Stabilize training
4. **h**: Hidden representations for each position

**Prediction head**:
```
P(xᵢ | x\ᵢ) = Softmax(W × hᵢ + b)
```

**Step-by-step derivation**:
1. **hᵢ**: Hidden representation at position i (from encoder)
2. **W**: Weight matrix (|V| × d) where |V| is vocabulary size, d is hidden dimension
3. **W × hᵢ**: Linear transformation (|V| × 1 vector)
4. **+ b**: Add bias term (|V| × 1 vector)
5. **Softmax**: Convert to probability distribution
   ```
   Softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
   ```
6. **Result**: Probability distribution over vocabulary

#### **Step 4: Loss Function**

**Cross-entropy loss for each masked position**:
```
L_i = -log(P(xᵢ | x\ᵢ))
```

**Step-by-step**:
1. **P(xᵢ | x\ᵢ)**: Model's predicted probability of true word xᵢ
2. **log(P(xᵢ | x\ᵢ))**: Logarithm of probability
   - If P is high (close to 1), log is close to 0 (low loss)
   - If P is low (close to 0), log is very negative (high loss)
3. **Negative**: We want to maximize probability, so we minimize negative log
4. **Result**: Loss is low when model predicts correct word with high confidence

**Total loss over all masked positions**:
```
L = (1/|M|) × Σ_{i ∈ M} L_i
   = -(1/|M|) × Σ_{i ∈ M} log(P(xᵢ | x\ᵢ))
```

**Step-by-step**:
1. **L_i**: Loss for each masked position i
2. **Σ_{i ∈ M}**: Sum over all masked positions
3. **1/|M|**: Average over masked positions
4. **Result**: Average negative log-likelihood

**Why this works**:
- **Maximizing P(xᵢ | x\ᵢ)**: Model learns to predict correct words
- **Requires understanding context**: To predict well, model must understand:
  - Grammar and syntax (word order matters)
  - Semantics (meaning of words)
  - Discourse (how sentences connect)
- **Learns useful representations**: Hidden states h capture these understandings

#### **Step 5: Training Objective**

**Maximum Likelihood Estimation (MLE)**:
```
θ* = argmax_θ Π_{i ∈ M} P(xᵢ | x\ᵢ; θ)
```

**Step-by-step**:
1. **Π_{i ∈ M}**: Product over all masked positions
2. **P(xᵢ | x\ᵢ; θ)**: Probability depends on model parameters θ
3. **argmax_θ**: Find parameters that maximize product of probabilities
4. **Equivalent to minimizing negative log-likelihood**:
   ```
   log(Π P) = Σ log(P)
   -log(Π P) = -Σ log(P) = L (our loss function)
   ```

**Gradient descent update**:
```
θ^(t+1) = θ^(t) - η × ∇_θ L(θ^(t))
```

**Gradient calculation**:
```
∇_θ L = -(1/|M|) × Σ_{i ∈ M} ∇_θ log(P(xᵢ | x\ᵢ; θ))
```

**Step-by-step**:
1. **log(P(xᵢ | x\ᵢ; θ))**: Log probability depends on θ
2. **∇_θ log(P)**: Gradient with respect to parameters
3. **Chain rule**: 
   ```
   ∇_θ log(P) = (1/P) × ∇_θ P
              = (1/P) × (∂P/∂h) × (∂h/∂θ)
   ```
4. **Backpropagation**: Compute gradients through encoder and prediction head
5. **Average**: Sum over masked positions and divide by |M|

### 2.3 Contrastive Learning: Mathematical Derivation

#### **Step 1: Problem Setup**

**Given**:
- Anchor example: x
- Positive example: x⁺ (similar to x, e.g., augmented version)
- Negative examples: {x₁⁻, x₂⁻, ..., xₖ⁻} (different from x)

**Goal**: Learn representation where:
- x and x⁺ are close in representation space
- x and xᵢ⁻ are far in representation space

#### **Step 2: Representation Function**

**Encoder**: f(x; θ) → z (d-dimensional vector)

**Step-by-step**:
1. **x**: Input (image, text, etc.)
2. **f(·; θ)**: Encoder network with parameters θ
3. **z**: Learned representation (d-dimensional vector)
4. **Normalization**: Often ||z|| = 1 (unit vector)

#### **Step 3: Similarity Function**

**Cosine similarity**:
```
sim(x, x') = (f(x) · f(x')) / (||f(x)|| × ||f(x')||)
           = f(x) · f(x')  (if normalized)
```

**Step-by-step derivation**:
1. **f(x) · f(x')**: Dot product of representation vectors
   ```
   f(x) · f(x') = Σᵢ f(x)ᵢ × f(x')ᵢ
   ```
2. **||f(x)||**: L2 norm of f(x)
   ```
   ||f(x)|| = √(Σᵢ f(x)ᵢ²)
   ```
3. **Division**: Normalize by magnitudes
4. **If normalized**: ||f(x)|| = ||f(x')|| = 1, so:
   ```
   sim(x, x') = f(x) · f(x')
   ```
5. **Range**: [-1, 1] where 1 = identical, -1 = opposite

**Temperature-scaled similarity**:
```
sim_τ(x, x') = (f(x) · f(x')) / τ
```

**Step-by-step**:
1. **τ**: Temperature parameter (typically 0.1 to 0.5)
2. **Division by τ**: Scales similarity
   - Smaller τ: Sharper distribution (more confident)
   - Larger τ: Softer distribution (less confident)
3. **Effect**: Controls how much to separate positive and negative pairs

#### **Step 4: Contrastive Loss (InfoNCE)**

**InfoNCE Loss**:
```
L = -log(exp(sim_τ(x, x⁺)) / (exp(sim_τ(x, x⁺)) + Σᵢ exp(sim_τ(x, xᵢ⁻))))
```

**Step-by-step derivation**:

**Step 4a: Numerator**
```
exp(sim_τ(x, x⁺)) = exp((f(x) · f(x⁺)) / τ)
```

**Explanation**:
- **sim_τ(x, x⁺)**: Similarity between anchor and positive
- **exp(·)**: Exponential function
- **Result**: Large value when x and x⁺ are similar

**Step 4b: Denominator**
```
exp(sim_τ(x, x⁺)) + Σᵢ exp(sim_τ(x, xᵢ⁻))
```

**Step-by-step**:
1. **exp(sim_τ(x, x⁺))**: Positive pair similarity
2. **exp(sim_τ(x, xᵢ⁻))**: Negative pair similarities
3. **Σᵢ**: Sum over all negative examples
4. **Total**: Sum of all similarities (positive + negatives)

**Step 4c: Ratio**
```
exp(sim_τ(x, x⁺)) / (exp(sim_τ(x, x⁺)) + Σᵢ exp(sim_τ(x, xᵢ⁻)))
```

**Explanation**:
- **Numerator**: Similarity to positive example
- **Denominator**: Total similarity (positive + all negatives)
- **Interpretation**: Probability that x⁺ is the positive example
- **Range**: [0, 1]
- **High value**: Model correctly identifies positive
- **Low value**: Model confuses positive with negatives

**Step 4d: Negative Log**
```
L = -log(ratio)
```

**Step-by-step**:
1. **log(ratio)**: Natural logarithm of the ratio
   - If ratio = 1 (perfect): log(1) = 0, so L = 0
   - If ratio = 0 (worst): log(0) = -∞, so L = ∞
2. **Negative**: We want to maximize ratio, so minimize negative log
3. **Result**: Loss is low when model correctly separates positive from negatives

**Final form**:
```
L = -log(exp(sim_τ(x, x⁺)) / (exp(sim_τ(x, x⁺)) + Σᵢ exp(sim_τ(x, xᵢ⁻))))
```

#### **Step 5: Simplification**

**Using log properties**:
```
L = -[log(exp(sim_τ(x, x⁺))) - log(exp(sim_τ(x, x⁺)) + Σᵢ exp(sim_τ(x, xᵢ⁻)))]
```

**Step-by-step**:
1. **log(a/b) = log(a) - log(b)**: Property of logarithms
2. **Apply**: log(numerator) - log(denominator)
3. **log(exp(sim_τ(x, x⁺))) = sim_τ(x, x⁺)**: log and exp cancel
4. **Result**: Simplified form

**Final simplified form**:
```
L = -sim_τ(x, x⁺) + log(exp(sim_τ(x, x⁺)) + Σᵢ exp(sim_τ(x, xᵢ⁻)))
```

**Explanation**:
- **-sim_τ(x, x⁺)**: Pull positive pair closer (negative because we minimize)
- **log(exp(sim_τ(x, x⁺)) + Σᵢ exp(sim_τ(x, xᵢ⁻)))**: Log-sum-exp term
  - Pushes negative pairs away
  - Log-sum-exp is smooth approximation of max function

#### **Step 6: Gradient Calculation**

**Gradient with respect to anchor representation f(x)**:
```
∇_{f(x)} L = -∇_{f(x)} sim_τ(x, x⁺) + ∇_{f(x)} log(exp(sim_τ(x, x⁺)) + Σᵢ exp(sim_τ(x, xᵢ⁻)))
```

**Step 6a: First term gradient**
```
∇_{f(x)} sim_τ(x, x⁺) = (1/τ) × f(x⁺)
```

**Step-by-step**:
1. **sim_τ(x, x⁺) = (f(x) · f(x⁺)) / τ**
2. **∇_{f(x)} (f(x) · f(x⁺))**: Gradient of dot product
3. **Result**: f(x⁺) (the other vector in dot product)
4. **Divide by τ**: Scale by temperature

**Step 6b: Second term gradient**
```
∇_{f(x)} log(exp(sim_τ(x, x⁺)) + Σᵢ exp(sim_τ(x, xᵢ⁻))) 
  = (1/Z) × [exp(sim_τ(x, x⁺)) × (1/τ) × f(x⁺) + Σᵢ exp(sim_τ(x, xᵢ⁻)) × (1/τ) × f(xᵢ⁻)]
```

**Step-by-step**:
1. **Z = exp(sim_τ(x, x⁺)) + Σᵢ exp(sim_τ(x, xᵢ⁻))**: Normalization constant
2. **Chain rule**: 
   - ∇ log(g) = (1/g) × ∇g
   - ∇ exp(sim) = exp(sim) × ∇ sim
3. **∇ sim_τ(x, x⁺) = (1/τ) × f(x⁺)**: From Step 6a
4. **∇ sim_τ(x, xᵢ⁻) = (1/τ) × f(xᵢ⁻)**: Similar for negatives
5. **Combine**: Weighted sum of positive and negative representations

**Complete gradient**:
```
∇_{f(x)} L = -(1/τ) × f(x⁺) + (1/Z) × (1/τ) × [exp(sim_τ(x, x⁺)) × f(x⁺) + Σᵢ exp(sim_τ(x, xᵢ⁻)) × f(xᵢ⁻)]
```

**Interpretation**:
- **First term**: Pull toward positive example
- **Second term**: Push away from weighted combination of positive and negatives
- **Net effect**: Representations move closer to positives, farther from negatives

### 2.4 Autoencoding: Mathematical Derivation

#### **Step 1: Problem Setup**

**Given**: Input x (e.g., image, text)

**Goal**: Learn to reconstruct x from compressed representation

**Architecture**:
```
x → Encoder → z → Decoder → x̂
```

#### **Step 2: Encoder**

**Encoder function**:
```
z = Encoder(x; θ_e) = f_e(x; θ_e)
```

**Step-by-step**:
1. **x**: Input (high-dimensional, e.g., 784-dim for 28×28 image)
2. **f_e(·; θ_e)**: Encoder network with parameters θ_e
3. **z**: Latent representation (low-dimensional, e.g., 32-dim)
4. **Compression**: z has much fewer dimensions than x

#### **Step 3: Decoder**

**Decoder function**:
```
x̂ = Decoder(z; θ_d) = f_d(z; θ_d)
```

**Step-by-step**:
1. **z**: Latent representation
2. **f_d(·; θ_d)**: Decoder network with parameters θ_d
3. **x̂**: Reconstructed input (same dimension as x)
4. **Expansion**: z is expanded back to original dimension

#### **Step 4: Reconstruction Loss**

**Mean Squared Error (MSE)**:
```
L_reconstruction = (1/n) × ||x - x̂||²
                  = (1/n) × Σᵢ (xᵢ - x̂ᵢ)²
```

**Step-by-step derivation**:
1. **x**: Original input vector
2. **x̂**: Reconstructed input vector
3. **x - x̂**: Element-wise difference
4. **||x - x̂||²**: Squared L2 norm
   ```
   ||x - x̂||² = Σᵢ (xᵢ - x̂ᵢ)²
   ```
5. **1/n**: Average over n dimensions (or examples)
6. **Result**: Average squared reconstruction error

**For images (per pixel)**:
```
L_reconstruction = (1/(H×W×C)) × Σ_{h,w,c} (x_{h,w,c} - x̂_{h,w,c})²
```

**Where**:
- **H, W, C**: Height, width, channels
- **x_{h,w,c}**: Pixel value at position (h, w, c)
- **x̂_{h,w,c}**: Reconstructed pixel value

#### **Step 5: Complete Objective**

**Combined loss**:
```
L = L_reconstruction + λ × L_regularization
```

**Reconstruction term** (from Step 4):
```
L_reconstruction = (1/n) × ||x - Decoder(Encoder(x))||²
```

**Regularization term** (optional):
```
L_regularization = ||z||²  (L2 regularization on latent code)
```

**Step-by-step**:
1. **||z||²**: Squared L2 norm of latent representation
2. **Purpose**: Encourage compact representations
3. **λ**: Regularization weight
4. **Effect**: Prevents latent code from growing too large

**Final objective**:
```
L(θ_e, θ_d) = (1/n) × ||x - f_d(f_e(x; θ_e); θ_d)||² + λ × ||f_e(x; θ_e)||²
```

#### **Step 6: Optimization**

**Goal**: Find parameters that minimize reconstruction error

```
(θ_e*, θ_d*) = argmin_{θ_e, θ_d} L(θ_e, θ_d)
```

**Gradient descent**:
```
θ_e^(t+1) = θ_e^(t) - η × ∇_{θ_e} L
θ_d^(t+1) = θ_d^(t) - η × ∇_{θ_d} L
```

**Gradient calculation (encoder)**:
```
∇_{θ_e} L = ∇_{θ_e} L_reconstruction + λ × ∇_{θ_e} L_regularization
```

**Step-by-step**:
1. **L_reconstruction = (1/n) × ||x - f_d(f_e(x; θ_e); θ_d)||²**
2. **Chain rule**:
   ```
   ∇_{θ_e} L_reconstruction = (2/n) × (x - x̂) × (∂x̂/∂z) × (∂z/∂θ_e)
   ```
3. **∂x̂/∂z**: Gradient through decoder
4. **∂z/∂θ_e**: Gradient through encoder
5. **Backpropagation**: Compute gradients through both networks

**Gradient calculation (decoder)**:
```
∇_{θ_d} L = ∇_{θ_d} L_reconstruction
          = (2/n) × (x - x̂) × (∂x̂/∂θ_d)
```

**Step-by-step**:
1. **L_reconstruction depends on θ_d through x̂ = f_d(z; θ_d)**
2. **Chain rule**: Gradient flows through decoder
3. **No regularization term**: Decoder parameters not regularized (or separately)

### 2.5 When to Use Self-Supervised Learning

**✅ Use When:**
- Large amount of unlabeled data available
- Need good representations for downstream tasks
- Labeled data is scarce or expensive
- Working with sequential or structured data
- Building foundation models

**❌ Don't Use When:**
- Small datasets (may not learn useful representations)
- Simple tasks that don't need rich representations
- Real-time constraints (pre-training can be slow)
- Domain-specific tasks with no related unlabeled data

---

## 3. Other Learning Paradigms (2.5)

### 3.1 Multi-Model Learning

### 3.1.1 Mathematical Formulation

#### **Step 1: Problem Setup**

**Given**:
- Labeled data: L = {(x₁, y₁), ..., (xₗ, yₗ)}
- Unlabeled data: U = {xₗ₊₁, ..., xₗ₊ᵤ}
- M models: f₁, f₂, ..., fₘ
- Each model trained on different subset/view

#### **Step 2: Individual Model Training**

**Model i on its subset**:
```
f_i* = argmin_{f_i} (1/|L_i|) × Σ_{(x,y) ∈ L_i} l(f_i(x), y)
```

**Step-by-step**:
1. **L_i**: Subset of labeled data for model i
   - Could be different features (view)
   - Could be different examples (bootstrap sample)
   - Could be different algorithm
2. **l(f_i(x), y)**: Loss function
3. **argmin**: Find best parameters for model i
4. **Result**: Each model learns from its own perspective

#### **Step 3: Prediction Aggregation**

**For unlabeled example x**:
```
ŷ = Aggregate({f₁*(x), f₂*(x), ..., fₘ*(x)})
```

**Majority voting (classification)**:
```
ŷ = argmax_y Σ_{i=1}^M I(f_i*(x) = y)
```

**Step-by-step derivation**:
1. **f_i*(x)**: Prediction of model i
2. **I(f_i*(x) = y)**: Indicator function
   - I(condition) = 1 if condition true, 0 otherwise
3. **Σ_{i=1}^M I(f_i*(x) = y)**: Count how many models predict class y
4. **argmax_y**: Choose class with most votes
5. **Result**: Class predicted by majority of models

**Averaging (regression)**:
```
ŷ = (1/M) × Σ_{i=1}^M f_i*(x)
```

**Step-by-step**:
1. **f_i*(x)**: Prediction of model i (continuous value)
2. **Σ_{i=1}^M**: Sum over all models
3. **1/M**: Average
4. **Result**: Mean prediction across models

**Weighted averaging**:
```
ŷ = Σ_{i=1}^M w_i × f_i*(x)
```

**Where**:
- **w_i**: Weight for model i
- **Σ w_i = 1**: Weights sum to 1
- **w_i**: Could be based on model performance, confidence, etc.

#### **Step 4: High-Confidence Labeling**

**For each model i**:
```
U_i = {x ∈ U | confidence_i(x) > threshold}
```

**Step-by-step**:
1. **confidence_i(x)**: Model i's confidence in prediction
   - For classification: max probability
   - For regression: inverse of prediction variance
2. **threshold**: Confidence threshold (e.g., 0.9)
3. **U_i**: Unlabeled examples where model i is confident
4. **These become pseudo-labels for other models**

**Pseudo-label creation**:
```
L_i^pseudo = {(x, f_i*(x)) | x ∈ U_i}
```

**Step-by-step**:
1. **f_i*(x)**: Model i's prediction (treated as label)
2. **L_i^pseudo**: Pseudo-labeled set from model i
3. **Used to train other models**

#### **Step 5: Iterative Refinement**

**Update model i**:
```
f_i^(t+1) = argmin_{f_i} (1/|L_i ∪ L_{-i}^pseudo|) × Σ_{(x,y) ∈ L_i ∪ L_{-i}^pseudo} l(f_i(x), y)
```

**Step-by-step**:
1. **L_i**: Original labeled data for model i
2. **L_{-i}^pseudo**: Pseudo-labels from other models (not model i)
3. **L_i ∪ L_{-i}^pseudo**: Combined training set
4. **Train**: Model i learns from its own data + other models' confident predictions
5. **Result**: Models teach each other iteratively

### 3.2 Active Learning

### 3.2.1 Uncertainty Sampling: Mathematical Derivation

#### **Step 1: Entropy-Based Uncertainty**

**Entropy of prediction distribution**:
```
H(P(y|x)) = -Σᵢ P(yᵢ|x) × log(P(yᵢ|x))
```

**Step-by-step derivation**:
1. **P(y|x)**: Probability distribution over classes for input x
   - P(yᵢ|x): Probability of class i given x
   - Σᵢ P(yᵢ|x) = 1 (valid distribution)
2. **log(P(yᵢ|x))**: Natural logarithm of probability
3. **P(yᵢ|x) × log(P(yᵢ|x))**: Weighted log probability
4. **-Σᵢ**: Negative sum over all classes
5. **Result**: Entropy measures uncertainty
   - **H = 0**: Certain (one class has probability 1)
   - **H = log(K)**: Maximum uncertainty (uniform distribution over K classes)

**Example calculation**:
- 3 classes, prediction: [0.1, 0.8, 0.1]
- H = -(0.1×log(0.1) + 0.8×log(0.8) + 0.1×log(0.1))
- H = -(-0.230 - 0.179 - 0.230) = 0.639 (moderate uncertainty)

- Uniform: [0.33, 0.33, 0.33]
- H = -(0.33×log(0.33) + 0.33×log(0.33) + 0.33×log(0.33))
- H = -(-0.361 - 0.361 - 0.361) = 1.083 (high uncertainty, close to log(3) = 1.099)

#### **Step 2: Query Selection**

**Select most uncertain example**:
```
x* = argmax_{x ∈ U} H(P(y|x))
```

**Step-by-step**:
1. **U**: Set of unlabeled examples
2. **H(P(y|x))**: Entropy for each x ∈ U
3. **argmax**: Find x with maximum entropy
4. **Result**: Example where model is most uncertain

#### **Step 3: Margin-Based Uncertainty**

**Margin**:
```
margin(x) = P(y₁|x) - P(y₂|x)
```

**Step-by-step**:
1. **y₁**: Most likely class: y₁ = argmax_i P(yᵢ|x)
2. **y₂**: Second most likely class: y₂ = argmax_{i≠y₁} P(yᵢ|x)
3. **P(y₁|x) - P(y₂|x)**: Difference between top two probabilities
4. **Small margin**: Model unsure between top two classes (high uncertainty)
5. **Large margin**: Model confident in top class (low uncertainty)

**Query selection**:
```
x* = argmin_{x ∈ U} margin(x)
```

**Step-by-step**:
1. **margin(x)**: Margin for each unlabeled example
2. **argmin**: Find x with minimum margin
3. **Result**: Example where model is least confident between top two classes

#### **Step 4: Least Confident**

**Confidence**:
```
confidence(x) = max_i P(yᵢ|x)
```

**Step-by-step**:
1. **P(yᵢ|x)**: Probability of each class
2. **max_i**: Maximum probability
3. **confidence(x)**: Model's confidence in most likely class
4. **High confidence**: Model is sure
5. **Low confidence**: Model is uncertain

**Query selection**:
```
x* = argmin_{x ∈ U} confidence(x)
```

**Step-by-step**:
1. **confidence(x)**: Confidence for each unlabeled example
2. **argmin**: Find x with minimum confidence
3. **Result**: Example where model is least confident

### 3.2.2 Query-by-Committee: Mathematical Derivation

#### **Step 1: Committee Setup**

**Train K models**: f₁, f₂, ..., fₖ

**Different models**:
- Different algorithms
- Different hyperparameters
- Different training subsets
- Different initializations

#### **Step 2: Vote Entropy**

**Vote count for class y**:
```
V(y) = Σ_{i=1}^K I(f_i(x) = y)
```

**Step-by-step**:
1. **f_i(x)**: Prediction of model i
2. **I(f_i(x) = y)**: Indicator (1 if model i predicts y, 0 otherwise)
3. **Σ_{i=1}^K**: Sum over all K models
4. **V(y)**: Number of models voting for class y

**Vote entropy**:
```
H_vote(x) = -Σ_y (V(y)/K) × log(V(y)/K)
```

**Step-by-step derivation**:
1. **V(y)/K**: Proportion of models voting for class y
   - Range: [0, 1]
   - Σ_y (V(y)/K) = 1 (all models vote)
2. **log(V(y)/K)**: Logarithm of vote proportion
3. **(V(y)/K) × log(V(y)/K)**: Weighted log
4. **-Σ_y**: Negative sum over all classes
5. **Result**: Entropy of vote distribution
   - **High H_vote**: Models disagree (high uncertainty)
   - **Low H_vote**: Models agree (low uncertainty)

**Example**:
- 5 models, 3 classes
- Votes: [3, 2, 0] (3 vote class A, 2 vote class B, 0 vote class C)
- H_vote = -((3/5)×log(3/5) + (2/5)×log(2/5) + (0/5)×log(0/5))
- H_vote = -((0.6)×(-0.511) + (0.4)×(-0.916) + 0)
- H_vote = -(0.307 + 0.366) = 0.673 (moderate disagreement)

- Unanimous: [5, 0, 0]
- H_vote = -((5/5)×log(5/5) + 0 + 0) = -log(1) = 0 (no disagreement)

#### **Step 3: KL Divergence**

**Average prediction**:
```
P_avg(y|x) = (1/K) × Σ_{i=1}^K P_i(y|x)
```

**Step-by-step**:
1. **P_i(y|x)**: Probability distribution from model i
2. **Σ_{i=1}^K**: Sum over all models
3. **1/K**: Average
4. **P_avg(y|x)**: Average probability distribution

**KL divergence for each model**:
```
KL(P_avg || P_i) = Σ_y P_avg(y|x) × log(P_avg(y|x) / P_i(y|x))
```

**Step-by-step derivation**:
1. **P_avg(y|x) / P_i(y|x)**: Ratio of average to individual
2. **log(ratio)**: Logarithm of ratio
3. **P_avg(y|x) × log(ratio)**: Weighted log ratio
4. **Σ_y**: Sum over all classes
5. **Result**: Measures how much model i diverges from average
   - **High KL**: Model i disagrees with average (high uncertainty)
   - **Low KL**: Model i agrees with average (low uncertainty)

**Total disagreement**:
```
Disagreement(x) = (1/K) × Σ_{i=1}^K KL(P_avg || P_i)
```

**Step-by-step**:
1. **KL(P_avg || P_i)**: Divergence of each model from average
2. **Σ_{i=1}^K**: Sum over all models
3. **1/K**: Average divergence
4. **Result**: Average disagreement across committee

**Query selection**:
```
x* = argmax_{x ∈ U} Disagreement(x)
```

### 3.3 Transfer Learning

### 3.3.1 Mathematical Formulation

#### **Step 1: Source Task**

**Source dataset**: D_source = {(x_s, y_s)}

**Source model**:
```
f_source* = argmin_{f_source} (1/|D_source|) × Σ_{(x_s, y_s) ∈ D_source} l(f_source(x_s), y_s)
```

**Step-by-step**:
1. **f_source**: Model for source task
2. **l(·, ·)**: Loss function
3. **argmin**: Find best parameters
4. **Result**: Pre-trained model on source task

#### **Step 2: Transfer Objective**

**Target dataset**: D_target = {(x_t, y_t)} (small)

**Transfer learning objective**:
```
L_target = L_supervised + λ × L_transfer
```

**Supervised term**:
```
L_supervised = (1/|D_target|) × Σ_{(x_t, y_t) ∈ D_target} l(f_target(x_t), y_t)
```

**Step-by-step**:
1. **f_target**: Model for target task
2. **l(f_target(x_t), y_t)**: Loss on target data
3. **Average**: Over target examples
4. **Result**: Ensures model fits target data

**Transfer term**:
```
L_transfer = D(f_source(x_t), f_target(x_t))
```

**Step-by-step**:
1. **f_source(x_t)**: Source model's representation of target input
2. **f_target(x_t)**: Target model's representation
3. **D(·, ·)**: Distance metric (e.g., MSE, KL divergence)
4. **Result**: Encourages target model to learn similar representations to source

**Complete objective**:
```
L_target = (1/|D_target|) × Σ_{(x_t, y_t) ∈ D_target} l(f_target(x_t), y_t) 
          + λ × (1/|D_target|) × Σ_{x_t ∈ D_target} D(f_source(x_t), f_target(x_t))
```

#### **Step 3: Fine-Tuning**

**Initialize target model with source parameters**:
```
θ_target^(0) = θ_source*
```

**Step-by-step**:
1. **θ_source***: Learned parameters from source task
2. **θ_target^(0)**: Initial parameters for target task
3. **Start from source**: Target model begins with source knowledge

**Fine-tune on target task**:
```
θ_target* = argmin_{θ_target} L_target(θ_target)
```

**Gradient descent**:
```
θ_target^(t+1) = θ_target^(t) - η × ∇_{θ_target} L_target(θ_target^(t))
```

**Gradient calculation**:
```
∇_{θ_target} L_target = ∇_{θ_target} L_supervised + λ × ∇_{θ_target} L_transfer
```

**Step-by-step**:
1. **∇_{θ_target} L_supervised**: Gradient from target task loss
2. **∇_{θ_target} L_transfer**: Gradient from transfer term
3. **λ**: Balances between fitting target and staying close to source
4. **Result**: Model adapts to target while retaining source knowledge

#### **Step 4: Feature Extraction Alternative**

**Freeze source model**:
```
f_source(x) = fixed (parameters frozen)
```

**Extract features**:
```
z = f_source(x)
```

**Train new classifier**:
```
g* = argmin_g (1/|D_target|) × Σ_{(x_t, y_t) ∈ D_target} l(g(f_source(x_t)), y_t)
```

**Step-by-step**:
1. **f_source(x_t)**: Extract features using frozen source model
2. **g(·)**: New classifier (only this is trained)
3. **l(g(z), y_t)**: Loss on target labels
4. **Result**: Only classifier is learned, features come from source

### 3.4 Federated Learning

### 3.4.1 Federated Averaging: Mathematical Derivation

#### **Step 1: Problem Setup**

**Given**:
- N devices with local datasets: D₁, D₂, ..., Dₙ
- Each device i has: D_i = {(xᵢⱼ, yᵢⱼ)}_{j=1}^{|D_i|}
- Global model parameters: w

**Goal**: Learn global model without sharing raw data

#### **Step 2: Local Loss Function**

**Loss on device i**:
```
L_i(w) = (1/|D_i|) × Σ_{(x,y) ∈ D_i} l(f(x; w), y)
```

**Step-by-step**:
1. **f(x; w)**: Model prediction with parameters w
2. **l(f(x; w), y)**: Loss for example (x, y)
3. **Σ_{(x,y) ∈ D_i}**: Sum over all examples on device i
4. **1/|D_i|**: Average over device i's examples
5. **Result**: Average loss on device i's local data

#### **Step 3: Global Objective**

**Federated learning objective**:
```
L(w) = Σ_{i=1}^N (|D_i|/|D|) × L_i(w)
```

**Step-by-step derivation**:
1. **|D_i|/|D|**: Weight proportional to device i's data size
   - |D| = Σ_{i=1}^N |D_i|: Total data size
   - Larger devices contribute more
2. **(|D_i|/|D|) × L_i(w)**: Weighted loss from device i
3. **Σ_{i=1}^N**: Sum over all devices
4. **Result**: Weighted average of local losses

**Expanded form**:
```
L(w) = Σ_{i=1}^N (|D_i|/|D|) × (1/|D_i|) × Σ_{(x,y) ∈ D_i} l(f(x; w), y)
     = (1/|D|) × Σ_{i=1}^N Σ_{(x,y) ∈ D_i} l(f(x; w), y)
```

**Step-by-step simplification**:
1. **(|D_i|/|D|) × (1/|D_i|)**: Cancel |D_i|
2. **Result**: (1/|D|)
3. **Σ_{i=1}^N Σ_{(x,y) ∈ D_i}**: Double sum over all devices and all examples
4. **Result**: Average loss over all data (as if centralized)

#### **Step 4: Federated Averaging Algorithm**

**Round t**:

**Step 4a: Server sends global model**
```
w^t → devices
```

**Step 4b: Local training on each device**
```
w_i^(t,0) = w^t  (initialize with global model)

For epoch e = 1 to E:
  w_i^(t,e) = w_i^(t,e-1) - η × ∇_{w} L_i(w_i^(t,e-1))
```

**Step-by-step**:
1. **w_i^(t,0)**: Initialize device i's model with global model
2. **E**: Number of local epochs
3. **∇_{w} L_i(w)**: Gradient of local loss
4. **Update**: Device i performs E steps of gradient descent
5. **w_i^(t,E)**: Final local model after E epochs

**Step 4c: Devices send updates**
```
Δw_i^t = w_i^(t,E) - w^t  (update = final - initial)
```

**Step-by-step**:
1. **w_i^(t,E)**: Device i's model after local training
2. **w^t**: Initial global model
3. **Δw_i^t**: Change made by device i
4. **Send to server**: Only update is sent, not raw data

**Step 4d: Server aggregates updates**
```
w^(t+1) = w^t + (1/N) × Σ_{i=1}^N Δw_i^t
```

**Step-by-step derivation**:
1. **Δw_i^t**: Update from device i
2. **Σ_{i=1}^N**: Sum over all devices
3. **1/N**: Average updates
4. **w^t + average_update**: Add average update to current global model
5. **Result**: New global model

**Alternative: Weighted averaging**:
```
w^(t+1) = w^t + Σ_{i=1}^N (|D_i|/|D|) × Δw_i^t
```

**Step-by-step**:
1. **|D_i|/|D|**: Weight proportional to data size
2. **(|D_i|/|D|) × Δw_i^t**: Weighted update
3. **Σ_{i=1}^N**: Sum weighted updates
4. **Result**: Devices with more data contribute more

#### **Step 5: Convergence**

**Convergence criterion**:
```
||w^(t+1) - w^t|| < ε
```

**Step-by-step**:
1. **w^(t+1) - w^t**: Change in global model
2. **||·||**: L2 norm (magnitude of change)
3. **ε**: Small threshold
4. **Convergence**: When model stops changing significantly

### 3.5 Meta-Learning

### 3.5.1 MAML: Mathematical Derivation

#### **Step 1: Problem Setup**

**Task distribution**: P(T)

**Sample task T ~ P(T)**:
- Training set: D_T^train = {(x₁, y₁), ..., (xₖ, yₖ)} (few examples)
- Test set: D_T^test

**Goal**: Learn initialization θ that allows fast adaptation

#### **Step 2: Inner Loop (Task-Specific Adaptation)**

**For task T, adapt from θ**:
```
θ_T' = θ - α × ∇_θ L_T(θ, D_T^train)
```

**Step-by-step derivation**:
1. **L_T(θ, D_T^train)**: Loss on task T's training data
   ```
   L_T(θ, D_T^train) = (1/|D_T^train|) × Σ_{(x,y) ∈ D_T^train} l(f(x; θ), y)
   ```
2. **∇_θ L_T(θ, D_T^train)**: Gradient with respect to θ
3. **α**: Inner learning rate (step size for adaptation)
4. **θ - α × ∇_θ L_T**: One gradient step from initialization
5. **θ_T'**: Adapted parameters for task T

**Multiple steps (optional)**:
```
θ_T^(0) = θ
For step s = 1 to S:
  θ_T^(s) = θ_T^(s-1) - α × ∇_θ L_T(θ_T^(s-1), D_T^train)
θ_T' = θ_T^(S)
```

#### **Step 3: Outer Loop (Meta-Learning)**

**Meta-objective**:
```
L_meta(θ) = E_{T ~ P(T)} [L_T(θ_T', D_T^test)]
```

**Step-by-step**:
1. **θ_T'**: Adapted parameters (from Step 2)
2. **L_T(θ_T', D_T^test)**: Loss on task T's test data using adapted parameters
3. **E_{T ~ P(T)}**: Expectation over task distribution
4. **Result**: Average test loss across tasks

**Monte Carlo approximation**:
```
L_meta(θ) ≈ (1/M) × Σ_{i=1}^M L_{T_i}(θ_{T_i}', D_{T_i}^test)
```

**Step-by-step**:
1. **Sample M tasks**: T₁, T₂, ..., T_M ~ P(T)
2. **For each task**: Adapt and evaluate
3. **Average**: Over M tasks
4. **Result**: Approximate meta-objective

#### **Step 4: Meta-Gradient**

**Gradient of meta-objective**:
```
∇_θ L_meta(θ) = ∇_θ E_{T ~ P(T)} [L_T(θ_T', D_T^test)]
```

**Step-by-step derivation**:
1. **θ_T' = θ - α × ∇_θ L_T(θ, D_T^train)**: Depends on θ
2. **Chain rule**:
   ```
   ∇_θ L_T(θ_T', D_T^test) = (∂L_T/∂θ_T') × (∂θ_T'/∂θ)
   ```
3. **∂θ_T'/∂θ**: How adapted parameters depend on initialization
   ```
   ∂θ_T'/∂θ = I - α × ∇_θ² L_T(θ, D_T^train)
   ```
   - **I**: Identity matrix
   - **∇_θ² L_T**: Hessian (second derivative)
4. **Result**: Meta-gradient requires second-order derivatives

**First-order approximation** (simpler):
```
∇_θ L_meta(θ) ≈ (1/M) × Σ_{i=1}^M ∇_θ L_{T_i}(θ_{T_i}', D_{T_i}^test)
```

**Step-by-step**:
1. **Approximate**: Ignore second-order terms
2. **∇_θ L_{T_i}(θ_{T_i}', D_{T_i}^test)**: Gradient treating θ_T' as constant
3. **Average**: Over tasks
4. **Result**: Simpler but less accurate

#### **Step 5: Meta-Update**

**Update initialization**:
```
θ^(t+1) = θ^(t) - β × ∇_θ L_meta(θ^(t))
```

**Step-by-step**:
1. **∇_θ L_meta(θ^(t))**: Meta-gradient
2. **β**: Meta learning rate (outer loop step size)
3. **θ^(t+1)**: Updated initialization
4. **Result**: Initialization that allows fast adaptation

**Complete algorithm**:
```
Initialize: θ^(0)
For meta-iteration t = 1 to T:
  Sample tasks: T₁, ..., T_M ~ P(T)
  For each task T_i:
    Adapt: θ_{T_i}' = θ^(t) - α × ∇_θ L_{T_i}(θ^(t), D_{T_i}^train)
    Evaluate: L_i = L_{T_i}(θ_{T_i}', D_{T_i}^test)
  Meta-update: θ^(t+1) = θ^(t) - β × (1/M) × Σ_{i=1}^M ∇_θ L_i
```

---

## 4. Comparison & Integration

### 4.1 Paradigm Comparison

| Paradigm | Data Requirement | Key Strength | Main Use Case |
|----------|-----------------|--------------|---------------|
| **Semi-Supervised** | Few labeled + many unlabeled | Leverages unlabeled data | When labeling is expensive |
| **Self-Supervised** | Large unlabeled | No labels needed | Representation learning |
| **Multi-Model** | Multiple views/subsets | Model diversity | Complementary information |
| **Active Learning** | Interactive oracle | Minimizes labeling cost | Strategic data selection |
| **Transfer Learning** | Source + target tasks | Knowledge transfer | Related domains |
| **Federated Learning** | Distributed data | Privacy preservation | Decentralized settings |
| **Meta-Learning** | Many related tasks | Fast adaptation | Few-shot learning |

### 4.2 Integration Strategies

**Semi-Supervised + Active Learning**:
- Use active learning to select which unlabeled examples to label
- Combine strategic selection with unlabeled data leverage

**Self-Supervised + Transfer Learning**:
- Pre-train with self-supervision (no labels)
- Fine-tune on target task (few labels)

---

## 5. Best Practices & Applications

### 5.1 Choosing the Right Paradigm

**Decision Tree**:
- Limited labels? → Semi-Supervised or Active Learning
- Large unlabeled data? → Self-Supervised
- Related source task? → Transfer Learning
- Privacy critical? → Federated Learning
- Need fast adaptation? → Meta-Learning

### 5.2 Real-World Applications

**Healthcare**: Federated learning across hospitals, transfer learning from general to medical images

**NLP**: Self-supervised pre-training (GPT, BERT), transfer to specific domains

**Autonomous Vehicles**: Transfer from simulation to real world, federated learning across fleets

---

## Summary: Key Takeaways

### Mathematical Foundations

1. **Semi-Supervised**: Combined loss = supervised + λ × unsupervised (consistency)
2. **Self-Supervised**: Masked language modeling, contrastive learning, autoencoding
3. **Active Learning**: Uncertainty measures (entropy, margin, least confident)
4. **Transfer Learning**: L_target = L_supervised + λ × L_transfer
5. **Federated Learning**: Weighted averaging of local updates
6. **Meta-Learning**: Learn initialization for fast adaptation (MAML)

### Key Principles

- **Leverage Unlabeled Data**: Semi-supervised and self-supervised learning
- **Strategic Labeling**: Active learning minimizes cost
- **Knowledge Transfer**: Transfer learning reuses representations
- **Privacy Preservation**: Federated learning keeps data local
- **Learning Efficiency**: Meta-learning enables fast adaptation

---

## Practice Problems & Exercises

### Mathematical Derivations

1. **Derive the gradient of contrastive loss** with respect to anchor representation
2. **Show that federated averaging** minimizes global objective
3. **Prove that MAML meta-gradient** requires second-order derivatives
4. **Calculate information gain** when adding pseudo-labels in semi-supervised learning

### Conceptual Questions

1. **What's the difference between semi-supervised and self-supervised learning?**
2. **Why does co-training require independent feature views?**
3. **How does active learning reduce labeling cost?**
4. **When would transfer learning hurt performance (negative transfer)?**

---

**End of Advanced Learning Paradigms Guide**

*This comprehensive guide includes detailed step-by-step mathematical derivations with explanations for each equation, covering semi-supervised learning, self-supervised learning, and other advanced learning paradigms.*

