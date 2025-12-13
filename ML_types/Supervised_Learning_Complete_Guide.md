# Supervised Learning: Complete Guide from Scratch to Pro
**Topic 2.1: Supervised Learning**  
*Comprehensive guide with complete mathematical derivations - every step explained*

---

## Table of Contents
1. [Foundations of Supervised Learning](#1-foundations-of-supervised-learning)
2. [Linear Regression](#2-linear-regression)
3. [Logistic Regression](#3-logistic-regression)
4. [Support Vector Machines (SVM)](#4-support-vector-machines-svm)
5. [Naive Bayes](#5-naive-bayes)
6. [K-Nearest Neighbors (KNN)](#6-k-nearest-neighbors-knn)
7. [Decision Trees](#7-decision-trees)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Optimization & Training](#9-optimization--training)
10. [Advanced Topics & Best Practices](#10-advanced-topics--best-practices)

---

## 1. Foundations of Supervised Learning

### 1.1 What is Supervised Learning?

**Definition:**
Supervised learning uses labeled data (categorical or numerical) to learn a mapping function from inputs (features) to outputs (labels).

**Core Relationship:**
```
Y = f(X) + ε
```
Where:
- **Y**: Response variable (what we want to predict)
- **X**: Features (X₁, X₂, ..., Xₚ) - input variables
- **f**: Unknown function we're trying to learn
- **ε**: Random error with mean zero (noise)

### 1.2 Two Main Tasks

#### **Classification: Predict Categorical Labels**
- **Goal**: Assign input to one of several discrete categories
- **Output**: Discrete class labels

#### **Regression: Predict Continuous Values**
- **Goal**: Predict a continuous numerical value
- **Output**: Continuous real numbers

---

## 2. Linear Regression

### 2.1 Intuition & Goal

**Goal**: Find the best straight line (or hyperplane) that fits the data to predict continuous values.

### 2.2 Mathematical Foundation

#### **Simple Linear Regression (1 feature)**
```
Y = wX + b
```

#### **Multiple Linear Regression (p features)**
```
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε
```

**Vector Form:**
```
Y = w·x + b = w₁x₁ + w₂x₂ + ... + wₚxₚ + b
```

### 2.3 Complete Derivation: Mean Squared Error Loss

#### **Step 1: Define the Loss Function**

For n training examples, we want to minimize the prediction error:

```
L(w, b) = (1/n) Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²
```

**Explanation**: 
- We sum squared differences between actual (yᵢ) and predicted (ŷᵢ) values
- Divide by n to get average (mean)
- Squaring ensures positive values and penalizes large errors more

**Substitute prediction formula**:
```
ŷᵢ = w·xᵢ + b = w₁xᵢ₁ + w₂xᵢ₂ + ... + wₚxᵢₚ + b
```

**Therefore**:
```
L(w, b) = (1/n) Σᵢ₌₁ⁿ (yᵢ - (w·xᵢ + b))²
```

**Why squared error?**
- Mathematically convenient (differentiable everywhere)
- Under Gaussian noise assumption, maximizing likelihood = minimizing MSE
- Penalizes large errors more than small ones

### 2.4 Complete Derivation: Closed-Form Solution (Simple Case)

#### **For Simple Linear Regression (1 feature): Y = wX + b**

**Step 1: Expand the Loss Function**

```
L(w, b) = (1/n) Σᵢ₌₁ⁿ (yᵢ - w·xᵢ - b)²
```

**Step 2: Take Partial Derivatives**

To find minimum, set derivatives to zero:

**Partial derivative with respect to w:**
```
∂L/∂w = (1/n) Σᵢ₌₁ⁿ 2(yᵢ - w·xᵢ - b)(-xᵢ)
      = -(2/n) Σᵢ₌₁ⁿ (yᵢ - w·xᵢ - b)·xᵢ
```

**Explanation**: 
- Chain rule: derivative of (u)² is 2u·(du/dw)
- Here u = (yᵢ - w·xᵢ - b), so du/dw = -xᵢ
- The negative sign comes from the derivative of -w·xᵢ with respect to w

**Partial derivative with respect to b:**
```
∂L/∂b = (1/n) Σᵢ₌₁ⁿ 2(yᵢ - w·xᵢ - b)(-1)
      = -(2/n) Σᵢ₌₁ⁿ (yᵢ - w·xᵢ - b)
```

**Explanation**:
- Similar chain rule, but derivative of -b with respect to b is -1

**Step 3: Set Derivatives to Zero**

**For w:**
```
∂L/∂w = 0
-(2/n) Σᵢ₌₁ⁿ (yᵢ - w·xᵢ - b)·xᵢ = 0
```

**Multiply both sides by -n/2:**
```
Σᵢ₌₁ⁿ (yᵢ - w·xᵢ - b)·xᵢ = 0
```

**Expand:**
```
Σᵢ₌₁ⁿ yᵢ·xᵢ - w·Σᵢ₌₁ⁿ xᵢ² - b·Σᵢ₌₁ⁿ xᵢ = 0
```

**For b:**
```
∂L/∂b = 0
-(2/n) Σᵢ₌₁ⁿ (yᵢ - w·xᵢ - b) = 0
```

**Multiply both sides by -n/2:**
```
Σᵢ₌₁ⁿ (yᵢ - w·xᵢ - b) = 0
```

**Expand:**
```
Σᵢ₌₁ⁿ yᵢ - w·Σᵢ₌₁ⁿ xᵢ - n·b = 0
```

**Step 4: Solve the System of Equations**

**From the b equation:**
```
b = (1/n)(Σᵢ₌₁ⁿ yᵢ - w·Σᵢ₌₁ⁿ xᵢ)
b = ȳ - w·x̄
```

**Explanation**:
- ȳ = (1/n)Σyᵢ (mean of y)
- x̄ = (1/n)Σxᵢ (mean of x)
- This shows b centers the line at the mean point

**Substitute into w equation:**
```
Σᵢ₌₁ⁿ yᵢ·xᵢ - w·Σᵢ₌₁ⁿ xᵢ² - (ȳ - w·x̄)·Σᵢ₌₁ⁿ xᵢ = 0
```

**Expand the last term:**
```
Σᵢ₌₁ⁿ yᵢ·xᵢ - w·Σᵢ₌₁ⁿ xᵢ² - ȳ·Σᵢ₌₁ⁿ xᵢ + w·x̄·Σᵢ₌₁ⁿ xᵢ = 0
```

**Note**: x̄·Σxᵢ = (1/n)Σxᵢ · Σxᵢ = (1/n)(Σxᵢ)²

**Rearrange to isolate w:**
```
w·(Σᵢ₌₁ⁿ xᵢ² - (1/n)(Σᵢ₌₁ⁿ xᵢ)²) = Σᵢ₌₁ⁿ yᵢ·xᵢ - ȳ·Σᵢ₌₁ⁿ xᵢ
```

**Step 5: Final Formula**

**For w:**
```
w = [Σᵢ₌₁ⁿ yᵢ·xᵢ - ȳ·Σᵢ₌₁ⁿ xᵢ] / [Σᵢ₌₁ⁿ xᵢ² - (1/n)(Σᵢ₌₁ⁿ xᵢ)²]
```

**Alternative form using covariance and variance:**
```
w = Cov(X, Y) / Var(X) = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
```

**Explanation**:
- Numerator: Covariance between X and Y (how they vary together)
- Denominator: Variance of X (spread of X values)
- Slope = how much Y changes per unit change in X

**For b:**
```
b = ȳ - w·x̄
```

**Geometric interpretation**: Line passes through the mean point (x̄, ȳ)

### 2.5 Gradient Descent Derivation for Linear Regression

#### **Step 1: Loss Function**
```
L(w, b) = (1/n) Σᵢ₌₁ⁿ (yᵢ - w·xᵢ - b)²
```

#### **Step 2: Compute Gradients**

**Gradient with respect to w:**
```
∂L/∂w = (1/n) Σᵢ₌₁ⁿ 2(yᵢ - w·xᵢ - b)(-xᵢ)
      = -(2/n) Σᵢ₌₁ⁿ (yᵢ - w·xᵢ - b)·xᵢ
```

**Explanation**:
- Apply chain rule: d/dw[(yᵢ - w·xᵢ - b)²]
- = 2(yᵢ - w·xᵢ - b) × d/dw(yᵢ - w·xᵢ - b)
- = 2(yᵢ - w·xᵢ - b) × (-xᵢ)
- Sum over all examples and average

**Gradient with respect to b:**
```
∂L/∂b = (1/n) Σᵢ₌₁ⁿ 2(yᵢ - w·xᵢ - b)(-1)
      = -(2/n) Σᵢ₌₁ⁿ (yᵢ - w·xᵢ - b)
```

**Explanation**:
- Similar process, but d/db(-b) = -1

#### **Step 3: Update Rules**

**Gradient descent update:**
```
w = w - η × (∂L/∂w)
b = b - η × (∂L/∂b)
```

**Substitute gradients:**
```
w = w + η × (2/n) Σᵢ₌₁ⁿ (yᵢ - w·xᵢ - b)·xᵢ
b = b + η × (2/n) Σᵢ₌₁ⁿ (yᵢ - w·xᵢ - b)
```

**Explanation**:
- Move in opposite direction of gradient (negative gradient = descent)
- η (eta) is learning rate (step size)
- Factor of 2 often absorbed into learning rate

**Why this works**:
- If gradient is positive, w is too large → decrease w
- If gradient is negative, w is too small → increase w
- Repeats until gradient ≈ 0 (minimum reached)

---

## 3. Logistic Regression

### 3.1 Intuition & Goal

**Goal**: Predict probability that an instance belongs to a class.

### 3.2 Complete Derivation: From Linear to Probability

#### **Step 1: Start with Linear Combination**

We want to model: log-odds = w·x + b

**Why log-odds?**
- Probabilities are bounded [0, 1]
- Log-odds can be any real number (-∞ to +∞)
- Can use linear model on log-odds space

#### **Step 2: Define Log-Odds**

**Odds**: Ratio of probability to its complement
```
Odds = P / (1-P)
```

**Log-Odds (Logit)**:
```
log-odds = log(P / (1-P))
```

**Properties**:
- When P = 0.5: log-odds = log(1) = 0
- When P → 1: log-odds → +∞
- When P → 0: log-odds → -∞

#### **Step 3: Model Log-Odds Linearly**

```
log(P / (1-P)) = w·x + b
```

**Explanation**: We model log-odds as a linear function of features

#### **Step 4: Solve for Probability P**

**Exponentiate both sides:**
```
P / (1-P) = e^(w·x + b)
```

**Explanation**: e^log(a) = a, so e^log(P/(1-P)) = P/(1-P)

**Solve for P:**
```
P = (1-P) × e^(w·x + b)
P = e^(w·x + b) - P × e^(w·x + b)
P + P × e^(w·x + b) = e^(w·x + b)
P(1 + e^(w·x + b)) = e^(w·x + b)
```

**Final step:**
```
P = e^(w·x + b) / (1 + e^(w·x + b))
```

**Multiply numerator and denominator by e^-(w·x + b):**
```
P = [e^(w·x + b) × e^-(w·x + b)] / [(1 + e^(w·x + b)) × e^-(w·x + b)]
P = 1 / [e^-(w·x + b) + 1]
P = 1 / (1 + e^-(w·x + b))
```

**This is the sigmoid function!**

#### **Step 5: Sigmoid Function**

```
σ(z) = 1 / (1 + e^(-z))
```

Where z = w·x + b

**Properties**:
- σ(0) = 1/2 = 0.5
- σ(z) → 1 as z → +∞
- σ(z) → 0 as z → -∞
- S-shaped curve
- Always between 0 and 1

### 3.3 Complete Derivation: Maximum Likelihood Estimation

#### **Step 1: Likelihood Function**

For binary classification, each example follows Bernoulli distribution:
```
P(yᵢ|xᵢ) = pᵢ^yᵢ × (1-pᵢ)^(1-yᵢ)
```

Where:
- pᵢ = P(Y=1|xᵢ) = σ(w·xᵢ + b)
- yᵢ ∈ {0, 1}

**Explanation**:
- If yᵢ = 1: P = pᵢ¹ × (1-pᵢ)⁰ = pᵢ
- If yᵢ = 0: P = pᵢ⁰ × (1-pᵢ)¹ = 1-pᵢ
- This compact form handles both cases

#### **Step 2: Joint Likelihood**

For n independent examples:
```
L(w, b) = Πᵢ₌₁ⁿ pᵢ^yᵢ × (1-pᵢ)^(1-yᵢ)
```

**Explanation**: Multiply probabilities (assuming independence)

#### **Step 3: Log-Likelihood**

**Take logarithm** (converts product to sum, easier to optimize):
```
log L(w, b) = Σᵢ₌₁ⁿ log[pᵢ^yᵢ × (1-pᵢ)^(1-yᵢ)]
            = Σᵢ₌₁ⁿ [yᵢ·log(pᵢ) + (1-yᵢ)·log(1-pᵢ)]
```

**Explanation**:
- log(ab) = log(a) + log(b)
- log(a^b) = b·log(a)
- Expand: log[p^y × (1-p)^(1-y)] = y·log(p) + (1-y)·log(1-p)

#### **Step 4: Substitute Sigmoid**

**Recall**: pᵢ = σ(zᵢ) = 1 / (1 + e^(-zᵢ)) where zᵢ = w·xᵢ + b

**Also**: 1 - pᵢ = 1 - 1/(1+e^(-zᵢ)) = e^(-zᵢ)/(1+e^(-zᵢ)) = 1/(1+e^(zᵢ))

**Therefore**:
```
log L(w, b) = Σᵢ₌₁ⁿ [yᵢ·log(σ(zᵢ)) + (1-yᵢ)·log(1-σ(zᵢ))]
```

#### **Step 5: Simplify Using Sigmoid Properties**

**Key identity**: log(σ(z)) = -log(1 + e^(-z))

**Also**: log(1-σ(z)) = log(e^(-z)/(1+e^(-z))) = -z - log(1+e^(-z))

**Substitute**:
```
log L(w, b) = Σᵢ₌₁ⁿ [yᵢ·(-log(1+e^(-zᵢ))) + (1-yᵢ)·(-zᵢ - log(1+e^(-zᵢ)))]
            = Σᵢ₌₁ⁿ [-yᵢ·log(1+e^(-zᵢ)) - (1-yᵢ)·zᵢ - (1-yᵢ)·log(1+e^(-zᵢ))]
            = Σᵢ₌₁ⁿ [-(1-yᵢ)·zᵢ - log(1+e^(-zᵢ))]
```

**Further simplification**:
```
log L(w, b) = Σᵢ₌₁ⁿ [-(1-yᵢ)·zᵢ - log(1+e^(-zᵢ))]
            = Σᵢ₌₁ⁿ [-zᵢ + yᵢ·zᵢ - log(1+e^(-zᵢ))]
            = Σᵢ₌₁ⁿ [yᵢ·zᵢ - zᵢ - log(1+e^(-zᵢ))]
```

**Use identity**: z - log(1+e^(-z)) = log(1+e^z)

**Final form**:
```
log L(w, b) = Σᵢ₌₁ⁿ [yᵢ·zᵢ - log(1+e^zᵢ)]
```

Where zᵢ = w·xᵢ + b

#### **Step 6: Convert to Loss Function (Negative Log-Likelihood)**

**Maximize likelihood = Minimize negative log-likelihood**

```
Loss(w, b) = -log L(w, b) = -Σᵢ₌₁ⁿ [yᵢ·zᵢ - log(1+e^zᵢ)]
            = Σᵢ₌₁ⁿ [-yᵢ·zᵢ + log(1+e^zᵢ)]
```

**This is the log loss (cross-entropy loss)!**

**Alternative common form**:
```
Loss(w, b) = Σᵢ₌₁ⁿ [-yᵢ·log(σ(zᵢ)) - (1-yᵢ)·log(1-σ(zᵢ))]
```

### 3.4 Complete Derivation: Gradient of Log Loss

#### **Step 1: Loss Function**
```
L(w, b) = Σᵢ₌₁ⁿ [-yᵢ·log(σ(zᵢ)) - (1-yᵢ)·log(1-σ(zᵢ))]
```

Where zᵢ = w·xᵢ + b

#### **Step 2: Derivative of Sigmoid**

**Key result**: dσ/dz = σ(z)(1-σ(z))

**Proof**:
```
σ(z) = 1/(1+e^(-z))
dσ/dz = d/dz[(1+e^(-z))^(-1)]
      = -1 × (1+e^(-z))^(-2) × (-e^(-z))
      = e^(-z) / (1+e^(-z))²
      = [e^(-z)/(1+e^(-z))] × [1/(1+e^(-z))]
      = [1 - 1/(1+e^(-z))] × σ(z)
      = (1-σ(z)) × σ(z)
      = σ(z)(1-σ(z))
```

**Explanation**: 
- Apply chain rule
- Simplify using σ(z) = 1/(1+e^(-z))
- Result: σ'(z) = σ(z)(1-σ(z))

#### **Step 3: Gradient with Respect to w**

```
∂L/∂w = Σᵢ₌₁ⁿ ∂/∂w[-yᵢ·log(σ(zᵢ)) - (1-yᵢ)·log(1-σ(zᵢ))]
```

**Compute each term**:

**Term 1**: ∂/∂w[-yᵢ·log(σ(zᵢ))]
```
= -yᵢ × (1/σ(zᵢ)) × (∂σ(zᵢ)/∂w)
= -yᵢ × (1/σ(zᵢ)) × σ(zᵢ)(1-σ(zᵢ)) × (∂zᵢ/∂w)
= -yᵢ × (1-σ(zᵢ)) × xᵢ
```

**Explanation**:
- Chain rule: d/dw[log(σ)] = (1/σ) × (dσ/dw)
- dσ/dw = σ'(z) × (dz/dw) = σ(1-σ) × xᵢ
- Since zᵢ = w·xᵢ + b, ∂zᵢ/∂w = xᵢ

**Term 2**: ∂/∂w[-(1-yᵢ)·log(1-σ(zᵢ))]
```
= -(1-yᵢ) × (1/(1-σ(zᵢ))) × (∂(1-σ(zᵢ))/∂w)
= -(1-yᵢ) × (1/(1-σ(zᵢ))) × (-σ(zᵢ)(1-σ(zᵢ))) × xᵢ
= (1-yᵢ) × σ(zᵢ) × xᵢ
```

**Explanation**:
- Similar chain rule
- d(1-σ)/dw = -dσ/dw = -σ(1-σ) × xᵢ
- Negative signs cancel

**Combine terms**:
```
∂L/∂w = Σᵢ₌₁ⁿ [-yᵢ(1-σ(zᵢ))xᵢ + (1-yᵢ)σ(zᵢ)xᵢ]
      = Σᵢ₌₁ⁿ [σ(zᵢ)xᵢ - yᵢxᵢ]
      = Σᵢ₌₁ⁿ (σ(zᵢ) - yᵢ)xᵢ
```

**Explanation**:
- Expand: -yᵢ(1-σ) + (1-yᵢ)σ = -yᵢ + yᵢσ + σ - yᵢσ = σ - yᵢ
- Final form: (prediction - actual) × feature

#### **Step 4: Gradient with Respect to b**

**Similar process**:
```
∂L/∂b = Σᵢ₌₁ⁿ (σ(zᵢ) - yᵢ)
```

**Explanation**: Same as w gradient, but xᵢ = 1 (since z = w·x + b, ∂z/∂b = 1)

#### **Step 5: Update Rules**

```
w = w - η × Σᵢ₌₁ⁿ (σ(zᵢ) - yᵢ)xᵢ
b = b - η × Σᵢ₌₁ⁿ (σ(zᵢ) - yᵢ)
```

**Interpretation**: Update proportional to prediction error (σ(zᵢ) - yᵢ)

---

## 4. Support Vector Machines (SVM)

### 4.1 Intuition & Goal

**Goal**: Find the best separating hyperplane with maximum margin.

### 4.2 Complete Derivation: Margin Calculation

#### **Step 1: Hyperplane Equation**

**Decision boundary**: w·x + b = 0

**Explanation**:
- w is normal vector (perpendicular to hyperplane)
- b is offset from origin
- Points on one side: w·x + b > 0
- Points on other side: w·x + b < 0

#### **Step 2: Distance from Point to Hyperplane**

**Given point x₀, find distance to hyperplane w·x + b = 0**

**Step 2a: Find closest point on hyperplane**

Let x' be closest point on hyperplane to x₀. Then:
- x' = x₀ - t·w (move along normal direction)
- w·x' + b = 0 (x' is on hyperplane)

**Substitute**:
```
w·(x₀ - t·w) + b = 0
w·x₀ - t(w·w) + b = 0
t = (w·x₀ + b) / ||w||²
```

**Explanation**:
- Move from x₀ toward hyperplane along normal w
- Distance is t × ||w||
- Solve for t using hyperplane equation

**Step 2b: Calculate distance**

```
distance = ||x₀ - x'|| = ||t·w|| = |t| × ||w||
         = |(w·x₀ + b)| / ||w||² × ||w||
         = |w·x₀ + b| / ||w||
```

**Explanation**:
- Distance is magnitude of vector from x₀ to x'
- Substitute t and simplify

#### **Step 3: Margin Definition**

**Margin**: Distance between two parallel hyperplanes that separate classes

**Hyperplanes**:
- w·x + b = +1 (positive class boundary)
- w·x + b = -1 (negative class boundary)

**Distance between them**:
```
margin = |(+1) - (-1)| / ||w|| = 2 / ||w||
```

**Explanation**:
- Two parallel planes: ax + by + c₁ = 0 and ax + by + c₂ = 0
- Distance = |c₁ - c₂| / √(a² + b²)
- Here: |1 - (-1)| / ||w|| = 2/||w||

### 4.3 Complete Derivation: Optimization Problem

#### **Step 1: Constraints**

**All points must be correctly classified and outside margin**:

For positive class (yᵢ = +1):
```
w·xᵢ + b ≥ +1
```

For negative class (yᵢ = -1):
```
w·xᵢ + b ≤ -1
```

**Combine into single constraint**:
```
yᵢ(w·xᵢ + b) ≥ 1  for all i
```

**Explanation**:
- If yᵢ = +1: w·xᵢ + b ≥ 1 ✓
- If yᵢ = -1: Multiply by -1: -(w·xᵢ + b) ≥ 1, so w·xᵢ + b ≤ -1 ✓
- Single constraint handles both cases

#### **Step 2: Objective Function**

**Goal**: Maximize margin = Minimize ||w||

**Mathematical form**:
```
Minimize: (1/2)||w||²
Subject to: yᵢ(w·xᵢ + b) ≥ 1 for all i
```

**Why (1/2)||w||² instead of ||w||?**
- Equivalent (minimizing ||w||² minimizes ||w||)
- ||w||² = w·w is differentiable everywhere
- 1/2 makes derivatives cleaner (no factor of 2)

#### **Step 3: Lagrange Multipliers**

**Convert constrained to unconstrained optimization**

**Lagrangian**:
```
L(w, b, α) = (1/2)||w||² - Σᵢ₌₁ⁿ αᵢ[yᵢ(w·xᵢ + b) - 1]
```

Where:
- αᵢ ≥ 0 are Lagrange multipliers
- Each constraint gets one multiplier

**Explanation**:
- Original: minimize f(w) subject to gᵢ(w) ≥ 0
- Lagrangian: L = f(w) - Σαᵢgᵢ(w)
- At optimum, constraints are satisfied and gradients align

#### **Step 4: Karush-Kuhn-Tucker (KKT) Conditions**

**Condition 1: Stationarity** (gradients = 0)

**With respect to w**:
```
∂L/∂w = w - Σᵢ₌₁ⁿ αᵢyᵢxᵢ = 0
```

**Therefore**:
```
w = Σᵢ₌₁ⁿ αᵢyᵢxᵢ
```

**Explanation**:
- Gradient of (1/2)||w||² = w
- Gradient of constraint terms = -Σαᵢyᵢxᵢ
- Set sum to zero

**With respect to b**:
```
∂L/∂b = -Σᵢ₌₁ⁿ αᵢyᵢ = 0
```

**Therefore**:
```
Σᵢ₌₁ⁿ αᵢyᵢ = 0
```

**Explanation**: Constraint on multipliers

**Condition 2: Primal Feasibility**
```
yᵢ(w·xᵢ + b) ≥ 1  for all i
```

**Condition 3: Dual Feasibility**
```
αᵢ ≥ 0  for all i
```

**Condition 4: Complementary Slackness**
```
αᵢ[yᵢ(w·xᵢ + b) - 1] = 0  for all i
```

**Explanation**:
- Either αᵢ = 0 (constraint not active)
- Or yᵢ(w·xᵢ + b) = 1 (point is support vector)

#### **Step 5: Dual Formulation**

**Substitute w back into Lagrangian**:

```
L(w, b, α) = (1/2)||w||² - Σᵢ αᵢ[yᵢ(w·xᵢ + b) - 1]
```

**Substitute w = Σⱼ αⱼyⱼxⱼ**:
```
= (1/2)(Σᵢ αᵢyᵢxᵢ)·(Σⱼ αⱼyⱼxⱼ) - Σᵢ αᵢ[yᵢ(Σⱼ αⱼyⱼxⱼ·xᵢ + b) - 1]
```

**Expand first term**:
```
(1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ)
```

**Expand second term**:
```
- Σᵢ αᵢyᵢ(Σⱼ αⱼyⱼxⱼ·xᵢ) - bΣᵢ αᵢyᵢ + Σᵢ αᵢ
```

**Note**: Σᵢ αᵢyᵢ = 0 (from KKT condition)

**Simplify**:
```
= (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ) - ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ) + Σᵢ αᵢ
= -(1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ) + Σᵢ αᵢ
```

**Dual problem** (maximize Lagrangian):
```
Maximize: Σᵢ αᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ)
Subject to: Σᵢ αᵢyᵢ = 0, αᵢ ≥ 0
```

**Explanation**:
- Dual is easier to solve (fewer constraints)
- Only depends on dot products xᵢ·xⱼ (enables kernel trick!)
- αᵢ > 0 only for support vectors

#### **Step 6: Support Vectors**

**From complementary slackness**: αᵢ > 0 only when yᵢ(w·xᵢ + b) = 1

**These are the support vectors** - points on the margin boundaries

**Prediction**:
```
f(x) = w·x + b = (Σᵢ αᵢyᵢxᵢ)·x + b
     = Σᵢ αᵢyᵢ(xᵢ·x) + b
```

**Only need support vectors (αᵢ > 0) for prediction!**

---

## 5. Naive Bayes

### 5.1 Intuition & Goal

**Goal**: Predict class using Bayes' theorem with independence assumption.

### 5.2 Complete Derivation: Bayes' Theorem

#### **Step 1: Conditional Probability Definition**

**Conditional probability**:
```
P(Y|X) = P(Y and X) / P(X)
```

**Also**:
```
P(X|Y) = P(X and Y) / P(Y)
```

#### **Step 2: Joint Probability**

**From conditional probability**:
```
P(Y and X) = P(Y|X) × P(X)
P(X and Y) = P(X|Y) × P(Y)
```

**Since P(Y and X) = P(X and Y)**:
```
P(Y|X) × P(X) = P(X|Y) × P(Y)
```

#### **Step 3: Bayes' Theorem**

**Solve for P(Y|X)**:
```
P(Y|X) = P(X|Y) × P(Y) / P(X)
```

**Components**:
- **P(Y|X)**: Posterior (what we want)
- **P(X|Y)**: Likelihood (probability of features given class)
- **P(Y)**: Prior (how common the class is)
- **P(X)**: Evidence (normalizing constant)

#### **Step 4: Proportional Form**

**For classification, P(X) is constant across classes**:
```
P(Y|X) ∝ P(X|Y) × P(Y)
```

**Explanation**: We only need relative probabilities to choose class

### 5.3 Complete Derivation: Naive Independence Assumption

#### **Step 1: Multiple Features**

**With p features**:
```
P(Y|X₁, X₂, ..., Xₚ) = P(X₁, X₂, ..., Xₚ|Y) × P(Y) / P(X₁, X₂, ..., Xₚ)
```

#### **Step 2: Independence Assumption**

**Naive assumption**: Features independent given class
```
P(X₁, X₂, ..., Xₚ|Y) = P(X₁|Y) × P(X₂|Y) × ... × P(Xₚ|Y)
                      = Πᵢ₌₁ᵖ P(Xᵢ|Y)
```

**Explanation**: Joint probability = product of marginals (independence)

#### **Step 3: Naive Bayes Formula**

**Substitute into Bayes' theorem**:
```
P(Y|X₁, ..., Xₚ) ∝ P(Y) × Πᵢ₌₁ᵖ P(Xᵢ|Y)
```

**For classification**:
```
Predicted Class = argmax_Y [P(Y) × Πᵢ₌₁ᵖ P(Xᵢ|Y)]
```

### 5.4 Complete Derivation: Laplace Smoothing

#### **Step 1: Maximum Likelihood Estimate (Without Smoothing)**

**For categorical feature Xᵢ with value v and class Y = k**:
```
P(Xᵢ = v|Y = k) = Count(Xᵢ = v AND Y = k) / Count(Y = k)
```

**Problem**: If Count = 0, then P = 0, making entire product zero

#### **Step 2: Additive Smoothing**

**Add small constant α to counts**:
```
P(Xᵢ = v|Y = k) = (Count(Xᵢ = v AND Y = k) + α) / (Count(Y = k) + α × |V|)
```

Where:
- |V| = number of possible values for Xᵢ
- α = smoothing parameter (typically α = 1)

**Explanation**:
- Add α to numerator (pseudo-counts)
- Add α×|V| to denominator (normalize)
- Prevents zero probabilities
- α = 1 is Laplace smoothing

#### **Step 3: Why This Works**

**Intuition**: Assume we've seen each value α times before

**Bayesian interpretation**: 
- Prior: uniform distribution over values
- Posterior: combine prior (α counts) with data
- As data increases, prior influence decreases

---

## 6. K-Nearest Neighbors (KNN)

### 6.1 Intuition & Goal

**Goal**: Predict based on similarity to nearest training examples.

### 6.2 Complete Derivation: Distance Metrics

#### **Euclidean Distance (L2)**

**Definition**:
```
d(u, v) = √[Σᵢ₌₁ⁿ (uᵢ - vᵢ)²]
```

**Derivation from Pythagorean theorem**:
- In 2D: d = √[(u₁-v₁)² + (u₂-v₂)²]
- Generalize to n dimensions
- Measures straight-line distance

#### **Manhattan Distance (L1)**

**Definition**:
```
d(u, v) = Σᵢ₌₁ⁿ |uᵢ - vᵢ|
```

**Derivation**: Sum of absolute differences along each dimension

**Geometric interpretation**: Like city blocks (can't cut diagonally)

#### **Minkowski Distance (General)**

**Definition**:
```
d(u, v) = [Σᵢ₌₁ⁿ |uᵢ - vᵢ|ᵖ]^(1/p)
```

**Special cases**:
- p = 1: Manhattan
- p = 2: Euclidean
- p → ∞: Chebyshev (max |uᵢ - vᵢ|)

### 6.3 Complete Derivation: Weighted KNN

#### **Step 1: Inverse Distance Weighting**

**Weight for neighbor i**:
```
wᵢ = 1 / dᵢ
```

Where dᵢ is distance to neighbor i

**Explanation**: Closer neighbors get higher weights

#### **Step 2: Weighted Prediction (Regression)**

**Weighted average**:
```
ŷ = (Σᵢ₌₁ᵏ wᵢ × yᵢ) / (Σᵢ₌₁ᵏ wᵢ)
```

**Explanation**:
- Numerator: Sum of weighted values
- Denominator: Sum of weights (normalization)
- Ensures prediction is in valid range

#### **Step 3: Weighted Prediction (Classification)**

**Weighted voting**:
```
ŷ = argmax_c [Σᵢ₌₁ᵏ wᵢ × I(yᵢ = c)]
```

Where I(yᵢ = c) = 1 if neighbor i has class c, else 0

**Explanation**: Sum weights for each class, choose class with highest sum

---

## 7. Decision Trees

### 7.1 Intuition & Goal

**Goal**: Make decisions through sequential questions.

### 7.2 Complete Derivation: Entropy

#### **Step 1: Information Content**

**Shannon's information content**:
```
I(event) = -log₂(P(event))
```

**Explanation**:
- Rare events (low P) → high information
- Common events (high P) → low information
- Base 2 gives bits

#### **Step 2: Expected Information (Entropy)**

**For random variable Y with outcomes y₁, ..., yₖ**:
```
H(Y) = E[I(Y)] = Σᵢ₌₁ᵏ P(yᵢ) × I(yᵢ)
                = -Σᵢ₌₁ᵏ P(yᵢ) × log₂(P(yᵢ))
```

**Explanation**:
- Expected value of information content
- Weighted average by probabilities
- Measures uncertainty

#### **Step 3: Properties**

**Maximum entropy** (uniform distribution):
- P(yᵢ) = 1/k for all i
- H(Y) = -k × (1/k) × log₂(1/k) = log₂(k)

**Minimum entropy** (deterministic):
- One outcome has P = 1, others P = 0
- H(Y) = -1 × log₂(1) - 0 = 0

### 7.3 Complete Derivation: Information Gain

#### **Step 1: Conditional Entropy**

**Entropy of Y given X**:
```
H(Y|X) = Σₓ P(x) × H(Y|X=x)
```

**Explanation**: Weighted average of entropies in each subset

#### **Step 2: Information Gain**

**Definition**:
```
IG(Y|X) = H(Y) - H(Y|X)
```

**Explanation**: Reduction in uncertainty after knowing X

#### **Step 3: Detailed Calculation**

**Given split on feature X creating subsets S₁, ..., Sₘ**:
```
H(Y|X) = Σⱼ₌₁ᵐ (|Sⱼ|/|S|) × H(Sⱼ)
```

Where:
- |Sⱼ| = size of subset j
- |S| = total size
- H(Sⱼ) = entropy of subset j

**Information gain**:
```
IG(Y|X) = H(S) - Σⱼ (|Sⱼ|/|S|) × H(Sⱼ)
```

**Explanation**: 
- H(S): entropy before split
- Second term: weighted entropy after split
- Difference: information gained

### 7.4 Complete Derivation: Gini Index

#### **Step 1: Definition**

**Gini impurity**:
```
Gini(D) = 1 - Σᵢ₌₁ᵏ (pᵢ)²
```

Where pᵢ is proportion of class i

#### **Step 2: Derivation from Variance**

**For binary classification** (Y ∈ {0, 1}):
- Variance: Var(Y) = E[Y²] - (E[Y])²
- E[Y] = p (proportion of class 1)
- E[Y²] = p (since Y² = Y for binary)
- Var(Y) = p - p² = p(1-p)

**Gini = 2 × Var(Y) = 2p(1-p)**

**For k classes**:
```
Gini = Σᵢ pᵢ(1-pᵢ) = Σᵢ pᵢ - Σᵢ pᵢ² = 1 - Σᵢ pᵢ²
```

**Explanation**: Sum variances for each class

#### **Step 3: Properties**

**Maximum** (uniform): pᵢ = 1/k → Gini = 1 - k(1/k)² = 1 - 1/k

**Minimum** (pure): One pᵢ = 1, others 0 → Gini = 1 - 1 = 0

---

## 8. Evaluation Metrics

### 8.1 Classification Metrics

#### **Precision Derivation**

**Definition**:
```
Precision = TP / (TP + FP)
```

**Explanation**: Of all positive predictions, how many are correct?

**Derivation from conditional probability**:
- P(Actually Positive | Predicted Positive)
- = P(Positive and Predicted Positive) / P(Predicted Positive)
- = TP / (TP + FP)

#### **Recall Derivation**

**Definition**:
```
Recall = TP / (TP + FN)
```

**Explanation**: Of all actual positives, how many did we find?

**Derivation**:
- P(Predicted Positive | Actually Positive)
- = TP / (TP + FN)

#### **F1-Score Derivation**

**Harmonic mean**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Why harmonic mean?**
- Arithmetic mean: (P+R)/2 (doesn't penalize imbalance)
- Harmonic mean: 2PR/(P+R) (penalizes when one is low)
- F1 = 0 if either P or R = 0

**Alternative form**:
```
1/F1 = (1/P + 1/R) / 2
```

**Explanation**: Average of reciprocals

---

## 9. Optimization & Training

### 9.1 Complete Derivation: Gradient Descent

#### **Step 1: Taylor Series Expansion**

**Function f near point w₀**:
```
f(w) ≈ f(w₀) + f'(w₀)(w - w₀)
```

**For minimization**: Move in direction of negative gradient

#### **Step 2: Update Rule**

**From Taylor expansion**:
```
f(w₀ - η∇f) ≈ f(w₀) - η||∇f||²
```

**Explanation**: 
- Moving in -∇f direction decreases function value
- η controls step size
- ||∇f||² is always positive

**Update**:
```
w = w - η × ∇f(w)
```

#### **Step 3: Learning Rate**

**Too large**: May overshoot minimum
**Too small**: Slow convergence

**Optimal**: Largest η that still decreases loss

### 9.2 Complete Derivation: Regularization

#### **L2 Regularization (Ridge)**

**Modified loss**:
```
L(w) = Original Loss + λ × ||w||²
```

**Gradient**:
```
∇L = ∇(Original Loss) + 2λw
```

**Update**:
```
w = w - η[∇(Original Loss) + 2λw]
  = (1 - 2ηλ)w - η∇(Original Loss)
```

**Explanation**: 
- Shrinks weights by factor (1-2ηλ) each step
- Prevents weights from growing too large

#### **L1 Regularization (Lasso)**

**Modified loss**:
```
L(w) = Original Loss + λ × Σ|wᵢ|
```

**Subgradient** (since |w| not differentiable at 0):
```
∂L/∂wᵢ = ∂(Original Loss)/∂wᵢ + λ × sign(wᵢ)
```

Where sign(w) = +1 if w > 0, -1 if w < 0, 0 if w = 0

**Effect**: Can drive weights exactly to zero (sparsity)

---

## 10. Advanced Topics & Best Practices

### 10.1 Bias-Variance Decomposition

#### **Complete Derivation**

**Expected prediction error**:
```
E[(Y - Ŷ)²] = E[(Y - E[Ŷ] + E[Ŷ] - Ŷ)²]
```

**Expand**:
```
= E[(Y - E[Ŷ])²] + E[(E[Ŷ] - Ŷ)²] + 2E[(Y - E[Ŷ])(E[Ŷ] - Ŷ)]
```

**Third term is zero** (Y and Ŷ independent):
```
= E[(Y - E[Ŷ])²] + E[(E[Ŷ] - Ŷ)²]
= Var(Y) + Var(Ŷ)
```

**Add and subtract E[Y]**:
```
= E[(Y - E[Y] + E[Y] - E[Ŷ])²] + Var(Ŷ)
= Var(Y) + (E[Y] - E[Ŷ])² + Var(Ŷ)
```

**Final form**:
```
Error = Bias² + Variance + Irreducible Error
```

Where:
- **Bias²** = (E[Y] - E[Ŷ])² (systematic error)
- **Variance** = Var(Ŷ) (sensitivity to data)
- **Irreducible Error** = Var(Y) (inherent noise)

---

## Summary: Key Mathematical Derivations

### Essential Formulas with Complete Derivations

1. **Linear Regression Closed Form**: Derived from setting gradients to zero
2. **Logistic Regression Sigmoid**: Derived from log-odds transformation
3. **Log Loss**: Derived from maximum likelihood estimation
4. **SVM Margin**: Derived from distance formula to hyperplane
5. **SVM Dual Form**: Derived using Lagrange multipliers and KKT conditions
6. **Bayes' Theorem**: Derived from conditional probability definitions
7. **Naive Bayes**: Derived from independence assumption
8. **Entropy**: Derived from information theory (Shannon)
9. **Information Gain**: Derived from conditional entropy
10. **Gradient Descent**: Derived from Taylor series expansion
11. **Regularization**: Derived from constrained optimization

---

**End of Complete Mathematical Derivation Guide**

*Every mathematical step is explained with reasoning and intuition. This ensures deep understanding of why each transformation is made and how formulas are derived.*

