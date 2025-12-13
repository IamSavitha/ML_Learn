# Machine Learning (DATA 245) - Comprehensive Mindmap
**Course: DATA 245 Machine Learning**  
**Compiled from: Class Notes, Slides, and Flipped Class Materials**

---

## 1. FOUNDATIONS & INTRODUCTION

### 1.1 What is Machine Learning?
- Branch of artificial intelligence
- Algorithms that evolve behaviors based on empirical data
- Computers acquire knowledge from data
- **Different perspectives:**
  - Programmers: Machine Learning
  - Statisticians: Statistical Learning
  - Engineers: Pattern Recognition
  - Mathematicians: Curve Fitting
  - Business Leaders: Predictive Analytics
  - Database researchers: Data Mining
  - Database scientists: Data Science

### 1.2 ML Enabling Technologies
- Faster computers
- More data (web, genomes, gene expression arrays, parallel corpora)
- New ideas (kernel trick, large margins, boosting, graphical models)

### 1.3 Formal Perspective
- Relationship: Y = f(X) + ε
  - Y: response variable
  - X: features (X₁, X₂, ..., Xₚ)
  - f: unknown function
  - ε: random error with mean zero

### 1.4 ML as Modernization Driver
- Machine Learning as "mortar of modernization"
- Paradigm shift from traditional programming
- AI/ML in life-critical roles (e.g., autonomous vehicles)
- Generative AI applications (GANs, CycleGAN)

---

## 2. TYPES OF MACHINE LEARNING

### 2.1 Supervised Learning
- **Definition:** Uses labeled data (categorical/numerical)
- **Tasks:**
  - Classification: Predict categorical labels
  - Regression: Predict continuous values
- **Examples:** Linear Regression, Logistic Regression, SVM, Naive Bayes, KNN, Decision Trees

### 2.2 Unsupervised Learning
- **Definition:** No labels available
- **Tasks:**
  - Clustering: Group similar data points
  - Dimensionality reduction: Reduce feature space
  - Segmentation: Discover patterns
- **Examples:** K-Means, Hierarchical Clustering, DBSCAN, PCA

### 2.3 Semi-Supervised Learning
- Combines labeled and unlabeled data
- Few labeled examples guide learning
- Models teach each other (co-training, self-training)
- Majority voting for predictions

### 2.4 Self-Supervised Learning
- GPT uses this: mask words, predict/fill blanks
- Learn from data structure itself

### 2.5 Other Learning Paradigms
- **Multi-Model Learning:** Multiple models on different data subsets
- **Active Learning:** Model queries human (oracle) for uncertain points
- **Transfer Learning:** Pretrained model adapted to new task
- **Federated Learning:** Train on local devices, aggregate encrypted updates
- **Meta-Learning:** Learning to learn (ensemble methods)

---

## 3. DATA FOUNDATIONS

### 3.1 Data Quality & Veracity
- **Big Data's 4th V: Veracity**
  - As volume grows, quality and trustworthiness decrease
  - Reliability is major challenge
  - Problem of falsity on web is NP-hard
- **Data Labeling:**
  - Most expensive part of ML
  - Often crowdsourced
  - Risk of inaccuracy
  - Deep Learning automates more than traditional ML

### 3.2 Data Characteristics
- **Independent and Identically Distributed (i.i.d.):**
  - Features are independent
  - Come from same distribution
- **Noise in Data:**
  - ML relies on empirical data
  - Data can be noisy and inaccurate
  - Models should represent minority groups

### 3.3 Data Splits
- **Training Data:** Used to learn model
- **Test Data:** Unobserved, used for final evaluation
- **Validation Data:** Used for hyperparameter tuning
- **Train/Test Split:**
  - Default: 75:25 (can vary)
  - Data leakage: Test data must not influence training
  - Cross-validation: K-fold improves reliability
  - Randomization reduces bias

---

## 4. LINEAR MODELS

### 4.1 Linear Regression
- **Model:** Y = wX + b (single feature)
- **Multiple Linear Regression:** Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ + ε
- **Parameters:**
  - β₀: intercept (average Y when all X's are zero)
  - βⱼ: slope for jth variable (average increase in Y when Xⱼ increases by 1)
- **Interpretation:**
  - Slope indicates rate of change
  - Greater slope = more importance
- **Use Case:** Predict continuous values (e.g., University GPA from HS GPA)

### 4.2 Linear vs. Non-Linear
- **When to use Linear Regression:**
  - Linear relationships between features and target
- **When NOT to use:**
  - Non-linear relationships (e.g., income vs. education and seniority)
  - Wrong model → bad prediction

### 4.3 Logistic Regression
- **Classification method** (despite "regression" name)
- **Sigmoid Function:**
  - Maps weighted feature sums to probabilities [0, 1]
  - S-shaped curve: s = 1/(1+e^(-z))
- **Probability Foundation:**
  - Odds: P/(1-P)
  - Log-odds: log(P/(1-P))
  - Inversion via exponential function
- **Parameters:**
  - Weights (w): Stretch/tilt curve
  - Bias (b): Shifts curve left/right
  - Higher weights can overfit
- **Comparison:** For tabular data, can outperform deep learning in simplicity and interpretability

### 4.4 Generalized Linear Models (GLM)
- Extends linear models to different distributions
- Handles: Poisson, Gamma, Binomial
- Uses link functions (log, identity) to relate mean of response to predictors

---

## 5. SUPPORT VECTOR MACHINES (SVM)

### 5.1 Core Concept
- **Maximum Margin Classifier**
- Separates data into groups using best dividing line/plane
- Best line = maximum distance from nearest points of both groups
- Nearest points = **Support Vectors**

### 5.2 Mathematics
- **Equation:** w·x + b = 0
  - w: weights (direction of line)
  - b: constant (offset)
- **Optimization:** Maximize margin, keep all points correctly separated
- **Lagrange Multipliers:** Help find solution while following constraints

### 5.3 Geometric Intuition
- **Convex Hulls:** Boundary of a class
- Best line lies between class boundaries
- Convexity: Line segment between any two points in set is fully contained

### 5.4 Non-Linear SVM
- When data can't be separated by straight line
- Maps data to higher dimension where separation is possible
- **Kernel Trick:**
  - Shortcut for high-dimensional mapping
  - Common kernel: RBF (Radial Basis Function)
  - Very scalable for high-dimensional data

---

## 6. PROBABILISTIC CLASSIFIERS

### 6.1 Bayesian Classification
- **Probabilistic Classifier**
- Applies when relationship f(X) is non-deterministic
- Uses **Bayes' Theorem:**
  - P(Y|X) = P(X|Y) × P(Y) / P(X)
  - Posterior ∝ Likelihood × Prior
  - P(X) often skipped (proportionality)
- **Methodology:** Incremental learning
- **Performance:** Comparable to decision trees and neural networks
- **Use Case:** Theoretical baseline for comparison

### 6.2 Naive Bayes
- **Assumption:** All features are independent (naive assumption)
- **Formula:** P(Y|X) ∝ P(X|Y) × P(Y)
- **Types:**
  - Categorical features: Use probability function (PF)
  - Continuous features: Use probability density function (PDF)
- **Smoothing:**
  - Laplace smoothing avoids zero probability issues
  - If any feature probability is zero, entire product becomes zero
- **Impact of Feature Correlation:**
  - When features are correlated, introduces bias
  - Reduces accuracy
  - Ignores Bayes rate

### 6.3 Generative vs. Discriminative Models
- **Discriminative:**
  - Classify directly
  - Efficient (higher accuracy)
  - Less computation
  - Example: Logistic Regression
- **Generative:**
  - Model distribution, generate new data
  - Density estimation (often intractable)
  - Heavy computation
  - Example: Naive Bayes, Bayesian Classification
- Both can solve P(Y|X), but with different approaches

---

## 7. INSTANCE-BASED LEARNING

### 7.1 K-Nearest Neighbors (KNN)
- **Intuition:** "A person is known by the company they keep"
- **Approach:**
  - Instance-based/Memory-based learning
  - No abstraction or model creation
  - Training: Store feature vectors in memory (lean phase)
  - Testing: Compute similarities, choose k nearest neighbors

### 7.2 Choosing K
- **Small K (e.g., K=1):**
  - Overfitting
  - Issues with outliers and noise
  - Voronoi diagram (1-NN)
- **Large K (e.g., K=10):**
  - Underfitting
  - Hard to classify
- **Optimal K:**
  - Use cross-validation
  - Choose odd numbers (avoids ties)
  - Example: K=4 avoids overfitting

### 7.3 Distance Metrics
- **Euclidean Distance:** p = 2 in Minkowski
- **Manhattan Distance:** p = 1 in Minkowski
- **Minkowski Distance:** (Σ|uᵢ - vᵢ|ᵖ)^(1/p)
- **Chebyshev Distance:** p = ∞, max(|uᵢ - vᵢ|)
- **Cosine Similarity:** (U·V) / (||U|| × ||V||)
- **Dot Product:** Σuᵢ × vᵢ
- **Note:** Feature vectors need normalization!

### 7.4 Weighted KNN
- Closer neighbors get higher weights
- Prediction: weighted average of neighbors
- Weights = inverse of distance
- Formula: y = Σ(wᵢ × yᵢ) / Σwᵢ for i = 1 to k

### 7.5 KNN: Classification vs. Regression
- **Classification:** yₜ = sgn(Σᵢ₌₁ᵏ yᵢ)
- **Regression:** y = (1/k)(Σᵢ₌₁ᵏ yᵢ)

### 7.6 KNN Pros and Cons
- **Pros:**
  - Simple, easy to understand
  - Few hyperparameters
  - Works for nonlinear data
- **Cons:**
  - Slow for large datasets
  - Affected by high dimensions
  - Requires feature normalization
  - Large memory usage (stores all points)

---

## 8. DECISION TREES

### 8.1 Concept
- **Intuition:** Consider one factor after another (sequential decision making)
- **Structure:**
  - Internal nodes: Test attributes
  - Branches: Attribute values
  - Leaf nodes: Assign classifications

### 8.2 Entropy & Information Gain
- **Entropy (H):**
  - Measure of uncertainty or surprise
  - Formula: H = -Σ(p × log(p))
  - High entropy = more uncertainty
  - Low entropy = more information
  - Used to measure purity/impurity of data
- **Information Gain (IG):**
  - IG(Y|X) = H(Y) - H(Y|X)
  - Split with highest IG is chosen
  - Reduces uncertainty

### 8.3 Algorithms
- **ID3 Algorithm:**
  - Uses entropy and information gain
  - Binary or multiway splits
  - Works only with categorical predictors (limitation)
- **C4.5:**
  - Handles continuous and discrete data
  - Uses pruning
- **CART (Classification and Regression Trees):**
  - Supports both classification and regression
  - Uses Gini Index

### 8.4 Gini Index
- Alternative impurity measure
- Formula: Gini(D) = 1 - Σ(pᵢ)²
- Smaller Gini = purer split
- Used to minimize prediction error and overfitting

### 8.5 Bias-Variance Tradeoff
- **Overfitting:** Low bias, high variance (memorizes data)
- **Underfitting:** High bias, low variance (too simple)
- **Solution:** Pruning, complexity control, early stopping

### 8.6 Dimensionality Reduction in Trees
- Reduce unnecessary features
- Smaller attribute set improves interpretability
- Avoids overfitting
- Remove nodes/splits not contributing much

---

## 9. OPTIMIZATION & GRADIENT DESCENT

### 9.1 Optimization Problem
- **Model-based Supervised Learning:**
  1. Pick a model: y = b + Σwⱼfⱼ
  2. Pick criteria (objective function)
  3. Develop learning algorithm
- **Classification as Optimization:**
  - Minimize loss or maximize margin
  - Use convex (surrogate) loss function
  - Example: argmin Σexp(-yᵢ(w·xᵢ + b))

### 9.2 Gradient Descent Intuition
- **Blindfolded in Valley Analogy:**
  - Can see ground near feet
  - In convex-shaped valley
  - Escape at bottom/minimum
  - Move in direction of steepest descent
- **Gradient:**
  - Direction of steepest ascent
  - For descent: move opposite to gradient
  - Vector of partial derivatives
  - In 2D: derivative is slope

### 9.3 Mathematics
- **Update Rule:** wⱼ = wⱼ - η × (d loss(w) / dwⱼ)
  - η: learning rate (step size)
  - Derivative: slope of loss function
  - If slope positive → decrease weights
  - If slope negative → increase weights
- **Convex Functions:**
  - At most one minimum
  - Gradient descent won't get stuck in local minima

### 9.4 Types of Gradient Descent
- **Batch Gradient Descent:**
  - Uses all training data
  - Slower but more stable
- **Stochastic Gradient Descent (SGD):**
  - Uses random samples
  - Faster learning but noisier updates
  - Helps escape local minima
- **Mini-batch Gradient Descent:**
  - Balances efficiency and stability
  - Uses subset of data

### 9.5 Visualization
- **Contour Maps & Heatmaps:**
  - Visualize loss surfaces
  - Gradient direction perpendicular to contours
  - Always move steepest downhill direction

---

## 10. REGULARIZATION & OVERFITTING

### 10.1 Overfitting Problem
- **Definition:** Model fits training data too well, fails to generalize
- **Indicators:**
  - Very low training error
  - High error on new instances
  - Model memorizes noise in data

### 10.2 Components of Prediction Error
- **Bias:**
  - Result of simplifying assumptions
  - Parametric methods → high bias
  - Machine learning models make assumptions about real world
- **Variance:**
  - Amount estimate changes with different training data
  - Non-parametric methods → high variance
- **Irreducible Error:**
  - Cannot be reduced regardless of algorithm

### 10.3 Bias-Variance Tradeoff
- **Sweet Spot:** Balance between underfitting and overfitting
- **Performance Evaluation:**
  - Training accuracy vs. test accuracy
  - Gap indicates overfitting
- **Early Stopping:**
  - Stop training when loss stops improving
  - Prevents overfitting

### 10.4 Regularization Techniques
- **Purpose:** Add constraints to reduce overfitting
- **Mechanism:**
  - Penalize large weights
  - Add penalties to loss function using Lagrange multipliers
  - Smooth out model
  - Called "shrinkage" in statistics

### 10.5 L1 vs. L2 Regularization
- **L1 (Lasso) Regularization:**
  - Promotes sparsity
  - Some weights become exactly 0
  - Feature selection
- **L2 (Ridge) Regularization:**
  - Shrinks weights
  - Keeps all weights nonzero
  - Smoothness
- **Elastic Net:**
  - Combines L1 and L2
  - Balanced feature selection and smoothness

### 10.6 Loss Functions with Regularization
- **Least Squares:** No regularization, simple squared loss
- **Ridge Regression:** L2 penalty
- **Lasso Regression:** L1 penalty
- **Elastic Net:** Mixes L1 and L2
- **Logistic Regression:** Log loss for classification

### 10.7 Model Complexity Control
- Complexity should match the problem
- Constrain model weights
- Limit large coefficients
- Convex loss functions ensure stability

---

## 11. DIMENSIONALITY REDUCTION

### 11.1 Curse of Dimensionality
- **Hughes Phenomenon:**
  - As dimensions increase, data becomes sparse
  - Accuracy drops
  - Modeling becomes harder
  - Training data needs grow exponentially
- **Distance Issues:**
  - Distances become less meaningful in high dimensions
  - Variance tends to 0 when dimensions → ∞
- **Need for Reduction:** Dimensionality reduction is essential

### 11.2 Why Reduce Dimensions?
- **Visualization:** Project onto 2D/3D to identify patterns, clusters, outliers
- **Computational Efficiency:**
  - Simplify data complexity
  - Faster training and inference (especially distance-based algorithms)
- **Interpretability:**
  - Lower-dimensional representations more interpretable
  - Easier to understand and communicate results
- **Data Compression:** Beneficial for storage and transmission
- **Clustering/Classification:** Easier pattern identification
- **Anomaly Detection:** Outliers more apparent
- **Feature Selection:** Retain important features, discard redundant ones
- **Noise Reduction:** Focus on underlying patterns

### 11.3 Principal Component Analysis (PCA)
- **Definition:**
  - Unsupervised method
  - Projects data into fewer dimensions
  - Original data (m features) → k principal components (k < m)
  - Synthetic features (linear combinations)
- **Intuition:**
  - Movies: 3D shot but watched in 2D (little loss)
  - Music: Drop dimension (music) when focusing
- **Process:**
  - Find directions of maximum variance
  - Project onto lower-dimensional space
  - Preserve most information

### 11.4 Dimensionality in Models
- **2D:** Curve
- **3D:** Surface
- **nD:** Hypersurface
- Partial differential equations = Features + 1 (bias term)

---

## 12. CLUSTERING (UNSUPERVISED LEARNING)

### 12.1 Core Concepts
- **Classes vs. Clusters:**
  - Classes: Defined manually, labeled (Supervised)
  - Clusters: Deduced automatically, no labels (Unsupervised)
- **Features as Vectors:**
  - Features expressed as vectors
  - Example: [Genre, Historical Accuracy, Heroism, Battles, Emotional Depth, Rating]

### 12.2 Distance & Similarity
- **Distance Metrics:**
  - May not correspond to physical distance
  - Measure of similarity
  - Smaller distance = better similarity
  - Types: Cosine, Euclidean, Manhattan, Geodesic
- **Cluster Objectives:**
  - Maximize inter-cluster distances
  - Minimize intra-cluster distances

### 12.3 Clustering Types
- **Partitioning:** Non-overlapping subsets
- **Hierarchical:** Tree-like structure
  - Agglomerative (bottom-up)
  - Divisive (top-down)
- **Density-Based:** Based on density regions (e.g., DBSCAN)
- **Fuzzy Clustering:** Points belong to multiple clusters with varying degrees

### 12.4 Algorithms
- **K-Means:** Most common partitioning algorithm
- **Agglomerative Clustering:** Hierarchical bottom-up
- **Divisive Clustering:** Hierarchical top-down
- **DBSCAN:** Density-based spatial clustering

### 12.5 Hyperparameters
- **Similarity Metric:** Choose distance measure
- **Type of Clustering:** Partitioning, hierarchical, density-based, fuzzy
- **Clustering Algorithm:** K-Means, Agglomerative, Divisive, DBSCAN
- **Number of Clusters (K):** Critical parameter

### 12.6 Applications
- Social media analysis
- Image clustering
- Streaming services (movie/audio grouping)
- Customer segmentation
- Pattern discovery

---

## 13. EVALUATION METRICS

### 13.1 Classification Metrics
- **Confusion Matrix:**
  - True Positives (TP)
  - True Negatives (TN)
  - False Positives (FP)
  - False Negatives (FN)
- **Accuracy:** (TP + TN) / Total
- **Precision:** TP / (TP + FP)
- **Recall (Sensitivity):** TP / (TP + FN)
- **F1-Score:** Harmonic mean of precision and recall
- **G-mean:** Geometric mean of precision and recall (balances both)

### 13.2 ROC Curve
- **True Positive Rate vs. False Positive Rate**
- **Area Under ROC (AUC):**
  - High area = better model performance
  - Measures classifier's ability to distinguish classes

### 13.3 Cross-Validation
- **K-Fold Cross-Validation:**
  - Improves reliability
  - Reduces bias
  - Better estimate of model performance

### 13.4 Real-World Importance
- **Why Accuracy Matters:**
  - Every decimal % pays
  - Example: 1% reduction in nurse hours = $2M savings/year
  - 0.1% reduction in patient stay = $10M savings/year

### 13.5 Model Performance Limits
- **No Free Lunch Theorem:**
  - No single algorithm works best for all problems
  - Model choice depends on data
- **Occam's Razor:**
  - Simplest explanation/model is usually best

---

## 14. ENSEMBLE METHODS

### 14.1 Motivation
- **Problem:** Different models have different weaknesses
  - Parametric models (e.g., Logistic Regression) → impacted by bias
  - Others (e.g., Decision Trees) → impacted by variance
- **Solution:** Diversify (like stock portfolio)
  - Combine multiple models
  - Meta-learning approach

### 14.2 Ensemble Learning
- **Base Learners:** Often weak learners
- **Generation:** Parallel or sequential
- **Rationale:**
  - Each classifier is weak learner (e.g., 45% error rate)
  - For ensemble to misclassify, majority must misclassify
  - Uses binomial distribution
  - Ensemble error much lower than individual

### 14.3 Methods
- **Bagging:**
  - Bootstrap aggregating
  - Train on different bootstrap samples
- **Boosting:**
  - Sequential training
  - Focus on misclassified examples
  - AdaBoost
- **Random Forest:**
  - Ensemble of decision trees
  - Random feature selection

### 14.4 Generating Diversity
- **Perturb X:**
  - Row-wise (different samples)
  - Column-wise (different features)
  - Randomly, no bias
- **Perturb Y:**
  - Different labels
- **Goal:** Don't generate same tree/model

### 14.5 Meta-Learning
- **Definition:** Learning to learn
- **Minimizes:** Losses from previous learning rounds
- **Importance:** Big deal for deep learning (few-shot learning)

---

## 15. DEEP LEARNING & NEURAL NETWORKS

### 15.1 Basic Concepts
- **Hierarchical Feature Representations:**
  - Learns automatically
  - Multi-layer networks
  - Complex abstractions from raw data
- **Adding Sigmoids:**
  - Combining multiple sigmoid functions
  - Can approximate nearly any curve/function
  - Foundation for neural networks

### 15.2 Comparison with Classical Models
- **Neural Networks:**
  - Overfit less than classical models
  - However, large models (e.g., GPT-2) can overfit
- **Deep Learning:**
  - Automates labeling more than traditional ML
  - Form of meta-learning

### 15.3 Softmax Classifier
- **Process:**
  1. Use exp to calculate probabilities
  2. Normalize (probabilities sum = 1)
- **Maximum Likelihood Estimation:**
  - Maximize likelihood of observed data
- **Loss Function:**
  - Cross-entropy / Kullback-Leibler divergence
  - Minimal loss: 0 (100% confident, exactly correct)
  - Maximum loss: ∞ (totally wrong)
- **Temperature:**
  - Higher temperature = more even probabilities
  - Higher creativity in models (e.g., ChatGPT)

---

## 16. PROBABILITY & STATISTICAL FOUNDATIONS

### 16.1 Maximum Likelihood Estimation
- **Probability Density Function (PDF):**
  - Follows bell curve (normal distribution)
  - Used for parameter estimation
- **Finding μ (Mean):**
  1. Find μ to maximize function (arg max)
  2. Replace with probability formula
  3. Take logarithm
  4. Use Gaussian formula
  5. Derive to maximum (partial differential equation)
  6. Solution: μ = mean
- **Parameter Estimation:**
  - Need to solve partial differential equations (mean, variance)
  - Infinite pairs of parameters to consider
  - PDE number = Features + 1 (bias term)

### 16.2 Normal Distribution
- **Parameters:**
  - Mean (μ): Decides position of shape
  - Standard Deviation (σ): Decides curve height
    - Smaller (< 1): Higher curve
    - Bigger (> 1): Lower curve
- **Bell Curve:**
  - Expressed by combination of two sides functions (2 curves)
  - Has 3 parameters: a, b, c

### 16.3 Linear Functions & Generative Modeling
- **2D Linear Function:**
  - y = mx + c
  - m, c can determine and generate new data
  - Represented as w₀, w₁
- **3D (Plane):**
  - Parameters: w₀, w₁, w₂
- **Generative Modeling:**
  - Generate data points based on latent parameters
  - If correct approximate function found, can generate new data

### 16.4 Parameter Estimation for Logistic Regression
- **Target Variable Y:**
  - Use PMF (Probability Mass Function)
  - Best model: Bernoulli
  - Use sigmoid function to find p
- **Maximum Likelihood:**
  - w means weight
  - Two ways to maximize:
    1. Gradient Descent
    2. Update w based on training data

---

## 17. APPLICATIONS & SUSTAINABILITY

### 17.1 Social Causes & ML
- **Machine Learning for Social Causes**
- **Why Social Innovation?**
  - Economy grows as more people join core echelons
  - People are most important economic resources

### 17.2 UN Sustainable Development Goals
- ML applications for sustainability
- Focus: Classification and regression at scale

### 17.3 Energy Applications
- **Smart Energy, Smart Grids, Vehicle-to-Grid:**
  - Solar irradiance forecasting
    - Stations across climate zones
    - Tree-based methods (CUB, ERT, Random Forest) superior
  - Wind energy forecasting
    - SVR, MLP, RFR, GBR, XGB
    - Ensemble LSTM for short-term forecasting

### 17.4 Healthcare Applications
- Hospital readmission prediction
- Reducing patient length of stay
- Nurse hours optimization
- Significant cost savings from small improvements

### 17.5 Sustainability in ML
- ML has real impact only when applied sustainably and ethically
- Reduces uncertainty in:
  - Energy demand forecasting
  - Electric vehicle optimization
  - Carbon footprint tracking
- Balance demand, conservation, and sustainability goals

---

## 18. ADVANCED TOPICS & CONCEPTS

### 18.1 Multi-Class Classification
- **Strategies:**
  - One-vs-All (OVA)
  - One-vs-One (OVO)
- **Prediction:** Class with highest probability

### 18.2 Margin Loss & Optimization
- ML models aim to minimize loss or maximize margins
- Margin represents confidence of classification
- Larger margin → better generalization
- Objective function can be complex ("black box"), often multivariate

### 18.3 Soft vs. Hard Computing
- **Soft Computing:**
  - Fuzzy, flexible
  - Real-world problems need approximation
  - ML embraces soft computing
- **Hard Computing:**
  - Rigid, binary

### 18.4 Compression & Sketching
- Once trained, massive data not needed
- Models store knowledge in compressed form (sketch)
- Enables efficient predictions on new inputs

### 18.5 Misinformation & Veracity
- **Problem:** Lies spread faster than truth on social networks
- **Logistic Regression for Misinformation:**
  - Detect truth vs. falsehood
- **Challenges:**
  - Can web contain only truth?
  - Syntax for truthful information?
  - Detect lies live on web?
  - Identifiable features for truth/lies?
  - Model expertise programmatically?
  - Acceptable classification accuracy?
- **Status:** Problem still unsolved (NP-hard)

### 18.6 Spaced Learning
- Based on Ebbinghaus forgetting curve
- Important for effective learning
- Applied in course structure

---

## 19. DOMAIN KNOWLEDGE & MODEL SELECTION

### 19.1 Domain Knowledge Importance
- Key for selecting appropriate model
- **Examples:**
  - Air quality: Linear model
  - Energy: Quadratic model
  - Decay: Exponential model
  - Wrong model → bad prediction

### 19.2 Hyperparameters
- Parameters affecting training, validation, and test accuracy
- Examples: k (in KNN), split ratio, learning rate
- Need tuning via validation set

### 19.3 Model Selection Principles
- **Occam's Razor:** Simplest model usually best
- **No Free Lunch Theorem:** No algorithm best for all problems
- Model choice depends on data characteristics

---

## 20. VECTORS & LINEAR ALGEBRA

### 20.1 Vector Concepts
- **What is a Vector?**
  - Set of numbers
  - Point in space with magnitude and direction
  - Discretized function (if function sampled at points)
- **Why Linear Algebra Important for ML?**
  - Features represented as vectors
  - Many operations use vector/matrix math
- **Semantic Significance:**
  - Dot product: Measures similarity/alignment
  - Matrix multiplication: Transforms data

### 20.2 Feature Vectors
- Each feature is a dimension
- 3 dimensions = plane
- n dimensions = hypersurface
- Vector representation: [x₁, x₂, ..., xₙ]

---

## 21. KEY FORMULAS & EQUATIONS

### 21.1 Probability & Statistics
- **Bayes' Theorem:** P(Y|X) = P(X|Y) × P(Y) / P(X)
- **Odds:** P / (1-P)
- **Log-odds:** log(P / (1-P))
- **Sigmoid:** s = 1 / (1 + e^(-z))
- **Normal Distribution:** Bell curve with μ and σ

### 21.2 Entropy & Information
- **Entropy:** H = -Σ(p × log(p))
- **Information Gain:** IG(Y|X) = H(Y) - H(Y|X)
- **Gini Index:** Gini(D) = 1 - Σ(pᵢ)²

### 21.3 Regression & Classification
- **Linear Regression:** Y = β₀ + β₁X₁ + ... + βₚXₚ + ε
- **Logistic Regression:** P(Y=1|X) = 1 / (1 + e^(-(w·x + b)))

### 21.4 Optimization
- **Gradient Descent:** wⱼ = wⱼ - η × (∂loss/∂wⱼ)
- **Softmax:** P(y=k|x) = exp(sₖ) / Σexp(sⱼ)

### 21.5 Distance Metrics
- **Euclidean:** √(Σ(uᵢ - vᵢ)²)
- **Manhattan:** Σ|uᵢ - vᵢ|
- **Minkowski:** (Σ|uᵢ - vᵢ|ᵖ)^(1/p)
- **Cosine Similarity:** (U·V) / (||U|| × ||V||)

---

## 22. LEARNING PARADIGMS & METHODOLOGIES

### 22.1 Eager vs. Lazy Learning
- **Eager Learning:**
  - Build model before test data arrives
  - Examples: Logistic Regression, Decision Trees, Neural Networks
- **Lazy Learning:**
  - Do nothing until test data arrives
  - Examples: K-Nearest Neighbors

### 22.2 Model-Based vs. Instance-Based
- **Model-Based:**
  - Learn model/parameters
  - Abstract representation
  - Examples: Linear Regression, SVM, Naive Bayes
- **Instance-Based:**
  - Store instances in memory
  - No abstraction
  - Examples: KNN

### 22.3 Parametric vs. Non-Parametric
- **Parametric:**
  - Fixed number of parameters
  - High bias, low variance
  - Example: Linear Regression
- **Non-Parametric:**
  - Number of parameters grows with data
  - Low bias, high variance
  - Example: Decision Trees, KNN

---

## 23. PRACTICAL INSIGHTS & TAKEAWAYS

### 23.1 Data Preprocessing
- Feature normalization essential (especially for distance-based methods)
- Handle missing data
- Address class imbalance
- Feature selection important

### 23.2 Model Development Process
1. Understand problem and data
2. Select appropriate model type
3. Split data (train/validation/test)
4. Train model
5. Tune hyperparameters
6. Evaluate on test set
7. Deploy and monitor

### 23.3 Common Pitfalls
- Data leakage (test data influencing training)
- Overfitting (memorizing training data)
- Underfitting (too simple model)
- Ignoring domain knowledge
- Wrong model for problem type

### 23.4 Best Practices
- Use cross-validation
- Regularization to prevent overfitting
- Early stopping
- Feature engineering based on domain knowledge
- Model interpretability when needed
- Consider sustainability and ethics

---

## 24. EMERGING TRENDS & FUTURE DIRECTIONS

### 24.1 Generative AI
- **Generative Adversarial Networks (GANs):**
  - Generator vs. Discriminator
  - "What I cannot create, I do not understand" - Feynman
- **Applications:**
  - Image generation
  - Text-to-image
  - Style transfer (CycleGAN)
  - Music conversion

### 24.2 Large Language Models
- GPT-4o and similar models
- High performance on academic benchmarks
- Limitations in genuine logical reasoning
- Attempt to replicate reasoning from training data

### 24.3 Spatial Web
- Next milestone: Mix of virtual and real worlds
- Integration of ML with spatial computing

### 24.4 Explainability
- Need to consider model explainability
- Beyond quantitative performance metrics
- Important for trust and adoption

---

**END OF MINDMAP**

*This comprehensive mindmap consolidates all major topics, concepts, algorithms, and insights from the DATA 245 Machine Learning course materials including handwritten class notes, slides, and flipped class content.*

