# Unsupervised Learning: Complete Guide from Scratch to Pro
**Topic 2.2: Unsupervised Learning**  
*Comprehensive guide*

---

## Table of Contents
1. [Foundations of Unsupervised Learning](#1-foundations-of-unsupervised-learning)
2. [Clustering Fundamentals](#2-clustering-fundamentals)
3. [K-Means Clustering](#3-k-means-clustering)
4. [Hierarchical Clustering](#4-hierarchical-clustering)
5. [DBSCAN: Density-Based Clustering](#5-dbscan-density-based-clustering)
6. [Principal Component Analysis (PCA)](#6-principal-component-analysis-pca)
7. [Dimensionality Reduction](#7-dimensionality-reduction)
8. [Evaluation Metrics for Unsupervised Learning](#8-evaluation-metrics-for-unsupervised-learning)
9. [Advanced Topics & Best Practices](#9-advanced-topics--best-practices)

---

## 1. Foundations of Unsupervised Learning

### 1.1 What is Unsupervised Learning?

**Definition:**
Unsupervised learning discovers hidden patterns in data without labeled examples. Unlike supervised learning, there are no "correct answers" to learn from—the algorithm must find structure on its own.

**Key Difference from Supervised Learning:**
- **Supervised**: Y = f(X) + ε (we have both X and Y)
- **Unsupervised**: Only X is available, no Y labels
- The algorithm must discover patterns, groups, or representations from X alone

### 1.2 Main Tasks in Unsupervised Learning

#### **1. Clustering: Group Similar Data Points**
- **Goal**: Partition data into groups (clusters) where points in same group are similar
- **Examples**: Customer segmentation, image organization, document clustering

#### **2. Dimensionality Reduction: Reduce Feature Space**
- **Goal**: Reduce number of features while preserving important information
- **Examples**: Visualization, feature compression, noise reduction

#### **3. Segmentation: Discover Patterns**
- **Goal**: Identify distinct segments or regions in data
- **Examples**: Market segmentation, anomaly detection, pattern discovery

---

## 2. Clustering Fundamentals

### 2.1 Distance and Similarity Metrics

#### **Euclidean Distance (L2 norm) - Full Derivation**

**Starting Point**: We want to measure straight-line distance between two points in p-dimensional space.

**Step 1: Define Points**
- Point x = (x₁, x₂, ..., xₚ) in p-dimensional space
- Point y = (y₁, y₂, ..., yₚ) in p-dimensional space

**Step 2: Apply Pythagorean Theorem in 2D**
- In 2D: d² = (x₁ - y₁)² + (x₂ - y₂)²
- **Why**: Pythagorean theorem: distance² = sum of squared differences in each dimension

**Step 3: Generalize to p Dimensions**
- d² = (x₁ - y₁)² + (x₂ - y₂)² + ... + (xₚ - yₚ)²
- **Why**: Extend Pythagorean theorem to higher dimensions by summing squared differences across all dimensions

**Step 4: Write in Summation Notation**
- d² = Σᵢ₌₁ᵖ (xᵢ - yᵢ)²
- **Why**: Compact mathematical notation for sum over all p dimensions

**Step 5: Take Square Root**
- d(x, y) = √[Σᵢ₌₁ᵖ (xᵢ - yᵢ)²]
- **Why**: Convert squared distance back to actual distance (undo squaring from Step 2)

**Final Formula**:
```
d(x, y) = √[Σᵢ₌₁ᵖ (xᵢ - yᵢ)²]
```

**Interpretation**: Straight-line distance in p-dimensional Euclidean space.

#### **Manhattan Distance (L1 norm) - Derivation**

**Starting Point**: Measure distance when movement is restricted to grid-like paths (like city blocks).

**Step 1: In 2D (City Blocks)**
- Distance = |x₁ - y₁| + |x₂ - y₂|
- **Why**: Can't move diagonally, must move along axes. Total distance = sum of horizontal and vertical distances.

**Step 2: Generalize to p Dimensions**
- d(x, y) = |x₁ - y₁| + |x₂ - y₂| + ... + |xₚ - yₚ|
- **Why**: Sum absolute differences in each dimension independently

**Step 3: Summation Notation**
- d(x, y) = Σᵢ₌₁ᵖ |xᵢ - yᵢ|
- **Why**: Compact notation for sum of absolute values

**Final Formula**:
```
d(x, y) = Σᵢ₌₁ᵖ |xᵢ - yᵢ|
```

**Interpretation**: Sum of absolute differences, more robust to outliers than Euclidean.

#### **Cosine Similarity - Full Derivation**

**Starting Point**: Measure angle between two vectors (direction similarity, not magnitude).

**Step 1: Dot Product Definition**
- x·y = x₁y₁ + x₂y₂ + ... + xₚyₚ = Σᵢ₌₁ᵖ xᵢyᵢ
- **Why**: Dot product measures alignment between vectors

**Step 2: Magnitude (Length) of Vectors**
- ||x|| = √(x₁² + x₂² + ... + xₚ²) = √[Σᵢ₌₁ᵖ xᵢ²]
- ||y|| = √(y₁² + y₂² + ... + yₚ²) = √[Σᵢ₌₁ᵖ yᵢ²]
- **Why**: Euclidean norm (length) of vector from origin

**Step 3: Cosine of Angle from Dot Product Formula**
- From linear algebra: x·y = ||x|| × ||y|| × cos(θ)
- **Why**: Fundamental relationship between dot product and angle

**Step 4: Solve for Cosine**
- cos(θ) = (x·y) / (||x|| × ||y||)
- **Why**: Rearrange Step 3 to isolate cosine

**Step 5: Substitute Definitions**
- cos(θ) = [Σᵢ₌₁ᵖ xᵢyᵢ] / [√(Σᵢ₌₁ᵖ xᵢ²) × √(Σᵢ₌₁ᵖ yᵢ²)]
- **Why**: Replace dot product and norms with their definitions from Steps 1-2

**Final Formula**:
```
similarity(x, y) = (x·y) / (||x|| × ||y||) = [Σᵢ₌₁ᵖ xᵢyᵢ] / [√(Σᵢ₌₁ᵖ xᵢ²) × √(Σᵢ₌₁ᵖ yᵢ²)]
```

**Range**: [-1, 1] where 1 = identical direction, 0 = perpendicular, -1 = opposite direction.

---

## 3. K-Means Clustering

### 3.1 Mathematical Foundation - Complete Derivation

#### **Objective Function Derivation**

**Starting Point**: We want to minimize the sum of squared distances from points to their cluster centers.

**Step 1: Define the Problem**
- Given: n data points x₁, x₂, ..., xₙ
- Goal: Partition into K clusters C₁, C₂, ..., Cₖ
- Each cluster has centroid μᵢ (mean of points in cluster i)

**Step 2: Distance from Point to Centroid**
- For point x in cluster Cᵢ, distance to centroid: ||x - μᵢ||
- **Why**: Euclidean distance measures how far point is from cluster center

**Step 3: Squared Distance (Why Square?)**
- Squared distance: ||x - μᵢ||² = (x - μᵢ)ᵀ(x - μᵢ) = Σⱼ(xⱼ - μᵢⱼ)²
- **Why Square?**:
  - Penalizes large errors more (convex function)
  - Mathematically convenient (differentiable everywhere)
  - Equivalent to maximum likelihood under Gaussian assumption

**Step 4: Sum Over All Points in Cluster**
- For cluster Cᵢ: Σₓ∈Cᵢ ||x - μᵢ||²
- **Why**: Measure total "spread" or "compactness" of cluster i

**Step 5: Sum Over All Clusters**
- Total within-cluster sum of squares: Σᵢ₌₁ᵏ Σₓ∈Cᵢ ||x - μᵢ||²
- **Why**: Measure total compactness across all clusters

**Final Objective Function**:
```
WCSS = Σᵢ₌₁ᵏ Σₓ∈Cᵢ ||x - μᵢ||²
```

**Goal**: Minimize WCSS (also called inertia or distortion).

#### **Centroid Calculation - Derivation**

**Starting Point**: Centroid should be the point that minimizes sum of squared distances to all points in cluster.

**Step 1: Define Centroid as Minimizer**
- μᵢ = argmin_μ Σₓ∈Cᵢ ||x - μ||²
- **Why**: Centroid is the point that minimizes total squared distance to all cluster points

**Step 2: Expand Squared Distance**
- ||x - μ||² = (x - μ)ᵀ(x - μ) = xᵀx - 2xᵀμ + μᵀμ
- **Why**: Expand using (a-b)² = a² - 2ab + b²

**Step 3: Sum Over All Points**
- Σₓ∈Cᵢ ||x - μ||² = Σₓ∈Cᵢ (xᵀx - 2xᵀμ + μᵀμ)
- **Why**: Apply expansion to all points in cluster

**Step 4: Distribute Summation**
- = Σₓ∈Cᵢ xᵀx - 2(Σₓ∈Cᵢ x)ᵀμ + |Cᵢ|μᵀμ
- **Why**: 
  - First term: sum of xᵀx (constant with respect to μ)
  - Second term: factor out μ (linear in μ)
  - Third term: |Cᵢ| copies of μᵀμ (quadratic in μ)

**Step 5: Take Derivative with Respect to μ**
- ∂/∂μ [Σₓ∈Cᵢ ||x - μ||²] = -2Σₓ∈Cᵢ x + 2|Cᵢ|μ
- **Why**: 
  - Derivative of -2xᵀμ is -2x (treating μ as variable)
  - Derivative of |Cᵢ|μᵀμ is 2|Cᵢ|μ

**Step 6: Set Derivative to Zero (Minimization)**
- -2Σₓ∈Cᵢ x + 2|Cᵢ|μ = 0
- **Why**: At minimum, derivative equals zero

**Step 7: Solve for μ**
- 2|Cᵢ|μ = 2Σₓ∈Cᵢ x
- **Why**: Rearrange to isolate μ

**Step 8: Divide Both Sides**
- μᵢ = (1/|Cᵢ|) × Σₓ∈Cᵢ x
- **Why**: Divide by 2|Cᵢ| to get final formula

**Final Formula**:
```
μᵢ = (1/|Cᵢ|) × Σₓ∈Cᵢ x
```

**Interpretation**: Centroid is the arithmetic mean (average) of all points in the cluster.

**Component-wise** (for each dimension j):
```
μᵢⱼ = (1/|Cᵢ|) × Σₓ∈Cᵢ xⱼ
```

**Why This Works**: The mean minimizes sum of squared distances (this is a fundamental property of the mean).

### 3.2 The K-Means Algorithm

**Initialization**:
1. Choose K (number of clusters)
2. Initialize K centroids randomly

**Iteration** (repeat until convergence):

**Step 1: Assignment**
- For each data point x:
  - Calculate distance to all K centroids: d(x, μ₁), d(x, μ₂), ..., d(x, μₖ)
  - Assign x to nearest centroid: c(x) = argminᵢ ||x - μᵢ||²
  - **Why**: Minimize distance from point to cluster center

**Step 2: Update**
- For each cluster Cᵢ:
  - Recalculate centroid: μᵢ = (1/|Cᵢ|) × Σₓ∈Cᵢ x
  - **Why**: Update centroid to minimize WCSS for current assignments

**Convergence**: Stop when centroids don't change (or change < threshold).

---

## 4. Hierarchical Clustering

### 4.1 Linkage Criteria - Mathematical Derivation

#### **Single Linkage (Minimum Distance) - Derivation**

**Starting Point**: Distance between clusters = distance between closest pair of points.

**Step 1: Define Cluster Distance**
- d(Cᵢ, Cⱼ) = distance between clusters i and j
- **Why**: Need measure of how far apart two clusters are

**Step 2: Consider All Point Pairs**
- For clusters Cᵢ and Cⱼ, consider all pairs (x, y) where x ∈ Cᵢ and y ∈ Cⱼ
- **Why**: Must compare points from both clusters

**Step 3: Take Minimum Distance**
- d(Cᵢ, Cⱼ) = min{d(x, y) : x ∈ Cᵢ, y ∈ Cⱼ}
- **Why**: Single linkage uses closest pair (most optimistic measure)

**Final Formula**:
```
d_single(Cᵢ, Cⱼ) = min{d(x, y) : x ∈ Cᵢ, y ∈ Cⱼ}
```

**Interpretation**: Distance = minimum distance between any two points in different clusters.

#### **Complete Linkage (Maximum Distance) - Derivation**

**Starting Point**: Distance between clusters = distance between farthest pair of points.

**Step 1: Same as Single Linkage Steps 1-2**
- Define cluster distance and consider all point pairs

**Step 2: Take Maximum Distance**
- d(Cᵢ, Cⱼ) = max{d(x, y) : x ∈ Cᵢ, y ∈ Cⱼ}
- **Why**: Complete linkage uses farthest pair (most conservative measure)

**Final Formula**:
```
d_complete(Cᵢ, Cⱼ) = max{d(x, y) : x ∈ Cᵢ, y ∈ Cⱼ}
```

**Interpretation**: Distance = maximum distance between any two points in different clusters.

#### **Average Linkage (Mean Distance) - Derivation**

**Starting Point**: Distance between clusters = average distance between all point pairs.

**Step 1: Consider All Point Pairs**
- For clusters Cᵢ and Cⱼ, all pairs (x, y) where x ∈ Cᵢ, y ∈ Cⱼ
- Total number of pairs: |Cᵢ| × |Cⱼ|

**Step 2: Sum All Pairwise Distances**
- Total distance = Σₓ∈Cᵢ Σᵧ∈Cⱼ d(x, y)
- **Why**: Sum distances for all possible pairs between clusters

**Step 3: Average the Distances**
- d(Cᵢ, Cⱼ) = (1/(|Cᵢ| × |Cⱼ|)) × Σₓ∈Cᵢ Σᵧ∈Cⱼ d(x, y)
- **Why**: Divide total by number of pairs to get average

**Final Formula**:
```
d_average(Cᵢ, Cⱼ) = (1/(|Cᵢ| × |Cⱼ|)) × Σₓ∈Cᵢ Σᵧ∈Cⱼ d(x, y)
```

**Interpretation**: Distance = average of all pairwise distances between clusters.

#### **Ward's Linkage - Derivation**

**Starting Point**: Minimize increase in within-cluster variance when merging clusters.

**Step 1: Define Within-Cluster Variance Before Merge**
- For cluster Cᵢ: Vᵢ = (1/|Cᵢ|) × Σₓ∈Cᵢ ||x - μᵢ||²
- **Why**: Measure of cluster compactness (variance)

**Step 2: Define Combined Cluster After Merge**
- Merge Cᵢ and Cⱼ → C_new = Cᵢ ∪ Cⱼ
- New centroid: μ_new = (1/(|Cᵢ| + |Cⱼ|)) × (Σₓ∈Cᵢ x + Σₓ∈Cⱼ x)
- **Why**: Centroid of merged cluster is weighted average

**Step 3: Calculate Variance of Merged Cluster**
- V_new = (1/|C_new|) × Σₓ∈C_new ||x - μ_new||²
- **Why**: Variance of combined cluster

**Step 4: Calculate Increase in Variance**
- ΔV = V_new - (Vᵢ + Vⱼ)
- **Why**: Measure how much variance increases by merging

**Step 5: Ward's Distance (Proportional to ΔV)**
- d_ward(Cᵢ, Cⱼ) = ΔV × (|Cᵢ| × |Cⱼ|) / (|Cᵢ| + |Cⱼ|)
- **Why**: Weighted increase in variance (more complex formula accounts for cluster sizes)

**Final Formula** (simplified form):
```
d_ward(Cᵢ, Cⱼ) = ||μᵢ - μⱼ||² × (|Cᵢ| × |Cⱼ|) / (|Cᵢ| + |Cⱼ|)
```

**Interpretation**: Weighted squared distance between centroids, where weight depends on cluster sizes.

---

## 5. DBSCAN: Density-Based Clustering

### 5.1 Core Point Definition - Mathematical Derivation

**Starting Point**: A point is "core" if it has enough neighbors within distance ε.

**Step 1: Define Neighborhood**
- For point p, ε-neighborhood: N_ε(p) = {q : d(p, q) ≤ ε}
- **Why**: Set of all points within distance ε of p

**Step 2: Count Neighbors**
- |N_ε(p)| = number of points in ε-neighborhood (including p itself)
- **Why**: Measure local density around point p

**Step 3: Core Point Condition**
- Point p is core point if: |N_ε(p)| ≥ MinPts
- **Why**: Core point has at least MinPts neighbors (dense region)

**Final Definition**:
```
p is core point ⟺ |N_ε(p)| ≥ MinPts
where N_ε(p) = {q : d(p, q) ≤ ε}
```

**Interpretation**: Core points form the "backbone" of dense clusters.

### 5.2 Density-Reachability - Mathematical Derivation

#### **Directly Density-Reachable - Derivation**

**Starting Point**: Point q is directly reachable from p if q is close to p and p is a core point.

**Step 1: Condition 1 - q is in Neighborhood of p**
- q ∈ N_ε(p) ⟺ d(p, q) ≤ ε
- **Why**: q must be within distance ε of p

**Step 2: Condition 2 - p is Core Point**
- |N_ε(p)| ≥ MinPts
- **Why**: p must have enough neighbors to be core point

**Step 3: Combine Conditions**
- q is directly density-reachable from p if:
  - q ∈ N_ε(p) AND
  - |N_ε(p)| ≥ MinPts

**Final Definition**:
```
q is directly density-reachable from p ⟺ 
  (q ∈ N_ε(p)) ∧ (|N_ε(p)| ≥ MinPts)
```

**Interpretation**: q is directly reachable from p if q is in p's neighborhood and p is dense enough.

#### **Density-Reachable - Derivation**

**Starting Point**: q is reachable from p if there's a chain of directly reachable points.

**Step 1: Define Chain**
- Chain: p₁, p₂, ..., pₙ where p₁ = p and pₙ = q
- **Why**: Sequence of points connecting p to q

**Step 2: Each Step is Directly Reachable**
- For each i from 1 to n-1: pᵢ₊₁ is directly density-reachable from pᵢ
- **Why**: Each step in chain must satisfy direct reachability

**Step 3: Transitive Closure**
- q is density-reachable from p if there exists such a chain
- **Why**: Reachability is transitive (if a→b and b→c, then a→c)

**Final Definition**:
```
q is density-reachable from p ⟺ 
  ∃ chain p₁, p₂, ..., pₙ such that:
    p₁ = p, pₙ = q, and
    ∀i ∈ {1, ..., n-1}: pᵢ₊₁ is directly density-reachable from pᵢ
```

**Interpretation**: q is reachable from p if you can "walk" from p to q through a chain of core points.

#### **Density-Connected - Derivation**

**Starting Point**: Two points are connected if both are reachable from a common core point.

**Step 1: Common Core Point**
- There exists core point r such that:
  - p is density-reachable from r, AND
  - q is density-reachable from r
- **Why**: Both points can be reached from same core point

**Final Definition**:
```
p and q are density-connected ⟺ 
  ∃ core point r such that:
    (p is density-reachable from r) ∧ (q is density-reachable from r)
```

**Interpretation**: Points in same cluster are all density-connected (all reachable from cluster's core points).

---

## 6. Principal Component Analysis (PCA)

### 6.1 Mathematical Foundation - Complete Derivation

#### **Step 1: Data Centering - Derivation**

**Starting Point**: We have data matrix X (n × p) with n samples and p features.

**Step 1.1: Calculate Mean**
- For each feature j: x̄ⱼ = (1/n) × Σᵢ₌₁ⁿ xᵢⱼ
- **Why**: Mean of each feature across all samples

**Step 1.2: Center the Data**
- X_centered = X - x̄ (subtract mean from each column)
- Component-wise: x̃ᵢⱼ = xᵢⱼ - x̄ⱼ
- **Why**: 
  - Centers data at origin (mean becomes zero)
  - Necessary for covariance calculation
  - Removes bias from coordinate system

**Result**: Centered data matrix X̃ where each column has mean zero.

#### **Step 2: Covariance Matrix - Full Derivation**

**Starting Point**: We want to measure how features vary together.

**Step 2.1: Covariance Definition (Two Variables)**
- For features X and Y: Cov(X, Y) = E[(X - E[X])(Y - E[Y])]
- **Why**: Measures how two variables vary together (positive = increase together, negative = one increases when other decreases)

**Step 2.2: Sample Covariance Formula**
- Cov(X, Y) = (1/n) × Σᵢ₌₁ⁿ (xᵢ - x̄)(yᵢ - ȳ)
- **Why**: Estimate population covariance from sample (average of products of deviations from means)

**Step 2.3: Generalize to Matrix Form**
- For centered data X̃ (n × p), covariance between features j and k:
  - Cⱼₖ = (1/n) × Σᵢ₌₁ⁿ x̃ᵢⱼ × x̃ᵢₖ
- **Why**: Apply covariance formula to each pair of features

**Step 2.4: Matrix Multiplication Form**
- C = (1/n) × X̃ᵀ × X̃
- **Why**: 
  - X̃ᵀ is p × n (transpose)
  - X̃ᵀ × X̃ gives p × p matrix
  - Element (j, k) = Σᵢ x̃ᵢⱼ × x̃ᵢₖ (dot product of columns j and k)
  - Dividing by n gives average (covariance)

**Step 2.5: Verify Diagonal Elements**
- Cⱼⱼ = (1/n) × Σᵢ x̃ᵢⱼ² = Var(Xⱼ)
- **Why**: Diagonal elements are variances (covariance of feature with itself)

**Final Covariance Matrix**:
```
C = (1/n) × X̃ᵀ × X̃
```
where X̃ is centered data (mean of each column = 0).

**Properties**:
- C is p × p symmetric matrix
- Diagonal: Variances of each feature
- Off-diagonal: Covariances between features

#### **Step 3: Eigenvalue Decomposition - Derivation**

**Starting Point**: We want to find directions (eigenvectors) where data has maximum variance.

**Step 3.1: Eigenvalue Equation**
- For covariance matrix C, find vectors v and scalars λ such that:
  - Cv = λv
- **Why**: Eigenvectors are directions where matrix acts like scaling (direction doesn't change)

**Step 3.2: Rearrange Eigenvalue Equation**
- Cv - λv = 0
- (C - λI)v = 0
- **Why**: Factor out v, where I is identity matrix

**Step 3.3: Non-Trivial Solution Condition**
- For non-zero v, we need: det(C - λI) = 0
- **Why**: System has non-trivial solution only if matrix (C - λI) is singular (determinant = 0)

**Step 3.4: Characteristic Polynomial**
- det(C - λI) = 0 is polynomial equation in λ
- **Why**: Solving this gives eigenvalues λ₁, λ₂, ..., λₚ

**Step 3.5: Find Eigenvectors**
- For each eigenvalue λᵢ, solve: (C - λᵢI)vᵢ = 0
- **Why**: Each eigenvalue has corresponding eigenvector (direction)

**Step 3.6: Normalize Eigenvectors**
- ||vᵢ|| = 1 (unit length)
- **Why**: Standardize direction (magnitude doesn't matter, only direction)

**Result**: 
- Eigenvalues: λ₁ ≥ λ₂ ≥ ... ≥ λₚ (sorted in descending order)
- Eigenvectors: v₁, v₂, ..., vₚ (corresponding principal components)

#### **Step 4: Variance Explained - Derivation**

**Starting Point**: We want to know how much variance each principal component captures.

**Step 4.1: Variance Along Eigenvector**
- Variance along direction vᵢ: Var(vᵢ) = vᵢᵀCvᵢ
- **Why**: Variance in direction vᵢ is quadratic form vᵢᵀCvᵢ

**Step 4.2: Substitute Eigenvalue Equation**
- vᵢᵀCvᵢ = vᵢᵀ(λᵢvᵢ) = λᵢ(vᵢᵀvᵢ)
- **Why**: Use Cvᵢ = λᵢvᵢ from eigenvalue equation

**Step 4.3: Use Normalization**
- vᵢᵀvᵢ = ||vᵢ||² = 1 (since vᵢ is normalized)
- **Why**: Eigenvector has unit length

**Step 4.4: Final Result**
- Var(vᵢ) = λᵢ
- **Why**: Variance along eigenvector equals its eigenvalue!

**Interpretation**: 
- Larger eigenvalue = more variance in that direction
- Principal components are ordered by variance (largest first)

**Step 4.5: Total Variance**
- Total variance = Σⱼ₌₁ᵖ λⱼ = trace(C) = Σⱼ₌₁ᵖ Var(Xⱼ)
- **Why**: Sum of eigenvalues equals trace of covariance matrix (sum of diagonal = sum of variances)

**Step 4.6: Proportion of Variance Explained**
- Proportion by component i: λᵢ / Σⱼ λⱼ
- **Why**: Fraction of total variance captured by component i

**Step 4.7: Cumulative Variance**
- Cumulative by first k components: (Σᵢ₌₁ᵏ λᵢ) / (Σⱼ₌₁ᵖ λⱼ)
- **Why**: Total variance explained by keeping first k components

#### **Step 5: Data Projection - Derivation**

**Starting Point**: Project data onto principal components to reduce dimensions.

**Step 5.1: Projection onto Single Component**
- For data point x (p-dimensional), projection onto vᵢ:
  - zᵢ = vᵢᵀx = Σⱼ₌₁ᵖ vᵢⱼ × xⱼ
- **Why**: Dot product gives coordinate in direction vᵢ (how much x points in direction vᵢ)

**Step 5.2: Matrix of Principal Components**
- V_k = [v₁, v₂, ..., vₖ] (p × k matrix, first k eigenvectors as columns)
- **Why**: Stack k principal components as columns

**Step 5.3: Project All Data Points**
- Z = X̃ × V_k (n × k matrix)
- **Why**: 
  - Each row of X̃ is a data point
  - Each column of V_k is a principal component
  - Matrix multiplication: row i of Z = projection of data point i onto k components

**Step 5.4: Component-wise Formula**
- For data point i, component j: zᵢⱼ = vⱼᵀx̃ᵢ = Σₗ₌₁ᵖ vⱼₗ × x̃ᵢₗ
- **Why**: j-th coordinate in reduced space = dot product with j-th principal component

**Final Projection Formula**:
```
Z = X̃ × V_k
```
where:
- X̃: centered data (n × p)
- V_k: first k principal components (p × k)
- Z: projected data (n × k)

**Interpretation**: Data in k-dimensional space (reduced from p dimensions).

#### **Step 6: Reconstruction - Derivation**

**Starting Point**: Reconstruct original data from reduced representation.

**Step 6.1: Reverse Projection**
- X̃_reconstructed = Z × V_kᵀ
- **Why**: 
  - Z is n × k (projected data)
  - V_kᵀ is k × p (transpose of principal components)
  - Multiplication gives n × p (back to original dimension)

**Step 6.2: Component-wise**
- x̃_reconstructed = Σⱼ₌₁ᵏ zⱼ × vⱼ
- **Why**: Reconstruct by summing contributions from each principal component

**Step 6.3: Reconstruction Error**
- Error = ||X̃ - X̃_reconstructed||²
- **Why**: Measure how much information was lost

**Step 6.4: Optimality Property**
- PCA minimizes reconstruction error for given k
- **Why**: Principal components are optimal linear basis for reconstruction (this is a key theorem)

**Final Reconstruction**:
```
X̃_reconstructed = Z × V_kᵀ = (X̃ × V_k) × V_kᵀ
```

**Note**: Perfect reconstruction only if k = p (all components kept).

---

## 7. Dimensionality Reduction

### 7.1 Curse of Dimensionality - Mathematical Analysis

#### **Volume Growth - Derivation**

**Starting Point**: Volume in high dimensions grows exponentially.

**Step 1: Volume in d Dimensions**
- Hypercube with side length r: Volume = rᵈ
- **Why**: In 1D: length = r, in 2D: area = r², in 3D: volume = r³, generalize to rᵈ

**Step 2: Volume Ratio**
- Ratio of volume in shell to total: [(r)ᵈ - (r-ε)ᵈ] / rᵈ
- **Why**: Most volume is near surface in high dimensions

**Step 3: Limit as d → ∞**
- As d increases, almost all volume is in thin shell near surface
- **Why**: (r-ε)ᵈ / rᵈ → 0 as d → ∞ (exponential decay)

**Interpretation**: In high dimensions, data is concentrated near boundaries, making interior sparse.

#### **Distance Concentration - Derivation**

**Starting Point**: In high dimensions, distances become similar.

**Step 1: Expected Squared Distance**
- For random points in unit hypercube: E[d²] = d/6
- **Why**: Sum of variances of d independent uniform [0,1] variables

**Step 2: Variance of Distance**
- Var(d²) → 0 as d → ∞
- **Why**: Law of large numbers - variance of sum decreases

**Step 3: Concentration Result**
- As d → ∞, all distances concentrate around √(d/6)
- **Why**: Variance → 0 means all distances become similar

**Interpretation**: In high dimensions, nearest and farthest neighbors become almost equidistant!

---

## 8. Evaluation Metrics for Unsupervised Learning

### 8.1 Silhouette Score - Complete Derivation

**Starting Point**: Measure how well a point fits its cluster vs. other clusters.

**Step 1: Define Intra-Cluster Distance**
- For point i in cluster C:
  - a(i) = (1/|C| - 1) × Σⱼ∈C, j≠i d(i, j)
- **Why**: Average distance to other points in same cluster (exclude point i itself)

**Step 2: Define Inter-Cluster Distance**
- For point i in cluster C, distance to cluster C':
  - b(i, C') = (1/|C'|) × Σⱼ∈C' d(i, j)
- **Why**: Average distance to points in other cluster

**Step 3: Find Nearest Other Cluster**
- b(i) = min{b(i, C') : C' ≠ C(i)}
- **Why**: Distance to nearest cluster (most similar alternative)

**Step 4: Define Silhouette for Point i**
- s(i) = [b(i) - a(i)] / max{a(i), b(i)}
- **Why**: 
  - Numerator: b(i) - a(i) (how much farther is nearest other cluster vs. own cluster)
  - Positive = point is closer to own cluster (good)
  - Negative = point is closer to other cluster (bad)
  - Denominator: Normalize to [-1, 1] range

**Step 5: Average Silhouette Score**
- Silhouette = (1/n) × Σᵢ₌₁ⁿ s(i)
- **Why**: Average over all points

**Final Formula**:
```
s(i) = [b(i) - a(i)] / max{a(i), b(i)}
where:
  a(i) = average distance to points in same cluster
  b(i) = average distance to points in nearest other cluster
```

**Range**: [-1, 1]
- **+1**: Perfectly separated (point much closer to own cluster)
- **0**: On boundary between clusters
- **-1**: Assigned to wrong cluster (closer to other cluster)

### 8.2 Within-Cluster Sum of Squares (Inertia) - Derivation

**Starting Point**: Measure compactness of clusters.

**Step 1: Distance from Point to Centroid**
- For point x in cluster Cᵢ with centroid μᵢ:
  - Distance² = ||x - μᵢ||²
- **Why**: Squared Euclidean distance to cluster center

**Step 2: Sum Over Points in Cluster**
- For cluster Cᵢ: Σₓ∈Cᵢ ||x - μᵢ||²
- **Why**: Total squared distance within cluster i

**Step 3: Sum Over All Clusters**
- Inertia = Σᵢ₌₁ᵏ Σₓ∈Cᵢ ||x - μᵢ||²
- **Why**: Total within-cluster sum of squares across all clusters

**Final Formula**:
```
Inertia = Σᵢ₌₁ᵏ Σₓ∈Cᵢ ||x - μᵢ||²
```

**Interpretation**: Lower inertia = tighter clusters (points closer to centroids).

### 8.3 Davies-Bouldin Index - Derivation

**Starting Point**: Measure cluster quality using ratio of within-cluster spread to between-cluster separation.

**Step 1: Average Distance Within Cluster**
- For cluster Cᵢ: σᵢ = (1/|Cᵢ|) × Σₓ∈Cᵢ ||x - μᵢ||
- **Why**: Average distance from points to centroid (measure of cluster spread)

**Step 2: Distance Between Centroids**
- d(μᵢ, μⱼ) = ||μᵢ - μⱼ||
- **Why**: Distance between cluster centers (measure of separation)

**Step 3: Similarity Ratio**
- Rᵢⱼ = (σᵢ + σⱼ) / d(μᵢ, μⱼ)
- **Why**: 
  - Numerator: Sum of spreads (larger = worse)
  - Denominator: Separation (larger = better)
  - Ratio: Lower = better (tight clusters, well separated)

**Step 4: Maximum Similarity for Cluster i**
- Rᵢ = maxⱼ≠ᵢ Rᵢⱼ
- **Why**: Worst-case similarity (most similar other cluster)

**Step 5: Average Over All Clusters**
- DB = (1/k) × Σᵢ₌₁ᵏ Rᵢ
- **Why**: Average worst-case similarity across all clusters

**Final Formula**:
```
DB = (1/k) × Σᵢ₌₁ᵏ maxⱼ≠ᵢ [(σᵢ + σⱼ) / d(μᵢ, μⱼ)]
where:
  σᵢ = (1/|Cᵢ|) × Σₓ∈Cᵢ ||x - μᵢ||
  d(μᵢ, μⱼ) = ||μᵢ - μⱼ||
```

**Interpretation**: Lower DB = better clustering (tight clusters, well separated).

### 8.4 Explained Variance in PCA - Derivation

**Starting Point**: Measure how much variance is preserved by principal components.

**Step 1: Variance of Original Data**
- Total variance = trace(C) = Σⱼ₌₁ᵖ λⱼ
- **Why**: Sum of eigenvalues = trace of covariance matrix = sum of feature variances

**Step 2: Variance Preserved by Component i**
- Variance explained by PC_i = λᵢ
- **Why**: From PCA derivation, variance along eigenvector vᵢ equals eigenvalue λᵢ

**Step 3: Proportion of Variance**
- Proportion by PC_i = λᵢ / Σⱼ₌₁ᵖ λⱼ
- **Why**: Fraction of total variance captured by component i

**Step 4: Cumulative Variance**
- Cumulative by first k components = (Σᵢ₌₁ᵏ λᵢ) / (Σⱼ₌₁ᵖ λⱼ)
- **Why**: Total variance explained by keeping top k components

**Final Formula**:
```
Explained Variance Ratio (PC_i) = λᵢ / Σⱼ₌₁ᵖ λⱼ
Cumulative Explained Variance (first k) = (Σᵢ₌₁ᵏ λᵢ) / (Σⱼ₌₁ᵖ λⱼ)
```

**Interpretation**: Higher values = more information preserved.

---

## 9. Advanced Topics & Best Practices

### 9.1 Feature Normalization - Why It's Critical

**Problem**: Features on different scales dominate distance calculations.

**Mathematical Example**:
- Feature 1: Age (0-100), Feature 2: Income (0-1,000,000)
- Point A: (30, 50,000), Point B: (31, 50,000)
- Euclidean distance: √[(30-31)² + (50,000-50,000)²] = 1
- Point C: (30, 50,000), Point D: (30, 100,000)
- Euclidean distance: √[(30-30)² + (50,000-100,000)²] = 50,000

**Problem**: Income dominates! Age differences are ignored.

**Solution - Min-Max Scaling**:
- x_scaled = (x - min) / (max - min)
- **Why**: Maps all features to [0, 1] range, equal influence

**Solution - Z-score Normalization**:
- x_scaled = (x - μ) / σ
- **Why**: Centers at 0, scales by standard deviation, equal variance

### 9.2 Choosing the Right Algorithm

**Decision Tree**:
- Know K? → K-Means
- Unknown K? → Hierarchical or DBSCAN
- Spherical clusters? → K-Means
- Arbitrary shapes? → DBSCAN
- Need outliers? → DBSCAN
- Large dataset? → K-Means (fast)

### 9.3 Best Practices

1. **Always normalize features** before clustering
2. **Visualize results** (PCA to 2D/3D)
3. **Use multiple metrics** for evaluation
4. **Try different algorithms** and compare
5. **Validate with domain knowledge**
6. **Test stability** (run multiple times)

---

## Summary: Key Takeaways

### Mathematical Foundations

1. **Distance Metrics**: Euclidean (L2), Manhattan (L1), Cosine similarity
2. **K-Means**: Minimize WCSS, centroid = mean of cluster points
3. **Hierarchical**: Linkage criteria (single, complete, average, Ward's)
4. **DBSCAN**: Density-reachability through core points
5. **PCA**: Eigenvalue decomposition of covariance matrix, variance = eigenvalues

### Key Formulas

- **Euclidean Distance**: d = √[Σ(xᵢ - yᵢ)²]
- **K-Means Centroid**: μ = (1/n) × Σx
- **Covariance Matrix**: C = (1/n) × X̃ᵀX̃
- **PCA Projection**: Z = X̃ × V_k
- **Silhouette Score**: s = (b - a) / max(a, b)

---

**End of Unsupervised Learning Guide**

*This guide includes detailed step-by-step mathematical derivations with explanations for all key formulas and algorithms.*

