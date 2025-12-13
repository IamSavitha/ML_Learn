# Advanced Learning Paradigms: Complete Guide
**Topics 2.3, 2.4, and 2.5: Semi-Supervised, Self-Supervised, and Other Learning Paradigms**  
*Comprehensive guide covering all topics from flipped class, class notes, and slides*

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

**Real-World Analogy**:
- Learning a language: You have a few labeled examples (dictionary with translations) and many unlabeled examples (conversations, books)
- You can learn patterns from unlabeled data (grammar, word usage) and use labeled examples to anchor meanings
- Together, you learn better than with labels alone

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

**Key Assumption**: 
- Data points that are close in feature space should have similar labels
- Unlabeled data helps identify the structure of the feature space

#### **Why It Works**

**Smoothness Assumption**:
- Points close to each other are likely to have the same label
- Unlabeled data helps define "closeness" in the feature space

**Cluster Assumption**:
- Data tends to form clusters
- Points in the same cluster should have the same label
- Unlabeled data reveals cluster structure

**Manifold Assumption**:
- High-dimensional data lies on a lower-dimensional manifold
- Unlabeled data helps learn this manifold structure

### 1.3 How Semi-Supervised Learning Works

#### **Step-by-Step Process**

1. **Initial Training**:
   - Train model on small labeled dataset L
   - Get initial predictions

2. **Pseudo-Labeling**:
   - Use trained model to predict labels for unlabeled data U
   - Select high-confidence predictions
   - Treat these as "pseudo-labels"

3. **Iterative Refinement**:
   - Retrain model on L + pseudo-labeled subset of U
   - Update predictions on remaining unlabeled data
   - Repeat until convergence

4. **Final Prediction**:
   - Use refined model for new predictions

### 1.4 Key Methods

#### **1. Self-Training**

**Process**:
1. Train classifier on labeled data L
2. Predict labels for unlabeled data U
3. Select high-confidence predictions
4. Add pseudo-labeled examples to training set
5. Retrain and repeat

**Characteristics**:
- Simple and intuitive
- Works with any base classifier
- Risk: Errors can propagate (if model makes mistakes, it reinforces them)

**Example**:
```
Initial: 100 labeled emails (spam/not spam)
Unlabeled: 10,000 emails

Step 1: Train on 100 labeled → accuracy 85%
Step 2: Predict on 10,000 unlabeled
Step 3: Select top 1,000 high-confidence predictions
Step 4: Retrain on 100 labeled + 1,000 pseudo-labeled
Step 5: Repeat with remaining 9,000 unlabeled
```

#### **2. Co-Training**

**Core Idea**: Train two models on different views/features of the data, and they teach each other.

**Process**:
1. Split features into two independent sets (View 1 and View 2)
2. Train Model 1 on View 1 of labeled data
3. Train Model 2 on View 2 of labeled data
4. Each model predicts on unlabeled data
5. High-confidence predictions from Model 1 become labels for Model 2 (and vice versa)
6. Retrain both models iteratively

**Requirements**:
- Features can be split into two independent sets
- Each view is sufficient for learning
- Views are conditionally independent given the class

**Example**:
- **View 1**: Text content of web pages
- **View 2**: Hyperlinks pointing to web pages
- Model 1 learns from text, Model 2 learns from links
- They label unlabeled pages for each other

**Mathematical Formulation**:
```
Given: L = {(x₁, y₁), ..., (xₗ, yₗ)}, U = {xₗ₊₁, ..., xₗ₊ᵤ}
Features: X = [X₁, X₂] where X₁ and X₂ are independent views

Train f₁ on (X₁, Y) from L
Train f₂ on (X₂, Y) from L

For each iteration:
  - f₁ predicts on U → high-confidence predictions → add to L₂
  - f₂ predicts on U → high-confidence predictions → add to L₁
  - Retrain f₁ on L ∪ L₁
  - Retrain f₂ on L ∪ L₂
```

#### **3. Majority Voting**

**When Predictions Differ**:
- Multiple models make predictions on same data
- Use majority vote to decide final label
- Reduces individual model errors

**Process**:
1. Train multiple models (different algorithms or different views)
2. Each model predicts on unlabeled data
3. For each data point, collect all predictions
4. Assign label based on majority vote
5. Use agreed-upon labels for training

**Example**:
```
3 models predict on unlabeled email:
- Model 1: Spam (confidence: 0.9)
- Model 2: Spam (confidence: 0.7)
- Model 3: Not Spam (confidence: 0.6)

Majority vote: Spam (2 out of 3)
Use this as pseudo-label
```

### 1.5 Mathematical Foundation

#### **Objective Function**

**Combined Loss**:
```
L_total = L_supervised + λ × L_unsupervised
```

Where:
- **L_supervised**: Loss on labeled data (e.g., cross-entropy)
- **L_unsupervised**: Loss on unlabeled data (e.g., consistency loss)
- **λ**: Weight balancing the two terms

#### **Consistency Regularization**

**Idea**: Model should make consistent predictions for similar inputs.

**Formulation**:
```
L_unsupervised = Σ D(f(x), f(x'))
```

Where:
- **x'**: Augmented or perturbed version of x
- **D**: Distance metric (e.g., KL divergence, MSE)
- **f**: Model predictions

**Example**: 
- Add noise to image → predictions should be similar
- Apply data augmentation → predictions should be consistent

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

### 1.7 Advantages & Limitations

#### **Advantages**:
- ✅ Reduces need for expensive labeled data
- ✅ Leverages abundant unlabeled data
- ✅ Can improve performance over supervised-only
- ✅ Works with any base classifier
- ✅ Practical for real-world scenarios

#### **Limitations**:
- ❌ Risk of error propagation (wrong pseudo-labels)
- ❌ Requires assumptions (smoothness, cluster, manifold)
- ❌ May not help if labeled data is too small
- ❌ Sensitive to initial model quality
- ❌ Can amplify biases in base classifier

### 1.8 Applications

**Text Classification**:
- Email spam detection
- Sentiment analysis
- Document categorization

**Computer Vision**:
- Image classification
- Object detection
- Medical image analysis

**Speech Recognition**:
- Transcribe audio with few labeled examples
- Leverage large unlabeled audio corpora

---

## 2. Self-Supervised Learning (2.4)

### 2.1 Intuition & Goal

**Goal**: Learn useful representations from data itself without external labels by creating supervision signals from the data structure.

**Key Insight**: Data contains inherent structure that can be used as supervision!

**Real-World Analogy**:
- Learning to read: You don't need someone to tell you every word
- You can learn by predicting missing words in sentences
- The sentence structure itself provides the "label" (what word should go here)

**Famous Example**: GPT (Generative Pre-trained Transformer)
- Mask words in text
- Predict what word should fill the blank
- Learn language representations from this task

### 2.2 Core Concepts

#### **What is Self-Supervision?**

**Traditional Supervised Learning**:
- Need external labels: (image, "cat"), (text, "positive")
- Labels come from humans or external sources

**Self-Supervised Learning**:
- Create labels from data itself
- Task: Predict part of data from other parts
- No external labeling needed!

**Key Principle**: 
- Design a "pretext task" that forces the model to learn useful representations
- The pretext task is not the final goal, but helps learn good features

#### **Why It Matters**

**The Labeling Problem**:
- Labeling is expensive and time-consuming
- Some domains have too much data to label manually
- Self-supervision scales to massive datasets

**Representation Learning**:
- Learn general-purpose features
- Can transfer to downstream tasks
- Foundation for many modern AI systems

### 2.3 How Self-Supervised Learning Works

#### **General Framework**

1. **Design Pretext Task**:
   - Create a task that requires understanding data structure
   - Task should be solvable from data alone
   - Examples: predict missing parts, predict next item, predict transformation

2. **Generate Pseudo-Labels**:
   - Automatically create labels from data
   - No human annotation needed
   - Labels come from data structure

3. **Train Model**:
   - Train on pretext task
   - Model learns useful representations
   - Representations capture underlying patterns

4. **Transfer to Downstream Task**:
   - Use learned representations
   - Fine-tune on actual task (with or without labels)
   - Often achieves better performance

### 2.4 Key Methods

#### **1. Masked Language Modeling (GPT-style)**

**Process**:
1. Take a sentence: "The cat sat on the mat"
2. Mask some words: "The [MASK] sat on the [MASK]"
3. Train model to predict masked words
4. Model learns language structure

**Mathematical Formulation**:
```
Given sentence: x = [x₁, x₂, ..., xₙ]
Mask positions: M = {i₁, i₂, ..., iₘ}

Task: Predict P(xᵢ | x\ᵢ) for i ∈ M

Where:
- x\ᵢ: All words except position i
- Model learns: f(x\ᵢ) → probability distribution over vocabulary
```

**Why It Works**:
- To predict masked word, model must understand:
  - Grammar and syntax
  - Semantic relationships
  - Context and meaning
- These are exactly the representations we want!

**Example**:
```
Input: "I love to [MASK] in the morning"
Possible predictions:
- "run" (high probability - makes sense)
- "eat" (medium probability - also valid)
- "fly" (low probability - less likely)

Model learns that "run" and "eat" are more likely after "love to"
```

#### **2. Next Token Prediction**

**Process**:
1. Given sequence: [x₁, x₂, ..., xₜ]
2. Predict next token: xₜ₊₁
3. Train autoregressively (predict one token at a time)

**Example**:
```
Sequence: "The weather is"
Next token prediction: "nice" (or "sunny", "rainy", etc.)

Model learns:
- Sequential dependencies
- Language patterns
- Context understanding
```

#### **3. Contrastive Learning**

**Core Idea**: Learn by contrasting similar vs. different examples.

**Process**:
1. Create positive pairs (similar examples)
2. Create negative pairs (different examples)
3. Train model to:
   - Bring positive pairs closer in representation space
   - Push negative pairs apart

**Example - Images**:
- Positive pair: Same image with different augmentations (rotation, crop)
- Negative pair: Different images
- Model learns: Augmented versions should have similar representations

**Mathematical Formulation**:
```
Given:
- Anchor: x
- Positive: x⁺ (similar to x)
- Negatives: {x₁⁻, x₂⁻, ..., xₖ⁻} (different from x)

Objective: Maximize similarity(x, x⁺) and minimize similarity(x, xᵢ⁻)

Loss: -log(exp(sim(x, x⁺)) / (exp(sim(x, x⁺)) + Σexp(sim(x, xᵢ⁻))))
```

#### **4. Autoencoding**

**Process**:
1. Encode input to latent representation
2. Decode back to original input
3. Learn to reconstruct accurately

**Why It Works**:
- To reconstruct well, model must capture essential information
- Latent representation becomes useful feature

**Example**:
```
Input image → Encoder → Latent code → Decoder → Reconstructed image
Loss: ||original - reconstructed||²

Model learns compact representation that preserves important information
```

### 2.5 Pretext Tasks by Domain

#### **Natural Language Processing**

**Masked Language Modeling**:
- BERT, GPT: Predict masked words
- Learn word embeddings and context

**Next Sentence Prediction**:
- Predict if sentence B follows sentence A
- Learn discourse and coherence

**Sentence Ordering**:
- Given shuffled sentences, predict correct order
- Learn narrative structure

#### **Computer Vision**

**Image Inpainting**:
- Mask part of image, predict missing region
- Learn spatial structure

**Rotation Prediction**:
- Predict rotation angle of rotated image
- Learn object orientation

**Jigsaw Puzzles**:
- Rearrange shuffled image patches
- Learn spatial relationships

**Colorization**:
- Predict color from grayscale image
- Learn semantic understanding

#### **Time Series**

**Temporal Prediction**:
- Predict next value in sequence
- Learn temporal patterns

**Forecasting**:
- Predict future values from past
- Learn trends and seasonality

### 2.6 Transfer to Downstream Tasks

#### **Fine-Tuning**

**Process**:
1. Pre-train on self-supervised task (large unlabeled dataset)
2. Fine-tune on downstream task (small labeled dataset)
3. Often achieves better performance than training from scratch

**Example - NLP**:
```
Step 1: Pre-train GPT on Wikipedia (self-supervised, no labels)
Step 2: Fine-tune on sentiment analysis (small labeled dataset)
Result: Better than training from scratch on sentiment data alone
```

**Example - Vision**:
```
Step 1: Pre-train on ImageNet (self-supervised, no labels needed)
Step 2: Fine-tune on medical images (small labeled dataset)
Result: Better performance with less labeled data
```

#### **Feature Extraction**

**Process**:
1. Pre-train on self-supervised task
2. Extract features from pre-trained model
3. Train simple classifier on extracted features
4. Often works well with minimal fine-tuning

### 2.7 When to Use Self-Supervised Learning

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

### 2.8 Advantages & Limitations

#### **Advantages**:
- ✅ No need for labeled data (scales to massive datasets)
- ✅ Learns rich, general-purpose representations
- ✅ Transfers well to downstream tasks
- ✅ Foundation for modern AI (GPT, BERT, etc.)
- ✅ Can leverage structure in any data type

#### **Limitations**:
- ❌ Pre-training can be computationally expensive
- ❌ Pretext task must be carefully designed
- ❌ May not always transfer to target task
- ❌ Requires large amounts of data
- ❌ Less interpretable than supervised learning

### 2.9 Applications

**Large Language Models**:
- GPT, BERT, T5: Pre-trained on text, fine-tuned for specific tasks
- ChatGPT, Claude: Built on self-supervised foundation

**Computer Vision**:
- Image classification
- Object detection
- Medical imaging

**Speech Recognition**:
- Learn audio representations
- Transfer to transcription tasks

---

## 3. Other Learning Paradigms (2.5)

### 3.1 Multi-Model Learning

### 3.1.1 Intuition & Goal

**Goal**: Train multiple models on different data subsets or views, and have them collaborate to improve learning.

**Key Insight**: Different models see data differently - combine their perspectives!

**Real-World Analogy**:
- Medical diagnosis: Multiple doctors examine patient from different angles
- Each doctor has different expertise (different "views" of the data)
- Combine their opinions for better diagnosis

### 3.1.2 Core Concepts

#### **Multiple Models, Multiple Views**

**Different Data Subsets**:
- Model 1: Trained on subset A
- Model 2: Trained on subset B
- Model 3: Trained on subset C
- Each sees different examples

**Different Feature Views**:
- Model 1: Uses features [X₁, X₂, X₃]
- Model 2: Uses features [X₄, X₅, X₆]
- Each sees different aspects of data

**Different Algorithms**:
- Model 1: Decision Tree
- Model 2: Neural Network
- Model 3: SVM
- Each has different inductive bias

### 3.1.3 How Multi-Model Learning Works

#### **Process**

1. **Train Multiple Models**:
   - Each model on different subset/view
   - Models learn complementary information

2. **High-Confidence Labeling**:
   - Each model predicts on unlabeled data
   - High-confidence predictions become labels for other models
   - Models teach each other

3. **Iterative Refinement**:
   - Models retrain with new labels from other models
   - Process repeats until convergence

4. **Final Prediction**:
   - Combine predictions from all models
   - Ensemble or majority voting

#### **Mathematical Formulation**

```
Given:
- Labeled data: L = {(x₁, y₁), ..., (xₗ, yₗ)}
- Unlabeled data: U = {xₗ₊₁, ..., xₗ₊ᵤ}
- M models: f₁, f₂, ..., fₘ
- Each model trained on different subset/view

For each iteration:
  For each model fᵢ:
    - Predict on U: ŷ = fᵢ(x) for x ∈ U
    - Select high-confidence predictions: {x | confidence(ŷ) > threshold}
    - Add to training set for other models
  Retrain all models on expanded training sets
```

### 3.1.4 Key Variants

#### **Multi-View Learning**

**Definition**: Data has multiple independent views/representations.

**Example**:
- Web page classification:
  - View 1: Text content
  - View 2: Hyperlink structure
  - View 3: Images on page
- Each view provides complementary information

**Process**:
1. Train separate model for each view
2. Models make predictions independently
3. Combine predictions (voting, averaging)
4. High-confidence predictions from one view label data for others

#### **Co-Training (Multi-Model Variant)**

**Two-Model Co-Training**:
- Model 1: Trained on View 1
- Model 2: Trained on View 2
- They label unlabeled data for each other

**Multi-Model Extension**:
- N models, each on different view
- All models collaborate
- More robust than two-model version

### 3.1.5 When to Use Multi-Model Learning

**✅ Use When:**
- Data has multiple natural views/representations
- Different models capture different aspects
- Want to leverage model diversity
- Have computational resources for multiple models
- Working with semi-supervised setting

**❌ Don't Use When:**
- Single view is sufficient
- Computational resources limited
- Models are too similar (no diversity benefit)
- Simple problem that doesn't need complexity

### 3.1.6 Applications

**Fraud Detection**:
- Model 1: Transaction patterns
- Model 2: User behavior
- Model 3: Network analysis
- Combine for better detection

**Object Detection**:
- Model 1: RGB images
- Model 2: Depth information
- Model 3: Motion cues
- Multi-modal fusion

**Medical Diagnosis**:
- Model 1: Patient symptoms
- Model 2: Lab results
- Model 3: Imaging data
- Collaborative diagnosis

---

### 3.2 Active Learning

### 3.2.1 Intuition & Goal

**Goal**: Intelligently select which data points to label, minimizing labeling cost while maximizing model performance.

**Key Insight**: Not all data points are equally valuable for learning!

**Real-World Analogy**:
- Student studying for exam
- Instead of reading entire textbook, focus on areas you're uncertain about
- Ask teacher (oracle) about confusing topics
- Learn more efficiently

**The Oracle**:
- Human expert who can provide labels
- Expensive resource (time, money)
- Use strategically!

### 3.2.2 Core Concepts

#### **The Active Learning Loop**

1. **Start with Small Labeled Set**:
   - Initial labeled data: L₀
   - Train initial model

2. **Query Strategy**:
   - Model identifies uncertain/unlabeled points
   - Selects most informative examples
   - Queries oracle for labels

3. **Update Model**:
   - Add newly labeled examples
   - Retrain model
   - Model improves

4. **Repeat**:
   - Continue until:
     - Performance is satisfactory
     - Budget exhausted
     - No more informative examples

#### **Query Strategies**

**1. Uncertainty Sampling**:
- Select points where model is most uncertain
- High entropy predictions
- Model is "confused" about these points

**Mathematical Formulation**:
```
Uncertainty(x) = H(P(y|x)) = -Σ P(yᵢ|x) × log(P(yᵢ|x))

Select: argmax_{x ∈ U} Uncertainty(x)
```

**Example**:
- Model predicts: [0.45, 0.50, 0.05] (high uncertainty)
- Model predicts: [0.95, 0.03, 0.02] (low uncertainty)
- Query the first one!

**2. Query-by-Committee**:
- Train multiple models (committee)
- Points where models disagree are informative
- Select points with maximum disagreement

**Process**:
1. Train K different models
2. Each makes prediction on unlabeled data
3. Measure disagreement (variance, entropy)
4. Query points with highest disagreement

**3. Expected Model Change**:
- Select points that would change model the most
- Measure gradient or parameter change
- Most "surprising" examples

**4. Expected Error Reduction**:
- Select points that would reduce error the most
- Simulate labeling and measure error reduction
- Most "impactful" examples

### 3.2.3 How Active Learning Works

#### **Step-by-Step Process**

```
1. Initialization:
   L = {(x₁, y₁), ..., (xₗ, yₗ)}  // Small labeled set
   U = {xₗ₊₁, ..., xₗ₊ᵤ}          // Large unlabeled set
   Budget = B                      // Labeling budget

2. Train model f on L

3. While |L| < B:
   a. For each x ∈ U:
        - Compute informativeness: I(x) = QueryStrategy(f, x)
   
   b. Select: x* = argmax_{x ∈ U} I(x)
   
   c. Query oracle: y* = Oracle(x*)
   
   d. Update: L = L ∪ {(x*, y*)}
              U = U \ {x*}
   
   e. Retrain: f = Train(L)

4. Return final model f
```

### 3.2.4 Query Strategies Explained

#### **Uncertainty Sampling**

**Types**:

**Least Confident**:
```
x* = argmax_{x ∈ U} (1 - max_y P(y|x))
```
- Select point where max probability is lowest
- Model is least confident about top prediction

**Margin Sampling**:
```
x* = argmin_{x ∈ U} (P(y₁|x) - P(y₂|x))
```
- y₁, y₂: Top two predictions
- Small margin = model unsure between top two classes

**Entropy-Based**:
```
x* = argmax_{x ∈ U} H(P(y|x))
```
- High entropy = high uncertainty
- Most common approach

#### **Query-by-Committee**

**Disagreement Measures**:

**Vote Entropy**:
```
H_vote(x) = -Σ (V(y)/K) × log(V(y)/K)
```
- V(y): Number of models voting for class y
- K: Total number of models
- High when models disagree

**KL Divergence**:
```
KL(P_avg || P_i) for each model i
```
- Measure how much each model diverges from average
- High divergence = high disagreement

### 3.2.5 When to Use Active Learning

**✅ Use When:**
- Labeling is expensive (time, money, expertise)
- Large pool of unlabeled data available
- Can interactively query oracle
- Model uncertainty can be estimated
- Want to minimize labeling cost

**❌ Don't Use When:**
- Labeling is cheap and fast
- Small dataset (may not benefit)
- Batch labeling required (can't query interactively)
- Oracle is unreliable
- Model uncertainty estimates are poor

### 3.2.6 Advantages & Limitations

#### **Advantages**:
- ✅ Reduces labeling cost significantly
- ✅ Focuses on informative examples
- ✅ Often achieves same performance with fewer labels
- ✅ Interactive learning process
- ✅ Practical for real-world scenarios

#### **Limitations**:
- ❌ Requires interactive oracle (may not always be available)
- ❌ Query strategy must be well-designed
- ❌ Initial model quality matters
- ❌ May have cold start problem
- ❌ Computational overhead for query selection

### 3.2.7 Applications

**Medical Imaging**:
- Radiologist labels uncertain cases
- Focus on difficult diagnoses
- Reduce annotation time

**Text Classification**:
- Human annotator labels uncertain documents
- Focus on ambiguous cases
- Efficient labeling

**Speech Recognition**:
- Transcribe uncertain audio segments
- Focus on difficult pronunciations
- Improve with less data

---

### 3.3 Transfer Learning

### 3.3.1 Intuition & Goal

**Goal**: Leverage knowledge learned from one task (source) to improve performance on a related but different task (target).

**Key Insight**: Knowledge is transferable across related domains!

**Real-World Analogy**:
- Learning to drive a car helps you learn to drive a truck
- Knowledge of French helps you learn Spanish
- Skills transfer across related tasks

**The Transfer Problem**:
- Source task: Large labeled dataset, well-studied
- Target task: Small labeled dataset, new domain
- Transfer knowledge from source to target

### 3.3.2 Core Concepts

#### **Source vs. Target Tasks**

**Source Task (Pre-training)**:
- Large labeled dataset: D_source
- Well-studied domain
- Model learns general representations
- Examples: ImageNet (images), Wikipedia (text)

**Target Task (Fine-tuning)**:
- Small labeled dataset: D_target
- Related but different domain
- Adapt pre-trained model
- Examples: Medical images, domain-specific text

#### **What Gets Transferred?**

**Low-Level Features**:
- Edges, textures, patterns (vision)
- Word embeddings, syntax (NLP)
- General patterns that work across domains

**High-Level Representations**:
- Object parts, semantic concepts
- Language understanding
- Abstract knowledge

**Architecture Knowledge**:
- Model structure
- Learning strategies
- Optimization insights

### 3.3.3 How Transfer Learning Works

#### **Process**

1. **Pre-train on Source Task**:
   - Train model on large source dataset
   - Learn general-purpose representations
   - Model captures domain knowledge

2. **Adapt to Target Task**:
   - Option A: Fine-tune entire model
   - Option B: Freeze early layers, fine-tune later layers
   - Option C: Use as feature extractor, train new classifier

3. **Evaluate on Target Task**:
   - Test on target domain
   - Usually better than training from scratch

#### **Fine-Tuning Strategies**

**Full Fine-Tuning**:
- Update all parameters
- Use small learning rate
- Risk of catastrophic forgetting

**Layer-wise Fine-Tuning**:
- Freeze early layers (low-level features)
- Fine-tune later layers (high-level features)
- Common in vision: freeze CNN, fine-tune classifier

**Feature Extraction**:
- Freeze entire pre-trained model
- Extract features
- Train new classifier on top
- Fastest, least flexible

### 3.3.4 Transfer Learning Scenarios

#### **Scenario 1: Same Task, Different Domain**

**Example**:
- Source: Natural images (ImageNet)
- Target: Medical images (X-rays)
- Task: Image classification (same task, different domain)

**Process**:
1. Pre-train CNN on ImageNet
2. Fine-tune on medical images
3. Model adapts to new domain

#### **Scenario 2: Different Task, Same Domain**

**Example**:
- Source: Image classification
- Target: Object detection
- Domain: Natural images (same)

**Process**:
1. Pre-train on classification
2. Adapt architecture for detection
3. Fine-tune on detection task

#### **Scenario 3: Different Task, Different Domain**

**Example**:
- Source: Text classification
- Target: Image captioning
- Different tasks and domains

**Process**:
- Transfer high-level concepts
- Adapt architecture significantly
- More challenging

### 3.3.5 Mathematical Foundation

#### **Transfer Learning Objective**

```
L_target = L_supervised(D_target) + λ × L_transfer(f_source, f_target)
```

Where:
- **L_supervised**: Loss on target task
- **L_transfer**: Regularization from source model
- **λ**: Transfer weight
- **f_source**: Pre-trained source model
- **f_target**: Target model being trained

#### **Feature Similarity**

**Idea**: Representations should be similar for similar inputs.

```
L_transfer = ||f_source(x) - f_target(x)||²
```

- Encourage target model to learn similar features
- Especially for early layers

### 3.3.6 When to Use Transfer Learning

**✅ Use When:**
- Target task has limited labeled data
- Source and target tasks are related
- Pre-trained models available for source task
- Computational resources limited
- Want to leverage existing knowledge

**❌ Don't Use When:**
- Source and target are very different
- Have sufficient labeled data for target
- Pre-trained models don't exist
- Transfer might hurt (negative transfer)
- Target task is too different from source

### 3.3.7 Advantages & Limitations

#### **Advantages**:
- ✅ Reduces need for labeled target data
- ✅ Faster training (start from good initialization)
- ✅ Better performance with less data
- ✅ Leverages large-scale pre-training
- ✅ Common practice in modern ML

#### **Limitations**:
- ❌ Requires related source task
- ❌ May have negative transfer (hurts performance)
- ❌ Catastrophic forgetting possible
- ❌ Domain shift issues
- ❌ Architecture constraints

### 3.3.8 Applications

**Natural Language Processing**:
- BERT, GPT: Pre-trained on large text, fine-tuned for specific tasks
- Sentiment analysis, question answering, translation

**Computer Vision**:
- ImageNet pre-training → medical imaging, satellite imagery
- Object detection, segmentation

**Speech Recognition**:
- Pre-train on large audio corpus
- Fine-tune for specific languages or accents

---

### 3.4 Federated Learning

### 3.4.1 Intuition & Goal

**Goal**: Train machine learning models across decentralized data sources (devices) without centralizing the data.

**Key Insight**: Keep data local, share only model updates!

**Real-World Problem**:
- Data is distributed (phones, hospitals, companies)
- Privacy concerns (can't share raw data)
- Regulatory requirements (GDPR, HIPAA)
- Still want to train good models

**Real-World Analogy**:
- Multiple hospitals want to train medical AI
- Can't share patient data (privacy)
- But can share "lessons learned" (model updates)
- Aggregate knowledge without sharing data

### 3.4.2 Core Concepts

#### **The Federated Setting**

**Distributed Data**:
- Data on N devices: D₁, D₂, ..., Dₙ
- Each device has local dataset
- Data never leaves device

**Central Server**:
- Coordinates training
- Aggregates model updates
- Distributes global model
- Never sees raw data

**Privacy Preservation**:
- Raw data stays on devices
- Only model updates (weights) shared
- Updates can be encrypted
- Differential privacy possible

### 3.4.3 How Federated Learning Works

#### **Federated Averaging Algorithm**

**Process**:

1. **Initialization**:
   - Server initializes global model: w⁰
   - Distributes to all devices

2. **Local Training (Round t)**:
   - Each device i:
     - Receives global model: wᵗ
     - Trains on local data Dᵢ
     - Computes local update: Δwᵢᵗ
     - Sends update to server (encrypted)

3. **Aggregation**:
   - Server receives updates: {Δw₁ᵗ, Δw₂ᵗ, ..., Δwₙᵗ}
   - Aggregates: wᵗ⁺¹ = wᵗ + (1/n) × ΣΔwᵢᵗ
   - Or weighted average based on data size

4. **Distribution**:
   - Server sends updated model wᵗ⁺¹ to devices
   - Repeat for T rounds

#### **Mathematical Formulation**

```
Given:
- N devices with local datasets: D₁, D₂, ..., Dₙ
- Global model: w
- Local loss on device i: Lᵢ(w) = (1/|Dᵢ|) Σ l(w, x, y) for (x,y) ∈ Dᵢ

Federated Learning Objective:
Minimize: L(w) = Σ (|Dᵢ|/|D|) × Lᵢ(w)

Where |D| = Σ|Dᵢ| is total data size

Federated Averaging:
wᵗ⁺¹ = Σ (|Dᵢ|/|D|) × wᵢᵗ

Where wᵢᵗ is model trained on device i at round t
```

### 3.4.4 Key Challenges

#### **1. Non-IID Data**

**Problem**:
- Data on different devices may have different distributions
- Example: Phone 1 has photos of cats, Phone 2 has photos of dogs
- Standard averaging may not work well

**Solutions**:
- Weighted averaging (by data size)
- Clustered federated learning
- Personalization techniques

#### **2. Communication Efficiency**

**Problem**:
- Sending full model updates is expensive
- Limited bandwidth on mobile devices
- Need to reduce communication

**Solutions**:
- Compression (quantization, sparsification)
- Gradient compression
- Local training for multiple epochs before sending

#### **3. Privacy & Security**

**Problem**:
- Model updates might leak information about data
- Need to protect against inference attacks

**Solutions**:
- Differential privacy
- Secure aggregation (homomorphic encryption)
- Federated learning with secure multiparty computation

#### **4. System Heterogeneity**

**Problem**:
- Devices have different:
  - Computational power
  - Network connectivity
  - Availability (some devices offline)

**Solutions**:
- Asynchronous updates
- Device selection strategies
- Tolerant to stragglers

### 3.4.5 Privacy Techniques

#### **Differential Privacy**

**Idea**: Add noise to updates to prevent information leakage.

```
Δw_noisy = Δw + N(0, σ²)
```

- Noise masks individual contributions
- Privacy-accuracy tradeoff
- Formal privacy guarantees

#### **Secure Aggregation**

**Idea**: Encrypt updates so server can aggregate without seeing individual updates.

- Homomorphic encryption
- Secure multiparty computation
- Server computes sum without seeing individual terms

### 3.4.6 When to Use Federated Learning

**✅ Use When:**
- Data is distributed across devices/organizations
- Privacy is critical (can't centralize data)
- Regulatory compliance required
- Want to leverage distributed data
- Have communication infrastructure

**❌ Don't Use When:**
- Data can be centralized safely
- Communication costs too high
- Devices are unreliable
- Need real-time updates
- Simple centralized solution sufficient

### 3.4.7 Advantages & Limitations

#### **Advantages**:
- ✅ Preserves data privacy
- ✅ Complies with regulations (GDPR, HIPAA)
- ✅ Leverages distributed data
- ✅ Scales to many devices
- ✅ Reduces central storage needs

#### **Limitations**:
- ❌ Communication overhead
- ❌ Complex system design
- ❌ Non-IID data challenges
- ❌ Slower convergence
- ❌ Security/privacy tradeoffs

### 3.4.8 Applications

**Mobile Devices**:
- Google Keyboard: Learn from typing on millions of phones
- Apple Siri: Improve without sending audio to servers
- Predictive text, autocorrect

**Healthcare**:
- Multiple hospitals train medical AI
- Patient data stays at hospitals
- Aggregate knowledge for better models

**Finance**:
- Banks collaborate on fraud detection
- Customer data stays at each bank
- Shared model improvements

---

### 3.5 Meta-Learning

### 3.5.1 Intuition & Goal

**Goal**: Learn how to learn - develop algorithms that can quickly adapt to new tasks with few examples.

**Key Insight**: Learning itself can be learned!

**Real-World Analogy**:
- Expert learner: Someone who quickly masters new skills
- They've learned "how to learn"
- Meta-learning: Teach AI to be an expert learner

**The Meta-Problem**:
- Traditional ML: Learn task T from data D
- Meta-Learning: Learn learning algorithm A that can quickly learn new tasks

### 3.5.2 Core Concepts

#### **Learning to Learn**

**Traditional Learning**:
```
Given: Task T, Data D
Learn: Model f that performs well on T
```

**Meta-Learning**:
```
Given: Distribution of tasks P(T)
Learn: Learning algorithm A that quickly learns new tasks from P(T)

Then: For new task T' ~ P(T), use A to learn quickly
```

#### **Few-Shot Learning**

**Problem**: Learn new task with very few examples (1-5 examples per class).

**Example**:
- See 3 examples of "giraffe"
- Recognize giraffes in new images
- Human-like learning ability

**Meta-Learning Solution**:
- Train on many "few-shot learning tasks"
- Learn to extract useful features quickly
- Transfer to new few-shot tasks

### 3.5.3 How Meta-Learning Works

#### **General Framework**

1. **Meta-Training**:
   - Sample many tasks from task distribution
   - For each task, simulate few-shot learning
   - Learn learning algorithm that works across tasks

2. **Meta-Testing**:
   - New task from same distribution
   - Apply learned learning algorithm
   - Quickly adapt with few examples

#### **Mathematical Formulation**

```
Meta-Learning Objective:

Minimize: E_{T ~ P(T)} [L_T(A(D_T^train))]

Where:
- P(T): Distribution over tasks
- A: Learning algorithm (meta-learner)
- D_T^train: Training data for task T
- L_T: Loss function for task T
- A(D_T^train): Model learned by algorithm A on task T

Goal: Find A that minimizes expected loss across tasks
```

### 3.5.4 Key Approaches

#### **1. Model-Agnostic Meta-Learning (MAML)**

**Idea**: Learn good initialization that can quickly adapt to new tasks.

**Process**:
1. Initialize model parameters: θ
2. For each task T:
   - Take few gradient steps: θ' = θ - α∇L_T(θ)
   - Evaluate on task T
3. Update initialization: θ ← θ - β∇ΣL_T(θ')
4. Repeat

**Key Insight**: Initialization that's "close" to optimal for many tasks.

#### **2. Metric-Based Meta-Learning**

**Idea**: Learn a distance metric for comparing examples.

**Process**:
1. Learn embedding function: f(x)
2. For new task:
   - Embed support examples (few labeled)
   - Embed query example
   - Find nearest support example
   - Predict same label

**Example - Siamese Networks**:
- Learn to compare pairs of images
- "Are these the same class?"
- Use for few-shot classification

#### **3. Optimization-Based Meta-Learning**

**Idea**: Learn optimization algorithm itself.

**Process**:
1. Meta-learner learns update rule
2. Instead of standard gradient descent, use learned optimizer
3. Optimizer adapts to task quickly

#### **4. Memory-Augmented Meta-Learning**

**Idea**: Use external memory to store and retrieve task-specific information.

**Process**:
1. Model has memory bank
2. For new task, store key examples in memory
3. Retrieve relevant memories for prediction
4. Learn what to store and how to retrieve

### 3.5.5 Ensemble Methods as Meta-Learning

#### **Connection to Ensemble Learning**

**Ensemble Learning**:
- Combine multiple models
- Learn from diversity
- Meta-level: How to combine models?

**Meta-Learning Perspective**:
- Each model is a "learner"
- Meta-learner learns how to combine them
- Learning to learn from multiple sources

#### **Example - Random Forest**:
- Multiple decision trees (base learners)
- Voting mechanism (meta-learner)
- Learns how to aggregate predictions

### 3.5.6 When to Use Meta-Learning

**✅ Use When:**
- Need to quickly adapt to new tasks
- Have many related tasks available
- Few-shot learning required
- Transfer learning insufficient
- Want learning efficiency

**❌ Don't Use When:**
- Single task with sufficient data
- Tasks are very different
- Computational resources limited
- Simple transfer learning works
- Don't need fast adaptation

### 3.5.7 Advantages & Limitations

#### **Advantages**:
- ✅ Fast adaptation to new tasks
- ✅ Few-shot learning capability
- ✅ Generalizes learning strategies
- ✅ Human-like learning efficiency
- ✅ Foundation for advanced AI

#### **Limitations**:
- ❌ Requires many related tasks
- ❌ Computationally expensive
- ❌ Complex to implement
- ❌ May not transfer to very different tasks
- ❌ Still active research area

### 3.5.8 Applications

**Few-Shot Image Classification**:
- Recognize new object classes from few examples
- Rapid prototyping

**Robotics**:
- Quickly adapt to new environments
- Learn new manipulation tasks fast

**Natural Language Processing**:
- Adapt to new languages or domains
- Few-shot text classification

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

### 4.2 When to Combine Paradigms

#### **Semi-Supervised + Active Learning**
- Use active learning to select which unlabeled examples to label
- Combine strategic selection with unlabeled data leverage
- Most efficient use of labeling budget

#### **Self-Supervised + Transfer Learning**
- Pre-train with self-supervision (no labels)
- Fine-tune on target task (few labels)
- Common in modern NLP and vision

#### **Federated + Transfer Learning**
- Pre-train model centrally (if possible)
- Fine-tune in federated setting
- Leverage both knowledge transfer and privacy

#### **Meta-Learning + Few-Shot Active Learning**
- Meta-learn to quickly adapt
- Use active learning to select examples
- Most efficient few-shot learning

### 4.3 Integration Strategies

**Hierarchical Approach**:
1. Self-supervised pre-training (no labels)
2. Transfer to target domain (few labels)
3. Active learning for remaining labels
4. Semi-supervised with unlabeled data

**Parallel Approach**:
- Multiple paradigms simultaneously
- Ensemble their predictions
- Leverage complementary strengths

---

## 5. Best Practices & Applications

### 5.1 Choosing the Right Paradigm

#### **Decision Tree**

```
Do you have labeled data?
├─ Yes, sufficient → Supervised Learning
└─ No / Limited
   ├─ Large unlabeled data?
   │  ├─ Yes → Self-Supervised or Semi-Supervised
   │  └─ No → Active Learning
   │
   ├─ Related source task?
   │  └─ Yes → Transfer Learning
   │
   ├─ Data distributed, privacy critical?
   │  └─ Yes → Federated Learning
   │
   ├─ Need fast adaptation to new tasks?
   │  └─ Yes → Meta-Learning
   │
   └─ Multiple views/models available?
      └─ Yes → Multi-Model Learning
```

### 5.2 Common Workflows

#### **Modern NLP Pipeline**

1. **Self-Supervised Pre-training**:
   - Train GPT/BERT on large text corpus
   - Masked language modeling
   - No human labels needed

2. **Transfer Learning**:
   - Fine-tune on target task (sentiment, QA)
   - Small labeled dataset
   - Adapts to specific domain

3. **Active Learning (Optional)**:
   - Select uncertain examples
   - Human annotator labels
   - Iteratively improve

#### **Medical Imaging Pipeline**

1. **Transfer Learning**:
   - Pre-train on ImageNet (natural images)
   - Transfer to medical domain

2. **Federated Learning**:
   - Train across hospitals
   - Patient data stays local
   - Aggregate model updates

3. **Semi-Supervised**:
   - Use unlabeled medical images
   - Leverage structure in data

### 5.3 Best Practices

#### **Data Considerations**
- ✅ Ensure data quality and consistency
- ✅ Verify distribution assumptions
- ✅ Handle domain shift carefully
- ✅ Consider privacy and ethics

#### **Model Considerations**
- ✅ Start with simple baselines
- ✅ Monitor for overfitting
- ✅ Use appropriate evaluation metrics
- ✅ Consider computational constraints

#### **Implementation Considerations**
- ✅ Design good query strategies (active learning)
- ✅ Choose appropriate transfer approach
- ✅ Handle communication efficiently (federated)
- ✅ Balance privacy and performance

### 5.4 Real-World Applications

**Healthcare**:
- Federated learning across hospitals
- Transfer learning from general to medical images
- Active learning for rare disease diagnosis

**Natural Language Processing**:
- Self-supervised pre-training (GPT, BERT)
- Transfer to specific domains
- Few-shot learning for new languages

**Autonomous Vehicles**:
- Transfer learning from simulation to real world
- Federated learning across vehicle fleets
- Active learning for edge cases

**Finance**:
- Federated learning across banks
- Transfer learning for fraud detection
- Semi-supervised for transaction classification

---

## Summary: Key Takeaways

### Advanced Learning Paradigms Essentials

1. **Semi-Supervised Learning**:
   - Combines labeled and unlabeled data
   - Models teach each other (co-training, self-training)
   - Reduces labeling cost
   - Assumes data structure (smoothness, clusters)

2. **Self-Supervised Learning**:
   - Creates supervision from data structure
   - GPT: Mask words, predict/fill blanks
   - Learns general representations
   - Foundation for modern AI

3. **Multi-Model Learning**:
   - Multiple models on different views/subsets
   - Models collaborate and teach each other
   - Majority voting for predictions
   - Leverages model diversity

4. **Active Learning**:
   - Model queries oracle for uncertain points
   - Strategic data selection
   - Minimizes labeling cost
   - Interactive learning loop

5. **Transfer Learning**:
   - Pre-trained model adapted to new task
   - Leverages knowledge from source task
   - Common in NLP and computer vision
   - Reduces need for target labels

6. **Federated Learning**:
   - Train on local devices
   - Aggregate encrypted updates
   - Preserves privacy
   - Distributed learning

7. **Meta-Learning**:
   - Learning to learn
   - Fast adaptation to new tasks
   - Few-shot learning capability
   - Foundation for advanced AI

### Key Principles

- **Leverage Unlabeled Data**: Semi-supervised and self-supervised learning
- **Strategic Labeling**: Active learning minimizes cost
- **Knowledge Transfer**: Transfer learning reuses learned representations
- **Privacy Preservation**: Federated learning keeps data local
- **Learning Efficiency**: Meta-learning enables fast adaptation
- **Model Diversity**: Multi-model learning combines strengths

### When to Use Each

- **Limited Labels**: Semi-supervised, Active Learning, Transfer Learning
- **No Labels**: Self-Supervised Learning
- **Privacy Critical**: Federated Learning
- **Fast Adaptation**: Meta-Learning
- **Multiple Views**: Multi-Model Learning
- **Related Tasks**: Transfer Learning

---

## Practice Problems & Exercises

### Conceptual Questions

1. **What's the difference between semi-supervised and self-supervised learning?**
2. **Why does co-training require independent feature views?**
3. **How does active learning reduce labeling cost?**
4. **When would transfer learning hurt performance (negative transfer)?**
5. **What are the main challenges in federated learning?**
6. **How does meta-learning enable few-shot learning?**
7. **When would you combine multiple paradigms?**

### Practical Exercises

1. **Implement Self-Training**:
   - Start with 100 labeled examples
   - Use 10,000 unlabeled examples
   - Iteratively add high-confidence predictions
   - Compare to supervised-only baseline

2. **Design Active Learning Strategy**:
   - Implement uncertainty sampling
   - Compare to random sampling
   - Measure labels needed to reach target accuracy

3. **Transfer Learning Experiment**:
   - Pre-train on ImageNet
   - Fine-tune on medical images
   - Compare to training from scratch
   - Measure data efficiency

4. **Federated Learning Simulation**:
   - Simulate 10 devices with local data
   - Implement federated averaging
   - Compare to centralized training
   - Analyze communication cost

5. **Meta-Learning for Few-Shot**:
   - Implement MAML
   - Train on many few-shot tasks
   - Test on new few-shot task
   - Compare adaptation speed

### Advanced Questions

1. **How would you combine semi-supervised and active learning?**
2. **What privacy guarantees does federated learning provide?**
3. **How does self-supervised pre-training help transfer learning?**
4. **When does meta-learning outperform transfer learning?**
5. **How to handle non-IID data in federated learning?**

---

**End of Advanced Learning Paradigms Guide**

*This comprehensive guide covers semi-supervised learning, self-supervised learning, and other advanced learning paradigms from your course materials, providing intuitive explanations, mathematical foundations, and practical insights for each approach.*

