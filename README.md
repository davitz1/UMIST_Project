# UMIST Face Classification Project
Plots are on the report.

## Dataset Overview (UMIST Cropped)

This project uses a cropped face image dataset packaged as a single MATLAB `.mat` file. The file contains **575 grayscale face images** belonging to **20 identities (classes)**. Images are already cropped to a fixed resolution of **112 × 92** pixels (so each image can also be represented as a **10,304-dimensional** flattened vector).

### Inside the `.mat` file

The dataset is organized into two top-level variables:

* **`facedat`**: a **1×20 MATLAB cell array**

  * Each cell corresponds to one identity/class.
  * Each cell contains a 3D array of type **`uint8`** with shape:

    * **(112, 92, K)** where **K** is the number of images for that identity.
  * The third dimension is the image index within that identity.

* **`dirnames`**: a **1×20 MATLAB cell array** containing the label for each identity

  * In this file, the labels are strings: **`1a, 1b, …, 1t`**.
  * `dirnames[0, i]` is the label for `facedat[0, i]`.

### How labels work

There is no separate per-image label array. Instead, the label is implied by which cell the image came from:

* If an image is taken from `facedat[0, i]`, its class label is `dirnames[0, i]` (e.g., `"1m"`), or equivalently an integer class id **`i`** (0–19).

### Class distribution

The dataset is not perfectly balanced; identities have between **19 and 48** images.

---

## Data Split

The data was split into a **60/20/20** ratio. This split is practical choice because it balances learning capacity with reliable evaluation:

* **60% training:** gives the model enough examples to learn meaningful patterns without starving learning (especially important when each class has a limited number of samples).
* **20% validation:** provides a solid set for model selection and tuning (hyperparameters, early stopping, architecture choices) while being large enough that validation metrics don’t swing wildly due to a few samples.
* **20% test:** reserves a fully untouched subset for an unbiased final estimate of generalization performance. With small datasets, keeping a reasonably sized test set helps avoid over-interpreting noisy results.

### Stratified Sampling

Since UMIST’s classes are not perfectly balanced, a random split can easily create skewed subsets. **Stratified sampling** solves this by forcing each split to preserve the original class proportions as closely as possible. This helps in:

* **Maintaining class coverage**

  * Ensures every identity appears in train/val/test.
  * Minimizes the risk of missing minority classes in validation/test.

* **Improving metrics reliability**

  * Reduces variance in evaluation: results are less sensitive to which exact samples happened to fall into each split.

* **Supporting fair model comparison**

  * When tuning models, validation performance changes must reflect real improvements, not accidental changes in class makeup.

---

## Dimensionality Reduction

To reduce the original **10,304-dimensional** face vectors (112×92 flattened), two complementary methods were applied: **Principal Component Analysis (PCA)** and a **feed-forward Autoencoder (AE)**.

### PCA (linear, variance-preserving baseline)

PCA projects the data onto orthogonal directions that maximize variance. It required **100 components** to achieve **95% explained variance**.

### Autoencoder (nonlinear, reconstruction-driven embedding)

The autoencoder learns a nonlinear compression by training a neural network to reconstruct the input from a bottleneck representation. The model used a deep dense encoder/decoder with Dropout regularization and a latent dimension of **60**, trained with a **batch size of 128**, **learning rate of 5e-4**, for **200 epochs** using reconstruction loss (MSE).

### 2D Latent Plots

* The 2D PCA latent plot (PC1 vs PC2) provides a rough visualization of global structure, but class colors (person IDs) still overlap substantially.
* In the 2D AE latent plot, because the mapping is nonlinear, local neighborhoods and curved manifolds can be represented more flexibly. However, like PCA, the first two latent dimensions still overlap between identities.

These overlaps are expected since only two linear components/latent dimensions cannot fully separate identity-specific variations.

---

## Clustering

After learning lower-dimensional embeddings with PCA and an Autoencoder, we applied two clustering methods to evaluate whether identities (20 people) naturally form separable groups in the learned feature spaces:

* **K-Means** was selected as a strong baseline for compact, roughly spherical clusters. It is efficient, easy to reproduce, and commonly used for embedding spaces.
* **Agglomerative (Hierarchical) Clustering** was selected to capture non-spherical structure and provide an alternative assumption set (clusters can merge based on linkage criteria rather than a single centroid objective).

### Parameter tuning

#### K-Means

* **Elbow (inertia) analysis:** the inertia curve decays smoothly rather than showing a sharp elbow, but a knee region around **k = 20** is visible.
* Since we know that we have **20 distinct individuals**, we selected **k = 20** to align with domain knowledge while remaining within the suggested knee region.

#### Hierarchical clustering

* **Number of clusters:** **20**. We forced the dendrogram to be cut into exactly 20 clusters because the dataset contains 20 identities. This makes the hierarchical output directly comparable to K-Means (also set to 20) and avoids ambiguity from choosing an arbitrary distance threshold.
* **Linkage method:** **Ward**. Ward merges clusters by minimizing the increase in within-cluster variance (SSE) at each step. It's a reasonable choice because:

  * Matches the geometry of the embeddings: both PCA and the AE latent space are continuous, Euclidean-style feature spaces and Ward is designed for them.
  * Good baseline for identity-style grouping: for faces, samples of the same person should form compact neighborhoods, and Ward encourages compactness.

### Challenges with clustering face images

* **High dimensionality makes distance unreliable**

  * Raw images live in 10,304-D space. In such spaces, Euclidean distances tend to concentrate, which weakens clustering methods that rely heavily on distance structure.
  * Dimensionality reduction helped by removing noisy pixel-level variation and making distances more meaningful in a compact embedding.

* **Identity is not the only factor driving variation**

  * Even after cropping, faces vary by pose, expression, lighting, and minor alignment differences. These factors can dominate pixel variance, causing clusters to form around pose/lighting instead of identity.
  * PCA helps by keeping dominant variance directions, which can help or hurt identity clustering depending on whether identity is expressed strongly in early components.
  * Autoencoders help by learning nonlinear structure, but since they optimize reconstruction (not identity separation), they can also preserve pose/lighting features.

### Metrics comparison

| Feature Set | Algorithm    | ARI Score | NMI Score | Purity |
| ----------- | ------------ | --------: | --------: | -----: |
| PCA         | Hierarchical |     0.376 |     0.725 |  0.574 |
| PCA         | KMeans       |     0.323 |     0.698 |  0.531 |
| Autoencoder | Hierarchical |     0.234 |     0.657 |  0.505 |
| Autoencoder | KMeans       |     0.262 |     0.665 |  0.514 |

* **Best overall:** PCA + Hierarchical (ARI ~0.376, NMI ~0.725, Purity ~0.574)
* **Next:** PCA + KMeans
* **Lower:** Autoencoder + KMeans / Hierarchical

PCA produced a feature space where identity-related structure is more recoverable by distance-based clustering, while the autoencoder embedding likely preserved reconstruction-relevant details that do not align as cleanly with identity boundaries.

---

## Neural Network Classifier

### Model architecture

We trained a convolutional neural network to classify each face image into one of **20 identities**. The model follows a standard feature extractor → classifier head design:

* **Input:** 112 × 92 × 1

**Convolutional feature extractor:**

* Conv2D (32 filters, 3×3, same padding) + ReLU → MaxPool2D (2×2) → BatchNormalization
* Conv2D (64 filters, 3×3, same padding) + ReLU → MaxPool2D (2×2) → BatchNormalization
* Conv2D (128 filters, 3×3, same padding) + ReLU → MaxPool2D (2×2) → BatchNormalization

**Classifier head:**

* Flatten
* Dense (256) + ReLU → Dropout (0.5)
* Dense (128) + ReLU → Dropout (0.5)
* Dense (20) + Softmax

**Model Summary (Params)**

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 112, 92, 32)    │           320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 56, 46, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 56, 46, 32)     │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 56, 46, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 28, 23, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 28, 23, 64)     │           256 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 28, 23, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 14, 11, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_2           │ (None, 14, 11, 128)    │           512 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 19712)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 256)            │     5,046,528 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 128)            │        32,896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 20)             │         2,580 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
```

This architecture is appropriate for face classification because early convolution layers learn low-level edges/textures, deeper layers learn higher-level facial patterns, and the dense layers map learned features into identity logits.

### Functions and optimizer

**Activation Functions**

* **ReLU** was used in convolutional and hidden dense layers because it is computationally efficient, reduces vanishing gradients, and tends to converge faster on vision tasks.
* **Softmax** was used in the output layer because the task is multi-class, single-label classification (exactly one identity per image), and softmax produces normalized class probabilities.

**Loss function**

* **Categorical cross-entropy** was used because labels were one-hot encoded, and cross-entropy is the standard objective for softmax multi-class classification. It directly optimizes the log-likelihood of the correct identity.

**Optimizer and training strategy**

* **Adam** was chosen because it adapts learning rates per-parameter and typically converges reliably on CNNs without extensive manual tuning.
* **EarlyStopping** (`patience=10`, `restore_best_weights=True`): prevents over-training once validation loss stops improving.
* **ReduceLROnPlateau** (`factor=0.5`, `patience=5`, `min_lr=1e-6`): automatically lowers the learning rate when validation progress stalls, improving fine-tuning near convergence.

### Hyperparameter tuning and regularization

**Learning rate**

We ran quickly, controlled experiments over:

* `1e-2 → val_acc ≈ 0.078`
* `1e-3 → val_acc ≈ 0.165`
* `1e-4 → val_acc ≈ 0.252`

The trend shows that larger learning rates were too aggressive (poor early validation performance), while `1e-4` provided the most stable learning. We then trained the full model with `learning_rate = 1e-4`.

**Batch size and epochs**

* `batch_size = 32` is a common spot: stable gradient estimates without slowing training too much.
* `epochs = 50`, but with EarlyStopping and LR scheduling, the effective training length is governed by validation improvement rather than a fixed epoch count.

**Regularization techniques**

* **Dropout (0.5)** in both dense layers reduces reliance on any single neuron and helps prevent memorization.
* **BatchNormalization** stabilizes training and can act as implicit regularization by reducing internal covariate shift.

---

## Training Results

The training curves show **stable convergence**:

* **Training:** accuracy rises quickly and plateaus around **~0.95**, while loss steadily decreases.
* **Validation:** accuracy is noisier in the first epochs (expected with a small validation set), then improves sharply and reaches **~1.0** as validation loss drops toward near-zero.
* **Why validation can look better than training:** validation loss becomes lower than training loss and validation accuracy slightly exceeds training accuracy, which is consistent with dropout (active during training, inactive during validation) and does not imply overfitting.

---

## Test Results

The test results indicate **perfect generalization** on the test set.

**Qualitative examples:**

The sample test predictions (true = predicted) include different poses/angles, suggesting the CNN learned identity-relevant features that remain stable under typical UMIST variations (pose/expression), at least within this dataset’s conditions.

---

## Notes / Next Steps

* Add links to notebooks/scripts used for preprocessing, PCA/AE training, clustering, and CNN training.
* Add the actual plots/images into the placeholders above for a complete report-style README.
