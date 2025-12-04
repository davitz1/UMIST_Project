#01 data preparation
# Setup
import scipy.io as sio
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
import os

PROJECT_ROOT = ".."

DIRS = {
    "data": os.path.join(PROJECT_ROOT, "outputs", "data"),
    "models": os.path.join(PROJECT_ROOT, "outputs", "models"),
    "figures": os.path.join(PROJECT_ROOT, "outputs", "figures"),
    "results": os.path.join(PROJECT_ROOT, "outputs", "results"),
}
# Makes sure all defined output directories exist. If they don't, create them.
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

# Load the UMIST face dataset from the MATLAB file
mat = sio.loadmat("data/umist_cropped.mat")
faces = mat["facedat"][0]
names = mat["dirnames"][0]

X_list = []
y_list = []
# loop through each person's image stack
for img_stack, name_arr in zip(faces, names):
    label = name_arr[0]
    h, w, n_i = img_stack.shape
    # Flatten images from (112, 92) to (10304,) vector
    imgs_flat = img_stack.reshape(h * w, n_i).T
    X_list.append(imgs_flat)
    y_list.extend([label] * n_i)

# stack all images into a single matrix
X = np.vstack(X_list)
y_raw = np.array(y_list)


df = pd.DataFrame(X)
df["label"] = y_raw
print(f"DataFrame shape: {df.shape}")
print(df.head())

## Encode string labels to integers
classes = sorted(set(y_raw))
class_to_idx = {c: i for i, c in enumerate(classes)}
y = np.array([class_to_idx[label] for label in y_raw])

print(f"Total Samples: {X.shape[0]}")
print(f"Feature Vector Size: {X.shape[1]}")

# Saving
np.save(f"{DIRS['data']}/X_raw.npy", X)
np.save(f"{DIRS['data']}/y_raw.npy", y)

# Random Samples
# Pick 5 random indices from the total number of images
indices = np.random.choice(X.shape[0], 5, replace=False)

plt.figure(figsize=(15, 5))
for i, idx in enumerate(indices):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X[idx].reshape(112, 92), cmap="gray")
    plt.axis("off")
    plt.title(f"Label: {y_raw[idx]}")

plt.suptitle("Random Sample Faces")
plt.savefig(f"{DIRS['figures']}/01_raw_samples.png")
plt.show()



#02 data splitting
# Setup
import numpy as np
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

PROJECT_ROOT = ".." 
DIRS = {
    'data':    os.path.join(PROJECT_ROOT, 'outputs', 'data'),
    'models':  os.path.join(PROJECT_ROOT, 'outputs', 'models'),
    'figures': os.path.join(PROJECT_ROOT, 'outputs', 'figures'),
    'results': os.path.join(PROJECT_ROOT, 'outputs', 'results')
}

X = np.load(f"{DIRS['data']}/X_raw.npy")
y = np.load(f"{DIRS['data']}/y_raw.npy")

# Split 1: Separate 60% Train, leave 40% for Test/Val
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
# Split 2: Divide remaining 40% equally into Val (20%) and Test (20%)
# ensures balanced classes across all sets
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

#apply scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

splits = {
    'X_train': X_train_scaled, 'y_train':y_train,
    'X_val': X_val_scaled, 'y_val': y_val,
    'X_test': X_test_scaled, 'y_test': y_test
}

for name, data in splits.items():
    np.save(f"{DIRS['data']}/{name}.npy", data)

joblib.dump(scaler, f"{DIRS['models']}/scaler.pkl")


# visualize class distribution across Train/ Val/Test sets to confirm stratified splitting worked correctly

plt.figure(figsize=(18, 5))  

def count_dist(labels):
    uniq, counts = np.unique(labels, return_counts=True)
    return dict(zip(uniq, counts))

dist_train = count_dist(y_train)
dist_val   = count_dist(y_val)
dist_test  = count_dist(y_test)

classes = sorted(np.unique(y))

train_counts = [dist_train.get(c,0) for c in classes]
val_counts   = [dist_val.get(c,0) for c in classes]
test_counts  = [dist_test.get(c,0) for c in classes]

plt.subplot(1, 3, 1)  
plt.bar(classes, train_counts)
plt.title("Train Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=90)

plt.subplot(1, 3, 2) 
plt.bar(classes, val_counts)
plt.title("Validation Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=90)

plt.subplot(1, 3, 3) 
plt.bar(classes, test_counts)
plt.title("Test Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(f"{DIRS['figures']}/02_class_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

#03 dimensionality reduction
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.decomposition import PCA
import os

# Project Root Setup
PROJECT_ROOT = ".."
DIRS = {
    "data": os.path.join(PROJECT_ROOT, "outputs", "data"),
    "models": os.path.join(PROJECT_ROOT, "outputs", "models"),
    "figures": os.path.join(PROJECT_ROOT, "outputs", "figures"),
    "results": os.path.join(PROJECT_ROOT, "outputs", "results"),
}

# Load Data 
X_train = np.load(f"{DIRS['data']}/X_train.npy")
X_test = np.load(f"{DIRS['data']}/X_test.npy")
X_val = np.load(f"{DIRS['data']}/X_val.npy")

print(f"Data Loaded. Train Shape: {X_train.shape}")

# PCA
n_components_to_test = [10, 20, 50, 100, 150]
explained_variances = []

plt.figure(figsize=(10, 6))

# We need a temporary PCA to check variance for the max components we are interested in
pca_temp = PCA(n_components=200)
pca_temp.fit(X_train)

# Calculate cumulative variance
cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)

# Plot the curve
plt.plot(range(1, 201), cumulative_variance, label="Cumulative Variance", color="blue")
plt.xlabel("Number of Components")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Explained Variance Ratio")
plt.grid(True)

# Mark specific points
for n in n_components_to_test:
    var = cumulative_variance[n - 1]
    plt.scatter(n, var, color="red", zorder=5)
    plt.text(n + 5, var - 0.02, f"{n} comps: {var:.2f}", fontsize=9)
    print(f"Components: {n}, Explained Variance: {var:.4f}")

plt.axhline(y=0.95, color="g", linestyle="--", label="95% Variance")
plt.legend()
plt.savefig(f"{DIRS['figures']}/03_pca_variance_analysis.png")
plt.show()


best_n_components = 100

print(f"Fitting final PCA with n_components={best_n_components}...")
pca = PCA(n_components=best_n_components)
Z_pca_train = pca.fit_transform(X_train)
Z_pca_test = pca.transform(X_test)
Z_pca_val = pca.transform(X_val)  # Transform validation set too

# Save Results
joblib.dump(pca, f"{DIRS['models']}/pca_model.pkl")
np.save(f"{DIRS['data']}/Z_pca_train.npy", Z_pca_train)
np.save(f"{DIRS['data']}/Z_pca_test.npy", Z_pca_test)
np.save(f"{DIRS['data']}/Z_pca_val.npy", Z_pca_val)
print("PCA Complete.")

class AutoEncoder(models.Model):
    def __init__(self, input_dim, latent_dim=40):
        super(AutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = models.Sequential([
            layers.Dense(1024, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.2),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(latent_dim, activation='linear', name='bottleneck') 
        ])
        
        # Decoder
        self.decoder = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(latent_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1024, activation='relu'),
            layers.Dense(input_dim, activation='linear') # Reconstruction
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate
input_dim = X_train.shape[1] # 10304
model = AutoEncoder(input_dim, latent_dim=60)

# Hyperparameters
epochs = 200
batch_size = 128
learning_rate = 5e-4

# Compile
model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss="mse")

# Train (Using X_train as both input and target for reconstruction)
print("Starting Autoencoder Training...")
history = model.fit(
    X_train,
    X_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(X_val, X_val),
    verbose=1,
)

# Save the full model
model.save_weights(f"{DIRS['models']}/autoencoder.weights.h5")

# Extract Features (Dimensionality Reduction)
print("Extracting compressed features...")
Z_ae_train = model.encoder.predict(X_train)
Z_ae_test = model.encoder.predict(X_test)

# Save Compressed Data
np.save(f"{DIRS['data']}/Z_ae_train.npy", Z_ae_train)
np.save(f"{DIRS['data']}/Z_ae_test.npy", Z_ae_test)

# Plot Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Autoencoder Training History (Keras)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{DIRS['figures']}/03_ae_training_loss.png", dpi=300, bbox_inches="tight")
plt.show()

# Visualize PCA and Autoencoder latent spaces in 2D by plotting their first two dimensions, colored by person ID, to compare how well they separate individuals.

y_train = np.load(f"{DIRS['data']}/y_train.npy")  # Need labels for coloring

plt.figure(figsize=(12, 5))

# Plot PCA
plt.subplot(1, 2, 1)
plt.scatter(
    Z_pca_train[:, 0], Z_pca_train[:, 1], c=y_train, cmap="tab20", s=10, alpha=0.6
)
plt.title("PCA Reduced (First 2 Components)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Person ID")

# Plot Autoencoder (First 2 dims of latent space)
plt.subplot(1, 2, 2)
plt.scatter(
    Z_ae_train[:, 0], Z_ae_train[:, 1], c=y_train, cmap="tab20", s=10, alpha=0.6
)
plt.title("Autoencoder Reduced (First 2 Latent Dims)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.colorbar(label="Person ID")

plt.tight_layout()
plt.savefig(f"{DIRS['figures']}/03_latent_space_scatter.png")
plt.show()

#04 clustering
# Setup
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import os
import matplotlib.pyplot as plt
PROJECT_ROOT = ".." 
DIRS = {
    'data':    os.path.join(PROJECT_ROOT, 'outputs', 'data'),
    'models':  os.path.join(PROJECT_ROOT, 'outputs', 'models'),
    'figures': os.path.join(PROJECT_ROOT, 'outputs', 'figures'),
    'results': os.path.join(PROJECT_ROOT, 'outputs', 'results')
}

Z_pca = np.load(f"{DIRS['data']}/Z_pca_test.npy")
Z_ae = np.load(f"{DIRS['data']}/Z_ae_test.npy")
y_true = np.load(f"{DIRS['data']}/y_test.npy")
# Elbow Method for K-Means
print("Tuning K-Means: Running Elbow Method...")
inertia = []
K_range = range(10, 41, 2)  # Test clusters from 10 to 40

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(Z_pca)  # Use PCA features for tuning
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.axvline(
    x=len(np.unique(y_true)),
    color="r",
    linestyle="--",
    label=f"Actual People ({len(np.unique(y_true))})",
)
plt.legend()
plt.savefig(f"{DIRS['figures']}/04_kmeans_elbow_tuning.png")
plt.show()

# Define Purity Function
from sklearn.metrics import confusion_matrix


def purity_score(y_true, y_pred):
    # compute confusion matrix
    contingency_matrix = confusion_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


#  Clustering & Collect Metrics
n_clusters = len(np.unique(y_true))
results = []

algorithms = {
    "KMeans": KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
    "Hierarchical": AgglomerativeClustering(n_clusters=n_clusters),
}

feature_sets = {"PCA": Z_pca, "Autoencoder": Z_ae}

# store labels to plot  later
pred_labels = {}

for feat_name, X_feat in feature_sets.items():
    for algo_name, model in algorithms.items():

        labels_pred = model.fit_predict(X_feat)

        # Save for plotting
        pred_labels[f"{feat_name}_{algo_name}"] = labels_pred

        ari_score = adjusted_rand_score(y_true, labels_pred)
        nmi_score = normalized_mutual_info_score(y_true, labels_pred)
        p_score = purity_score(y_true, labels_pred)  # Calculate Purity

        results.append(
            {
                "Feature Set": feat_name,
                "Algorithm": algo_name,
                "ARI Score": ari_score,
                "NMI Score": nmi_score,
                "Purity": p_score,
            }
        )
        print(f"Processed: {feat_name} + {algo_name}")

#Save & Print Results
df = pd.DataFrame(results)
df = df.sort_values(by=["Purity", "ARI Score"], ascending=False).reset_index(drop=True)
df.to_csv(f"{DIRS['results']}/clustering_metrics.csv", index=False)

print("\n--- Clustering Performance Summary ---")
print(df)

# Visualization
plt.figure(figsize=(15, 10))

# Plot first 4 combinations
for i, (key, labels) in enumerate(pred_labels.items()):
    if i >= 4:
        break

    # Determine which data to plot (PCA vs AE)
    feat_type = key.split("_")[0]
    data = Z_pca if feat_type == "PCA" else Z_ae

    plt.subplot(2, 2, i + 1)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab20", s=10)
    plt.title(f"{key} Clusters")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(label="Cluster ID")

plt.tight_layout()
plt.savefig(f"{DIRS['figures']}/04_clustering_results.png")
plt.show()

#05 cnn classifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import os

# Setup Directories
PROJECT_ROOT = ".."
DIRS = {
    "data": os.path.join(PROJECT_ROOT, "outputs", "data"),
    "models": os.path.join(PROJECT_ROOT, "outputs", "models"),
    "figures": os.path.join(PROJECT_ROOT, "outputs", "figures"),
    "results": os.path.join(PROJECT_ROOT, "outputs", "results"),
}

# GPU Check
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# Load Data
X_train = np.load(f"{DIRS['data']}/X_train.npy")
y_train = np.load(f"{DIRS['data']}/y_train.npy")
X_val = np.load(f"{DIRS['data']}/X_val.npy")
y_val = np.load(f"{DIRS['data']}/y_val.npy")
X_test = np.load(f"{DIRS['data']}/X_test.npy")
y_test = np.load(f"{DIRS['data']}/y_test.npy")

# Reshape for CNN: (N, 112, 92, 1)
img_height, img_width = 112, 92
X_train_cnn = X_train.reshape(-1, img_height, img_width, 1)
X_val_cnn = X_val.reshape(-1, img_height, img_width, 1)
X_test_cnn = X_test.reshape(-1, img_height, img_width, 1)

# Convert labels to One-Hot Encoding for multi-class classification
num_classes = len(np.unique(y_train))
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

print(f"Input Shape: {X_train_cnn.shape}")
print(f"Number of Classes: {num_classes}")

def build_cnn_model(input_shape, num_classes):
    model = models.Sequential(
        [
            # Block 1
            layers.Conv2D(
                32, (3, 3), padding="same", activation="relu", input_shape=input_shape
            ),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            # Block 2
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            # Block 3
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            # Flatten & Dense
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),  # Regularization
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),  # Regularization
            # Output Layer
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


model = build_cnn_model(input_shape=(112, 92, 1), num_classes=num_classes)
model.summary()

# Quick Hyperparameter Tuning Experiments

test_learning_rates = [1e-2, 1e-3, 1e-4]
val_accuracies = []

for lr in test_learning_rates:
    print(f"\nTesting learning rate: {lr}")

    temp_model = build_cnn_model(input_shape=(112, 92, 1), num_classes=num_classes)
    temp_model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    hist = temp_model.fit(
        X_train_cnn,
        y_train_cat,
        epochs=5,  # small, fast test
        batch_size=32,
        validation_data=(X_val_cnn, y_val_cat),
        verbose=0,
    )

    acc = hist.history["val_accuracy"][-1]
    val_accuracies.append(acc)
    print(f"Validation Accuracy: {acc:.4f}")

print("\nLearning Rate Tuning Results:")
for lr, acc in zip(test_learning_rates, val_accuracies):
    print(f"LR={lr} → Val Acc={acc:.4f}")

    # Hyperparameters
learning_rate = 1e-4 # Tuned lower for stability
batch_size = 32
epochs = 50

model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for better training
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

print("Starting Training...")
history = model.fit(
    X_train_cnn, y_train_cat,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val_cnn, y_val_cat),
    callbacks=callbacks_list,
    verbose=1
)

# Save Model
model.save(f"{DIRS['models']}/cnn_classifier.h5")
print("Model saved.")

# Plot Training History
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(14, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.grid(True)

plt.savefig(f"{DIRS['figures']}/05_cnn_training_history.png")
plt.show()

# Predictions
y_pred_probs = model.predict(X_test_cnn)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - CNN Classifier')
plt.tight_layout()
plt.savefig(f"{DIRS['figures']}/05_cnn_confusion_matrix.png")
plt.show()

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Save Report to CSV
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(f"{DIRS['results']}/05_cnn_classification_report.csv")
print(f"Report saved to {DIRS['results']}/05_cnn_classification_report.csv")

# Select 5 random test images
indices = np.random.choice(len(X_test), 5, replace=False)

plt.figure(figsize=(15, 4))
for i, idx in enumerate(indices):
    plt.subplot(1, 5, i + 1)
    img = X_test_cnn[idx].reshape(112, 92)
    true_lbl = y_test[idx]
    pred_lbl = y_pred[idx]

    # Color code: Green if correct, Red if wrong
    color = "green" if true_lbl == pred_lbl else "red"

    plt.imshow(img, cmap="gray")
    plt.title(f"True: {true_lbl}\nPred: {pred_lbl}", color=color)
    plt.axis("off")

plt.suptitle("Sample Test Predictions")
plt.savefig(f"{DIRS['figures']}/05_cnn_sample_predictions.png")
plt.show()
