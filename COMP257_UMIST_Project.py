"""
COMP257 UMIST Project
Combined script for Data Prep, Splitting, Dimensionality Reduction, Clustering, and CNN Classification.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix, classification_report

# ==========================================
# 0. Global Setup
# ==========================================
PROJECT_ROOT = ".."
DIRS = {
    "data": os.path.join(PROJECT_ROOT, "outputs", "data"),
    "models": os.path.join(PROJECT_ROOT, "outputs", "models"),
    "figures": os.path.join(PROJECT_ROOT, "outputs", "figures"),
    "results": os.path.join(PROJECT_ROOT, "outputs", "results"),
}

for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

print("Directories setup complete.")

# ==========================================
# 1. Data Preparation
# ==========================================
print("\n--- 1. Data Preparation ---")

# Load the UMIST face dataset
mat_path = "data/umist_cropped.mat"
if not os.path.exists(mat_path):
    if os.path.exists(os.path.join(PROJECT_ROOT, "data", "umist_cropped.mat")):
        mat_path = os.path.join(PROJECT_ROOT, "data", "umist_cropped.mat")
    else:
        print("Warning: umist_cropped.mat not found in expected paths.")

mat = sio.loadmat(mat_path)
faces = mat["facedat"][0]
names = mat["dirnames"][0]

X_list = []
y_list = []

for img_stack, name_arr in zip(faces, names):
    label = name_arr[0]
    h, w, n_i = img_stack.shape
    # Flatten images from (112, 92) to (10304,) vector
    imgs_flat = img_stack.reshape(h * w, n_i).T
    X_list.append(imgs_flat)
    y_list.extend([label] * n_i)

X = np.vstack(X_list)
y_raw = np.array(y_list)

# Encode string labels to integers
classes = sorted(set(y_raw))
class_to_idx = {c: i for i, c in enumerate(classes)}
y = np.array([class_to_idx[label] for label in y_raw])

print(f"Total Samples: {X.shape[0]}")
print(f"Feature Vector Size: {X.shape[1]}")

# Save Raw Data
np.save(f"{DIRS['data']}/X_raw.npy", X)
np.save(f"{DIRS['data']}/y_raw.npy", y)

# Plot Random Samples
indices = np.random.choice(X.shape[0], 5, replace=False)
plt.figure(figsize=(15, 5))
for i, idx in enumerate(indices):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X[idx].reshape(112, 92), cmap="gray")
    plt.axis("off")
    plt.title(f"Label: {y_raw[idx]}")
plt.suptitle("Random Sample Faces")
plt.savefig(f"{DIRS['figures']}/01_raw_samples.png")
# plt.show() # Commented out to prevent blocking in script execution

# ==========================================
# 2. Data Splitting
# ==========================================
print("\n--- 2. Data Splitting ---")

# Split 1: Separate 60% Train, leave 40% for Test/Val
X_train_raw, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
# Split 2: Divide remaining 40% equally into Val (20%) and Test (20%)
X_test_raw, X_val_raw, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Apply Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

# Save Splits
splits = {
    'X_train': X_train, 'y_train': y_train,
    'X_val': X_val, 'y_val': y_val,
    'X_test': X_test, 'y_test': y_test
}
for name, data in splits.items():
    np.save(f"{DIRS['data']}/{name}.npy", data)

joblib.dump(scaler, f"{DIRS['models']}/scaler.pkl")

# Plot Class Distribution
plt.figure(figsize=(18, 5))
def count_dist(labels):
    uniq, counts = np.unique(labels, return_counts=True)
    return dict(zip(uniq, counts))

dist_train = count_dist(y_train)
dist_val = count_dist(y_val)
dist_test = count_dist(y_test)

all_classes = sorted(np.unique(y))
train_counts = [dist_train.get(c, 0) for c in all_classes]
val_counts = [dist_val.get(c, 0) for c in all_classes]
test_counts = [dist_test.get(c, 0) for c in all_classes]

for i, (counts, title) in enumerate(zip([train_counts, val_counts, test_counts], ["Train", "Validation", "Test"])):
    plt.subplot(1, 3, i + 1)
    plt.bar(all_classes, counts)
    plt.title(f"{title} Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig(f"{DIRS['figures']}/02_class_distribution.png", dpi=300, bbox_inches='tight')
# plt.show()

# ==========================================
# 3. Dimensionality Reduction
# ==========================================
print("\n--- 3. Dimensionality Reduction ---")

# PCA Analysis
print("Running PCA Analysis...")
pca_temp = PCA(n_components=200)
pca_temp.fit(X_train)
cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 201), cumulative_variance, label="Cumulative Variance", color="blue")
plt.axhline(y=0.95, color="g", linestyle="--", label="95% Variance")
plt.xlabel("Number of Components")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Explained Variance Ratio")
plt.legend()
plt.grid(True)
plt.savefig(f"{DIRS['figures']}/03_pca_variance_analysis.png")
# plt.show()

# Final PCA
best_n_components = 100
print(f"Fitting final PCA with n_components={best_n_components}...")
pca = PCA(n_components=best_n_components)
Z_pca_train = pca.fit_transform(X_train)
Z_pca_test = pca.transform(X_test)
Z_pca_val = pca.transform(X_val)

joblib.dump(pca, f"{DIRS['models']}/pca_model.pkl")
np.save(f"{DIRS['data']}/Z_pca_train.npy", Z_pca_train)
np.save(f"{DIRS['data']}/Z_pca_test.npy", Z_pca_test)
np.save(f"{DIRS['data']}/Z_pca_val.npy", Z_pca_val)

# AutoEncoder
print("Training AutoEncoder...")
class AutoEncoder(models.Model):
    def __init__(self, input_dim, latent_dim=40):
        super(AutoEncoder, self).__init__()
        self.encoder = models.Sequential([
            layers.Dense(1024, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.2),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(latent_dim, activation='linear', name='bottleneck') 
        ])
        self.decoder = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(latent_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1024, activation='relu'),
            layers.Dense(input_dim, activation='linear')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

input_dim = X_train.shape[1]
ae_model = AutoEncoder(input_dim, latent_dim=60)
ae_model.compile(optimizer=optimizers.Adam(learning_rate=5e-4), loss="mse")

history = ae_model.fit(
    X_train, X_train,
    epochs=200,
    batch_size=128,
    shuffle=True,
    validation_data=(X_val, X_val),
    verbose=0 # Reduced verbosity
)

ae_model.save_weights(f"{DIRS['models']}/autoencoder.weights.h5")

Z_ae_train = ae_model.encoder.predict(X_train, verbose=0)
Z_ae_test = ae_model.encoder.predict(X_test, verbose=0)
Z_ae_val = ae_model.encoder.predict(X_val, verbose=0)

np.save(f"{DIRS['data']}/Z_ae_train.npy", Z_ae_train)
np.save(f"{DIRS['data']}/Z_ae_test.npy", Z_ae_test)

# Plot AE Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Autoencoder Training History")
plt.legend()
plt.savefig(f"{DIRS['figures']}/03_ae_training_loss.png")
# plt.show()

# Latent Space Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(Z_pca_train[:, 0], Z_pca_train[:, 1], c=y_train, cmap="tab20", s=10, alpha=0.6)
plt.title("PCA Reduced (First 2 Components)")
plt.subplot(1, 2, 2)
plt.scatter(Z_ae_train[:, 0], Z_ae_train[:, 1], c=y_train, cmap="tab20", s=10, alpha=0.6)
plt.title("Autoencoder Reduced (First 2 Latent Dims)")
plt.tight_layout()
plt.savefig(f"{DIRS['figures']}/03_latent_space_scatter.png")
# plt.show()

# ==========================================
# 4. Clustering
# ==========================================
print("\n--- 4. Clustering ---")

# Elbow Method (FIXED: Using Train Data instead of Test Data)
print("Tuning K-Means: Running Elbow Method on Training Data...")
inertia = []
K_range = range(10, 41, 2)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(Z_pca_train)  # FIXED: Use Z_pca_train
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, marker="o")
plt.title("Elbow Method for Optimal k (on Train Data)")
plt.grid(True)
plt.savefig(f"{DIRS['figures']}/04_kmeans_elbow_tuning.png")
# plt.show()

# Clustering Evaluation
def purity_score(y_true, y_pred):
    contingency_matrix = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

n_clusters = len(np.unique(y_test)) # Using actual number of people for evaluation
results = []
algorithms = {
    "KMeans": KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
    "Hierarchical": AgglomerativeClustering(n_clusters=n_clusters),
}
feature_sets = {"PCA": Z_pca_test, "Autoencoder": Z_ae_test} # Evaluate on Test Data

pred_labels = {}
for feat_name, X_feat in feature_sets.items():
    for algo_name, model in algorithms.items():
        labels_pred = model.fit_predict(X_feat)
        pred_labels[f"{feat_name}_{algo_name}"] = labels_pred
        
        ari = adjusted_rand_score(y_test, labels_pred)
        nmi = normalized_mutual_info_score(y_test, labels_pred)
        purity = purity_score(y_test, labels_pred)
        
        results.append({
            "Feature Set": feat_name, "Algorithm": algo_name,
            "ARI Score": ari, "NMI Score": nmi, "Purity": purity
        })

df_results = pd.DataFrame(results).sort_values(by=["Purity"], ascending=False)
df_results.to_csv(f"{DIRS['results']}/clustering_metrics.csv", index=False)
print("Clustering Performance Summary:")
print(df_results)

# ==========================================
# 5. CNN Classifier
# ==========================================
print("\n--- 5. CNN Classifier ---")

# Reshape for CNN
img_height, img_width = 112, 92
X_train_cnn = X_train.reshape(-1, img_height, img_width, 1)
X_val_cnn = X_val.reshape(-1, img_height, img_width, 1)
X_test_cnn = X_test.reshape(-1, img_height, img_width, 1)

num_classes = len(np.unique(y_train))
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model

cnn_model = build_cnn_model((112, 92, 1), num_classes)

# Training
cnn_model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])

callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

print("Training CNN...")
history_cnn = cnn_model.fit(
    X_train_cnn, y_train_cat,
    epochs=50,
    batch_size=32,
    validation_data=(X_val_cnn, y_val_cat),
    callbacks=callbacks_list,
    verbose=1
)

cnn_model.save(f"{DIRS['models']}/cnn_classifier.h5")

# Evaluation
y_pred_probs = cnn_model.predict(X_test_cnn, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nCNN Classification Report:")
print(classification_report(y_test, y_pred))

report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
report_df.to_csv(f"{DIRS['results']}/05_cnn_classification_report.csv")

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - CNN Classifier')
plt.tight_layout()
plt.savefig(f"{DIRS['figures']}/05_cnn_confusion_matrix.png")
# plt.show()

print("\nAll tasks completed successfully.")
