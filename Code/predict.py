import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
import itertools
import os
import sys
import random
import heapq

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Custom class to redirect print output to both console and file
class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()

# Step 1: Load embeddings and disease-miRNA pairs, filter invalid pairs
def load_data(disease_embeddings_file, miRNA_embeddings_file, disease_miRNA_file):
    print("Loading embeddings...")
    disease_embeddings_df = pd.read_csv(disease_embeddings_file)
    miRNA_embeddings_df = pd.read_csv(miRNA_embeddings_file)

    valid_diseases = set(disease_embeddings_df[disease_embeddings_df['type'] == 'disease']['node_id'])
    valid_miRNAs = set(miRNA_embeddings_df[miRNA_embeddings_df['type'] == 'miR']['node_id'])

    disease_emb = disease_embeddings_df[disease_embeddings_df['type'] == 'disease'].set_index('node_id')
    miRNA_emb = miRNA_embeddings_df[miRNA_embeddings_df['type'] == 'miR'].set_index('node_id')

    disease_emb_cols = [col for col in disease_embeddings_df.columns if col.startswith('dim_')]
    embedding_size_disease = len(disease_emb_cols)
    miRNA_emb_cols = [col for col in miRNA_embeddings_df.columns if col.startswith('dim_')]
    embedding_size_miRNA = len(miRNA_emb_cols)

    disease_emb = disease_emb[disease_emb_cols].to_numpy()
    miRNA_emb = miRNA_emb[miRNA_emb_cols].to_numpy()

    diseases = list(valid_diseases)
    miRNAs = list(valid_miRNAs)

    print(f"Number of diseases with embeddings: {len(diseases)}")
    print(f"Number of miRNAs with embeddings: {len(miRNAs)}")

    print("Loading positive disease-miRNA pairs...")
    disease_miRNA_df = pd.read_csv(disease_miRNA_file)

    positive_pairs = []
    total_pairs = len(disease_miRNA_df)
    for _, row in disease_miRNA_df.iterrows():
        disease, miRNA = row['disease'], row['miRNA']
        if disease in valid_diseases and miRNA in valid_miRNAs:
            positive_pairs.append((disease, miRNA))

    positive_pairs = set(positive_pairs)
    skipped_pairs = total_pairs - len(positive_pairs)
    print(f"Number of positive pairs loaded: {len(positive_pairs)}")
    print(f"Number of pairs skipped (missing embeddings): {skipped_pairs}")

    if not positive_pairs:
        raise ValueError("No valid positive pairs found after filtering. Check embeddings_file and disease_miRNA_file.")

    return diseases, miRNAs, disease_emb, miRNA_emb, positive_pairs, embedding_size_disease, embedding_size_miRNA

# Step 2: Generate feature vectors and labels with balanced negative sampling
def generate_features_labels(diseases, miRNAs, disease_emb, miRNA_emb, positive_pairs, embedding_size_disease, embedding_size_miRNA):
    print("Generating feature vectors and labels with balanced negative sampling...")
    features = []
    labels = []

    disease_idx = {disease: i for i, disease in enumerate(diseases)}
    miRNA_idx = {miRNA: i for i, miRNA in enumerate(miRNAs)}

    positive_pairs_list = list(positive_pairs)
    num_positive = len(positive_pairs_list)
    print(f"Number of positive pairs: {num_positive}")

    all_pairs = itertools.product(diseases, miRNAs)
    negative_pairs = [(d, dis) for d, dis in all_pairs if (d, dis) not in positive_pairs]
    print(f"Total negative pairs available: {len(negative_pairs)}")

    if len(negative_pairs) < num_positive:
        print("Warning: Fewer negative pairs than positive pairs. Using all available negative pairs.")
        selected_negative_pairs = negative_pairs
    else:
        selected_negative_pairs = random.sample(negative_pairs, num_positive)

    selected_pairs = positive_pairs_list + selected_negative_pairs
    random.shuffle(selected_pairs)

    for disease, miRNA in tqdm(selected_pairs, desc="Generating features for training pairs"):
        label = 1 if (disease, miRNA) in positive_pairs else 0
        labels.append(label)

        disease_vec = disease_emb[disease_idx[disease]]
        miRNA_vec = miRNA_emb[miRNA_idx[miRNA]]
        feature_vec = np.concatenate([disease_vec, miRNA_vec])
        features.append(feature_vec)

    return np.array(features), np.array(labels)

# Step 3: Train MLP model and predict novel associations (memory safe)
def train_and_predict(diseases, miRNAs, disease_emb, miRNA_emb, positive_pairs, embedding_size_disease, embedding_size_miRNA,
                      disease_embeddings_file, miRNA_embeddings_file, base_name, k=100, batch_size=500_000):

    print("Training MLP model on all labeled data...")
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            learning_rate_init=1e-3,
            alpha=1e-4,   # regularization
            max_iter=500,
            early_stopping=True,
            random_state=42
        ))
    ])

    # Generate training data
    features, labels = generate_features_labels(diseases, miRNAs, disease_emb, miRNA_emb,
                                                positive_pairs, embedding_size_disease, embedding_size_miRNA)

    # Train model
    mlp.fit(features, labels)

    # Evaluate model on training data
    y_pred_proba = mlp.predict_proba(features)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auroc = roc_auc_score(labels, y_pred_proba)
    auprc = average_precision_score(labels, y_pred_proba)
    f1 = f1_score(labels, y_pred)
    accuracy = accuracy_score(labels, y_pred)

    if auroc > 0.999:
        print("Warning: AUROC=1.0000 or very high, indicating potential overfitting or data leakage.")

    print("\nTraining Data Performance:")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    metrics_df = pd.DataFrame([{
        'disease_emb_file': disease_embeddings_file,
        'miRNA_emb_file': miRNA_embeddings_file,
        'auroc_mean': auroc,
        'auroc_std': 0.0,
        'auprc_mean': auprc,
        'auprc_std': 0.0,
        'f1_mean': f1,
        'f1_std': 0.0,
        'accuracy_mean': accuracy,
        'accuracy_std': 0.0
    }])
    metrics_csv = f'../Prediction/{base_name}_top_{k}_model_metrics.csv'
    metrics_df.to_csv(metrics_csv, index=False, float_format="%.4f")
    print(f"Saved model metrics to {metrics_csv}")

    # --- Predict for ALL novel pairs in batches ---
    print("Generating predictions for novel disease-miRNA associations (batched)...")
    disease_idx = {disease: i for i, disease in enumerate(diseases)}
    miRNA_idx = {miRNA: i for i, miRNA in enumerate(miRNAs)}

    all_pairs_iter = itertools.product(diseases, miRNAs)
    novel_pairs_iter = (pair for pair in all_pairs_iter if pair not in positive_pairs)

    top_k_heap = []  # min-heap for top k
    batch_pairs = []
    batch_features = []

    total_novel = len(diseases) * len(miRNAs) - len(positive_pairs)
    for pair in tqdm(novel_pairs_iter, total=total_novel, desc="Processing batches"):
        disease, miRNA = pair
        disease_vec = disease_emb[disease_idx[disease]]
        miRNA_vec = miRNA_emb[miRNA_idx[miRNA]]
        batch_features.append(np.concatenate([disease_vec, miRNA_vec]))
        batch_pairs.append(pair)

        if len(batch_features) >= batch_size:
            probs = mlp.predict_proba(np.array(batch_features))[:, 1]
            for p, prob in zip(batch_pairs, probs):
                if len(top_k_heap) < k:
                    heapq.heappush(top_k_heap, (prob, p))
                else:
                    heapq.heappushpop(top_k_heap, (prob, p))
            batch_pairs.clear()
            batch_features.clear()

    # Process last batch
    if batch_features:
        probs = mlp.predict_proba(np.array(batch_features))[:, 1]
        for p, prob in zip(batch_pairs, probs):
            if len(top_k_heap) < k:
                heapq.heappush(top_k_heap, (prob, p))
            else:
                heapq.heappushpop(top_k_heap, (prob, p))

    # Sort results
    top_k_heap.sort(reverse=True, key=lambda x: x[0])
    top_k_probs, top_k_pairs = zip(*top_k_heap)

    predictions_df = pd.DataFrame({
        'disease': [pair[0] for pair in top_k_pairs],
        'miRNA': [pair[1] for pair in top_k_pairs],
        'predicted_probability': top_k_probs
    })
    predictions_csv = f'../Prediction/{base_name}_top_{k}_predictions.csv'
    predictions_df.to_csv(predictions_csv, index=False, float_format="%.4f")
    print(f"Saved top {k} predictions to {predictions_csv}")

    return auroc, 0.0, auprc, 0.0, f1, 0.0, accuracy, 0.0

# Main function
def main():
    miRNA_net = "miRNANetWSB"
    emb_method = "gat"
    emb_size = 512
    epoch = 100

    disease_embeddings_file = f"../Results/DiseaseSimNet_OMIM_{emb_method}_d_{emb_size}_e_{epoch}.csv"
    miRNA_emb_file = f"{miRNA_net}_{emb_method}_d_{emb_size}_e_{epoch}"
    miRNA_embeddings_file = f"../Results/{miRNA_emb_file}.csv"
    disease_miRNA_file = os.path.expanduser("../Data/Phenotype2miRNAs_HMDD.csv")

    base_name_disease = os.path.splitext(os.path.basename(disease_embeddings_file))[0]
    base_name_miRNA = os.path.splitext(os.path.basename(miRNA_embeddings_file))[0]
    base_name = base_name_disease + "_" + base_name_miRNA + "_Balanced_MLP"

    output_file = f'../Prediction/{base_name}_top_{k}_output.txt'
    tee = Tee(output_file)
    sys.stdout = tee

    try:
        diseases, miRNAs, disease_emb, miRNA_emb, positive_pairs, disease_emb_size, miRNA_emb_size = load_data(
            disease_embeddings_file, miRNA_embeddings_file, disease_miRNA_file
        )

        train_and_predict(
            diseases, miRNAs, disease_emb, miRNA_emb, positive_pairs, disease_emb_size, miRNA_emb_size,
            disease_embeddings_file, miRNA_embeddings_file, base_name, k=10000, batch_size=500_000
        )

    except Exception as e:
        print(f"Error processing: {str(e)}")

    finally:
        sys.stdout = tee.stdout
        tee.close()

if __name__ == "__main__":
    main()
