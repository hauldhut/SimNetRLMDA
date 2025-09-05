import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer

from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import os
import sys
import random

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


# Step 1: Load embeddings and disease-miRNA pairs
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


# Step 2: Generate feature vectors and labels
def generate_features_labels(diseases, miRNAs, disease_emb, miRNA_emb, positive_pairs, embedding_size_disease, embedding_size_miRNA):
    print("Generating feature vectors and labels with balanced negative sampling...")
    features = []
    labels = []
    pairs = []
    
    disease_idx = {disease: i for i, disease in enumerate(diseases)}
    miRNA_idx = {miRNA: i for i, miRNA in enumerate(miRNAs)}
    
    positive_pairs_list = list(positive_pairs)
    num_positive = len(positive_pairs_list)
    print(f"Number of positive pairs: {num_positive}")
    
    all_pairs = list(itertools.product(diseases, miRNAs))
    negative_pairs = [pair for pair in all_pairs if pair not in positive_pairs]
    print(f"Total negative pairs available: {len(negative_pairs)}")
    
    if len(negative_pairs) < num_positive:
        print("Warning: Fewer negative pairs than positive pairs. Using all available negative pairs.")
        selected_negative_pairs = negative_pairs
    else:
        selected_negative_pairs = random.sample(negative_pairs, num_positive)
    
    print(f"Number of negative pairs sampled: {len(selected_negative_pairs)}")
    
    selected_pairs = positive_pairs_list + selected_negative_pairs
    random.shuffle(selected_pairs)
    
    for disease, miRNA in tqdm(selected_pairs, desc="Generating features for pairs"):
        pair = (disease, miRNA)
        pairs.append(pair)
        label = 1 if pair in positive_pairs else 0
        labels.append(label)
        
        disease_vec = disease_emb[disease_idx[disease]]
        miRNA_vec = miRNA_emb[miRNA_idx[miRNA]]
        feature_vec = np.concatenate([disease_vec, miRNA_vec])
        features.append(feature_vec)
    
    return np.array(features), np.array(labels), pairs


# Step 3: Train and evaluate MLP with k-fold cross-validation
kfold = 5
def evaluate_model(features, labels, base_name):
    print("Training and evaluating MLP model...")
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

    
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
    
    auroc_scores = []
    auprc_scores = []
    f1_scores = []
    accuracy_scores = []
    
    roc_data = []
    pr_data = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    
    tprs = []
    precisions = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels), 1):
        print(f"Processing fold {fold}/{kfold}...")
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        mlp.fit(X_train, y_train)
        y_pred_proba = mlp.predict_proba(X_test)[:, 1]
        
        auroc = roc_auc_score(y_test, y_pred_proba)
        auprc = average_precision_score(y_test, y_pred_proba)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        auroc_scores.append(auroc)
        auprc_scores.append(auprc)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        roc_data.append(pd.DataFrame({'fold': fold, 'fpr': fpr, 'tpr': tpr}))
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        precision_interp = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions.append(precision_interp)
        pr_data.append(pd.DataFrame({'fold': fold, 'recall': recall, 'precision': precision}))
        
        print(f"Fold {fold} - AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
    
    auroc_mean, auroc_std = np.mean(auroc_scores), np.std(auroc_scores)
    auprc_mean, auprc_std = np.mean(auprc_scores), np.std(auprc_scores)
    f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)
    accuracy_mean, accuracy_std = np.mean(accuracy_scores), np.std(accuracy_scores)
    
    print("\nFinal Results:")
    print(f"AUROC: {auroc_mean:.4f} ± {auroc_std:.4f}")
    print(f"AUPRC: {auprc_mean:.4f} ± {auprc_std:.4f}")
    print(f"F1-score: {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Accuracy: {accuracy_mean:.4f} ± {accuracy_std:.4f}")
    
    return auroc_mean, auroc_std, auprc_mean, auprc_std, f1_mean, f1_std, accuracy_mean, accuracy_std


# === Main function kept as in your original ===
def main():
    disease_emb_files = []

    DiSimNet = "DiseaseSimNet_OMIM"
    
    disease_emb_files.append("DiseaseSimNet_OMIM_gat_d_128_e_100")
    disease_emb_files.append("DiseaseSimNet_OMIM_gat_d_256_e_100")
    disease_emb_files.append("DiseaseSimNet_OMIM_gat_d_512_e_100")

    disease_emb_files.append("DiseaseSimNet_OMIM_gat_d_128_e_200")
    disease_emb_files.append("DiseaseSimNet_OMIM_gat_d_256_e_200")
    disease_emb_files.append("DiseaseSimNet_OMIM_gat_d_512_e_200")

    disease_emb_files.append("DiseaseSimNet_OMIM_gat_d_128_e_400")
    disease_emb_files.append("DiseaseSimNet_OMIM_gat_d_256_e_400")
    disease_emb_files.append("DiseaseSimNet_OMIM_gat_d_512_e_400")

    
    # miRNA_nets = ["miRNANetW", "miRNANetS", "miRNANetB","miRNANetWS"]#miRNANetW/miRNANetS/miRNANetB/miRNANetWS/miRNANetWB/miRNANetSB/miRNANetWSB
    # miRNA_nets = ["miRNANetW", "miRNANetS", "miRNANetB", "miRNANetD"]
    miRNA_nets = ["miRNANetW"]
    # emb_methods = ["gat", "gtn", "mp2v"]
    emb_methods = ["gat"]
    miRNA_emb_files = []
    for miRNA_net in miRNA_nets:
        for emb_method in emb_methods:
            miRNA_emb_files.append(f"{miRNA_net}_{emb_method}_d_128_e_100")
            miRNA_emb_files.append(f"{miRNA_net}_{emb_method}_d_256_e_100")
            miRNA_emb_files.append(f"{miRNA_net}_{emb_method}_d_512_e_100")
            miRNA_emb_files.append(f"{miRNA_net}_{emb_method}_d_128_e_200")
            miRNA_emb_files.append(f"{miRNA_net}_{emb_method}_d_256_e_200")
            miRNA_emb_files.append(f"{miRNA_net}_{emb_method}_d_512_e_200")
            miRNA_emb_files.append(f"{miRNA_net}_{emb_method}_d_128_e_400")            
            miRNA_emb_files.append(f"{miRNA_net}_{emb_method}_d_256_e_400")
            miRNA_emb_files.append(f"{miRNA_net}_{emb_method}_d_512_e_400")

    results = []
    for disease_emb_file in disease_emb_files:
        for miRNA_emb_file in miRNA_emb_files:
            disease_embeddings_file = f"../Results/{disease_emb_file}.csv"
            miRNA_embeddings_file = f"../Results/{miRNA_emb_file}.csv"

            disease_miRNA_file = os.path.expanduser(f"../Data/Phenotype2miRNAs_HMDD.csv")
            
            base_name_disease = os.path.splitext(os.path.basename(disease_embeddings_file))[0]
            base_name_miRNA = os.path.splitext(os.path.basename(miRNA_embeddings_file))[0]
            
            base_name = base_name_disease + "_" + base_name_miRNA + "_Balanced_XGB"

            print(f"\nProcessing pair:")
            print(f"disease_embeddings_file: {disease_embeddings_file}")
            print(f"miRNA_embeddings_file: {miRNA_embeddings_file}")
            print(f"disease-miRNA file: {disease_miRNA_file}")
            
            output_file = f'../Results/Detail/{base_name}_output.txt'
            tee = Tee(output_file)
            sys.stdout = tee
            
            try:
                diseases, miRNAs, disease_emb, miRNA_emb, positive_pairs, disease_emb_size, miRNA_emb_size = load_data(
                    disease_embeddings_file, miRNA_embeddings_file, disease_miRNA_file
                )
                
                features, labels, pairs = generate_features_labels(
                    diseases, miRNAs, disease_emb, miRNA_emb, positive_pairs, disease_emb_size, miRNA_emb_size
                )
                
                auroc_mean, auroc_std, auprc_mean, auprc_std, f1_mean, f1_std, accuracy_mean, accuracy_std = evaluate_model(features, labels, base_name)
                
                # Collect results
                results.append({
                    'disease_emb_file': disease_emb_file,
                    'miRNA_emb_file': miRNA_emb_file,
                    'auroc_mean': auroc_mean,
                    'auroc_std': auroc_std,
                    'auprc_mean': auprc_mean,
                    'auprc_std': auprc_std,
                    'f1_mean': f1_mean,
                    'f1_std': f1_std,
                    'accuracy_mean': accuracy_mean,
                    'accuracy_std': accuracy_std
                })
            
            except Exception as e:
                print(f"Error processing: {str(e)}")
            
            finally:
                sys.stdout = tee.stdout
                tee.close()
            
    # Save summary results to a CSV file
    # emb_str = "_".join(emb_methods)
    summary_df = pd.DataFrame(results)
    summary_file = f"../Results/Performance_summary_metrics_MLP.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary results saved to {summary_file}")

# Execute
if __name__ == "__main__":
    main()