import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import networkx as nx
import os
import glob

emb_method = "gat"

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Step 1: Load the disease similarity network
def load_disease_network(disease_sim_file="../Data/DiseaseSimNet_OMIM.txt"):
    print(disease_sim_file)
    
    print("Loading disease similarity network...")
    disease_sim_df = pd.read_csv(disease_sim_file, sep='\t', header=None, names=['disease1', 'sim', 'disease2'])
    diseases = set(disease_sim_df['disease1']).union(set(disease_sim_df['disease2']))
    
    disease_G = nx.Graph()
    for disease in tqdm(diseases, desc="Adding disease nodes"):
        disease_G.add_node(disease, type="disease")
    
    for _, row in tqdm(disease_sim_df.iterrows(), total=len(disease_sim_df), desc="Adding disease-disease edges"):
        disease_G.add_edge(row['disease1'], row['disease2'], weight=float(row['sim']))
    
    return disease_G, list(diseases)

# Step 2: Convert networkx graph to PyTorch Geometric Data object
def graph_to_pyg_data(G, embedding_size=128):
    print(f"Graph has {len(G.nodes())} nodes and {len(G.edges())} edges")
    node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
    print("Node index mapping created")
    
    for node in G.nodes():
        if not isinstance(node, (str, int)):
            print(f"Invalid node ID: {node} (type: {type(node)})")
    
    edge_index = []
    edge_weight = []
    for u, v, data in G.edges(data=True):
        edge_index.append([node_to_idx[u], node_to_idx[v]])
        edge_index.append([node_to_idx[v], node_to_idx[u]])
        weight = data.get('weight', 1.0)
        if not isinstance(weight, (int, float)) or weight <= 0 or np.isnan(weight) or np.isinf(weight):
            print(f"Invalid weight found for edge ({u}, {v}): {weight}, using default 1.0")
            weight = 1.0
        edge_weight.append(weight)
        edge_weight.append(weight)
    print(f"Edge index and weights created: {len(edge_index)} edges")
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    print("Edge index tensor created")
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    print("Edge weight tensor created")
    
    num_nodes = len(G.nodes())
    x = torch.ones((num_nodes, embedding_size), dtype=torch.float)
    print("Node features tensor created")
    
    data = Data(x=x, edge_index=edge_index)
    print("PyTorch Geometric Data object created without edge weights")
    data.edge_weight = edge_weight
    print("Edge weights added to Data object")
    return data, node_to_idx

# Step 3: Define a custom WeightedGATConv layer that supports edge weights
class WeightedGATConv(GATConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0, edge_dim=1, **kwargs):
        super(WeightedGATConv, self).__init__(in_channels, out_channels, heads=heads, concat=concat, dropout=dropout, edge_dim=edge_dim, **kwargs)
    
    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None:
            edge_attr = edge_weight.view(-1, 1)
        else:
            edge_attr = None
        return super().forward(x, edge_index, edge_attr=edge_attr)

# Step 4: Define GAT model with WeightedGATConv
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.5):
        super(GAT, self).__init__()
        self.conv1 = WeightedGATConv(input_dim, hidden_dim, heads=heads, concat=True, dropout=dropout, edge_dim=1)
        self.conv2 = WeightedGATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout, edge_dim=1)
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.elu(x)  # ELU activation as commonly used with GAT
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

# Step 5: Train GAT model with unsupervised loss (graph reconstruction)
def train_gat(data, embedding_size=128, hidden_dim=64, epochs=50, lr=0.01, node_type="nodes"):
    device = torch.device('cpu')
    data = data.to(device)
    
    model = GAT(input_dim=data.x.shape[1], hidden_dim=hidden_dim, output_dim=embedding_size, heads=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in tqdm(range(epochs), desc=f"Training GAT for {node_type}", ascii=False, smoothing=0.1):
        optimizer.zero_grad()
        embeddings = model(data)
        
        u, v = data.edge_index[0], data.edge_index[1]
        edge_scores = torch.sum(embeddings[u] * embeddings[v], dim=-1)
        pos_loss = -torch.log(torch.sigmoid(edge_scores) + 1e-15).mean()
        
        num_neg_samples = data.edge_index.shape[1]
        neg_u = torch.randint(0, data.num_nodes, (num_neg_samples,), device=device)
        neg_v = torch.randint(0, data.num_nodes, (num_neg_samples,), device=device)
        neg_mask = torch.ones(num_neg_samples, dtype=torch.bool, device=device)
        for i in range(num_neg_samples):
            u, v = neg_u[i], neg_v[i]
            edge_exists = (data.edge_index[0].eq(u) & data.edge_index[1].eq(v)).any().item()
            if edge_exists:
                neg_mask[i] = 0
        neg_u, neg_v = neg_u[neg_mask.bool()], neg_v[neg_mask.bool()]
        
        loss = pos_loss
        if len(neg_u) > 0:
            neg_scores = torch.sum(embeddings[neg_u] * embeddings[neg_v], dim=-1)
            neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-15).mean()
            loss = loss + neg_loss
        else:
            print(f"Epoch Rhein{epoch+1}: No negative samples after filtering, skipping neg_loss")
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        embeddings = model(data).cpu().numpy()
    return embeddings

# Step 6: Extract and process embeddings
def extract_embeddings(embeddings, nodes, node_to_idx, node_type):
    embeddings_dict = {}
    for node in tqdm(nodes, desc=f"Extracting embeddings for {node_type}"):
        idx = node_to_idx[node]
        embeddings_dict[node] = embeddings[idx]
    return embeddings_dict

# Step 7: Save embeddings
def save_embeddings_disease(disease_embeddings, disease_G, output_file):
    print("Saving embeddings...")
    data = []
    
    for node, emb in disease_embeddings.items():
        node_type = disease_G.nodes[node]["type"]
        row = [node, node_type] + emb.tolist()
        data.append(row)
    
    embedding_size = len(next(iter(disease_embeddings.values())))
    columns = ['node_id', 'type'] + [f'dim_{i+1}' for i in range(embedding_size)]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_file, index=False)
    print(f"Disease Embeddings saved to {output_file}")

def save_embeddings_disease(disease_embeddings, disease_G, output_file):
    print("Saving embeddings...")
    data = []
    
    for node, emb in disease_embeddings.items():
        node_type = disease_G.nodes[node]["type"]
        row = [node, node_type] + emb.tolist()
        data.append(row)
    
    embedding_size = len(next(iter(disease_embeddings.values())))
    columns = ['node_id', 'type'] + [f'dim_{i+1}' for i in range(embedding_size)]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_file, index=False)
    print(f"Disease Embeddings saved to {output_file}")

# Main function
def main(embedding_size=128, epochs=50, disease_sim_file="../Data/MeSHID_Net.txt"):
    # Load disease network (only once)
    disease_base_name = os.path.splitext(os.path.basename(disease_sim_file))[0]
    disease_output_file = f"../Results/{disease_base_name}_{emb_method}_d_{embedding_size}_e_{epochs}.csv" 
    if os.path.exists(disease_output_file):
        print(f"File {disease_output_file} is existing!")
    else: 
        disease_G, diseases = load_disease_network(disease_sim_file)
        print("Converting disease graph to PyTorch Geometric format...")
        disease_data, disease_node_to_idx = graph_to_pyg_data(disease_G, embedding_size)
        print("Computing disease embeddings...")
        disease_embeddings = train_gat(disease_data, embedding_size=embedding_size, epochs=epochs, node_type="diseases")
        disease_embeddings_dict = extract_embeddings(disease_embeddings, diseases, disease_node_to_idx, "diseases")
        # Save disease embeddings
        save_embeddings_disease(disease_embeddings_dict, disease_G, disease_output_file)

        # Print sample embeddings
        print(f"\nSample disease embeddings for {disease_output_file}:")
        for node, emb in list(disease_embeddings_dict.items())[:5]:
            print(f"Disease Node: {node}, Embedding: {emb[:5]}...")
    
    
# Execute and get embeddings
if __name__ == "__main__":
    disease_sim_file = "../Data/DiseaseSimNet_OMIM.txt"
    print(f"disease_sim_file: {disease_sim_file}")

    embedding_size_list = [128, 256, 512]
    epochs_list = [100, 200, 400]
    for epochs in epochs_list:    
        for embedding_size in embedding_size_list:
            print(f"embedding_size: {embedding_size}, epochs: {epochs}")
            main(
                embedding_size=embedding_size,
                epochs=epochs,
                disease_sim_file=disease_sim_file,
            )