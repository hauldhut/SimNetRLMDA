# SimNetRLMDA: Similarity Network-based Representation Learning for the prediction of disease-miRNA associations

- SimNetRLMDA integrates multiple miRNA functional similarity networks, constructed from miRNA-target datasets (TargetScan, miRTarBase, miRWalk) independent of known disease-miRNA associations, with a MeSH-based disease similarity network, using graph attention networks (GATs) and a Multi-Layer Perceptron (MLP) to predict disease-miRNA associations.
- By constructing similarity networks independent of known disease-miRNA associations and learning disease and miRNA representations separately, SimNetRLMDA eliminates data leakage and enables predictions for diseases and miRNAs without prior associations, unlike existing methods reliant on such data. 

![SimNetRLMDA](https://github.com/hauldhut/SimNetRLMDA/blob/main/Figure1.png)

## Repo structure
- **Data**: Contains all data 
- **Code**: Contains all source code to reproduce all the results
- **Results**: To store simulation results

## How to run
- Download the repo
- Follow instructions (README.md) in the folder **Code** to run
  
