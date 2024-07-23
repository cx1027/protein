Data
Accompanying data - Papyrus - A large scale curated dataset aimed at bioactivity predictions
https://zenodo.org/records/7019874
05.5_combined_set_protein_targets.tsv.xz
Classification:Enzyme->1, others->0

Task
classification(binary)

Accuracy
1. RF protein
Data:
./rf/protein/protein_classification.csv
Steps:
Follow steps in /rf/readme.txt
Result:
Accuracy: 0.9919678714859438
2. DeepWide protein
Data:
./dl/DeepWide/protein_code/protein_one_hot.csv
Steps:
Follow steps in /dl/DeepWide/readme.txt
/dl/DeepWide/protein_code
Result:
Wide 0.9919678568840027