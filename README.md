## LSCALE: Latent Space Clustering-Based Active Learning for Node Classification
This repository is the PyTorch implementation of "LSCALE: Latent Space Clustering-Based Active Learning for Node Classification".

### Dependencies
Our implementation works with PyTorch>=1.0.0. Install other dependencies: 

`$ pip install -r requirements.txt`

### Data
Cora, Citeseer, Pubmed, Coauthor-CS, and Coauthor-Physics.

All the dataset can be found in the data folder.

To get the datasets, refer to the appendix in our paper draft. 
### Usage
run_baselines.py for baselines using GCN model.

LSCALE.py for our methods `LSCALE`.

To get all baselines' results (GCN as the base model), the budget size is set before.
```
sh run_all_GCN_10.sh
```

To get the results of our model (LSCALE),
```
sh run_LSCALE.sh 
```


If you find our implementation useful in your research, please consider citing our paper:
```bibtex
@inproceedings{LSCALE,
  title={LSCALE: Latent Space Clustering-Based Active Learning for Node Classification},
  author={Liu, Juncheng and Wang, Yiwei and Hooi, Bryan and Yang, Renchi and Xiao, Xiaokui},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  year={2022},
  organization={Springer}
}
```