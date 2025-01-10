# MMSG-DTA: A Multimodal, Multiscale Model Based on Sequence and Graph Modalities for Drug-Target Affinity Prediction
MMSG-DTA combines graph neural networks with Transformers to effectively capture both local node-level features and global structural features of molecular graphs. Additionally, a graph-based modality is employed to improve the extraction of protein features from amino acid sequences. To further enhance the model's performance, an attention-based feature fusion module is incorporated to integrate diverse feature types, thereby strengthening its representation capacity and robustness.

![Overview of ours model](./figures/overview.png)

## License
This project is licensed under the MIT - see the [LICENSE](LICENSE) file for details.
## Dataset
All data used in this paper are publicly available can be accessed here:  
- Davis and KIBA: https://github.com/hkmztrk/DeepDTA/tree/master/data  
- Metz:  https://github.com/simonfqy/PADME/tree/master/metzdata

## Requirements 
pip install -r requirements.txt

## Data Preparation
The model requires data files in the following format:
data.csv: The dataset should include the drug ID, SMILES representation, as well as the target ID and its corresponding amino acid sequence.
Example format of the CSV file:

| drug_key | compound_iso_smiles                                              | target_key | target_sequence  | affinity |
|----------|------------------------------------------------------------------|------------|--------------------------|----------|
| 11338033 | O=C(NC1CCNCC1)c1[nH]ncc1NC(=O)c1c(Cl)cccc1Cl                     | FLT1       | MVSYWDTGVLLCALLSCLLLTG...STPPI | 5.0 |
| 447077   | CSc1cccc(Nc2ncc3cc(-c4c(Cl)cccc4Cl)c(=O)n(C)c3n2)c1              | TRKA       | MLRGGRRGQLGWHSWAAGPGSLLAWL...QAPPVYLDVLG | 5.0 |

## Step-by-step running:
### Train/test MMSG-DTA
- First, run graph_prepare.py using
  ```markdown
  python graph_prepare.py --dataset dataset
  ```     
    Running graph_prepare.py create the contact map of the protein.
    If you prefer not to generate the contact map for the graph yourself, you can find it available for download at [https://zenodo.org/records/14441791](https://zenodo.org/records/14441791).
  - --dataset: davis/kiba/metz
  
- Second, run train.py using
  ```markdown
  python train.py --dataset davis --save_model
  ```
    to train MMSG-DTA.
  Explanation of parameters

  - --dataset: davis/kiba/metz
  - --save_model: whether save model or not
  - --lr: learning rate, default =  5e-4
  - --batch_size: default = 512

- To test a trained model please run test.py using
```markdown
python test.py --dataset dataset --model_path model_path
```
  - --dataset: davis/kiba/metz
  - --model_path: Path to store the weight files
### Drug/Target/All split
- First, Use the Cold_data_split.py in the Code folder to split dataset in cold setting
    The name of the dataset can be set to "davis" or "kiba" by "dataset_name".
    

    ```markdown
    python split.py --dataset dataset --SEED seed
    ```
  - --dataset: davis or kiba
  - --SEED: The random seed can be set by "SEED"
    Then you will get the training, validation and test data sets of the three cold start settings corresponding to the data set.
