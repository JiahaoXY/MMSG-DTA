import torch
import esm
import math
import numpy as np

import json, pickle
from collections import OrderedDict
import os
from tqdm import tqdm

# Function to construct protein contact maps using ESM-2 model
def protein_graph_construct(proteins, save_dir):
    """
    Construct protein contact maps from protein sequences using the ESM-2 model.

    Parameters:
        proteins (dict): A dictionary where keys are protein IDs and values are protein sequences.
        save_dir (str): Directory to save the computed contact maps in .npy format.

    This function loads the pretrained ESM-2 model, processes the protein sequences,
    generates contact maps, and saves the results.
    """
    # Load the pretrained ESM-2 model and batch converter
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Set the model to evaluation mode

    # Initialize an empty dictionary to store the contact maps
    target_graph = {}
    count = 0

    # Create a list of protein keys for iteration
    key_list = list(proteins.keys())

    # Iterate over each protein in the dataset
    for k_i in tqdm(range(len(key_list))):
        key = key_list[k_i]
        data = []
        pro_id = key  # Protein ID
        seq = proteins[key]  # Protein sequence

        # Only process proteins with sequence length <= 1200
        if len(seq) <= 1200:
            data.append((pro_id, seq))  # Prepare data for batch conversion
            batch_labels, batch_strs, batch_tokens = batch_converter(data)  # Convert to model input format
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)  # Get model output
            contact_map = results["contacts"][0].numpy()  # Extract contact map from results
            target_graph[pro_id] = contact_map  # Store the contact map in the dictionary
        else:
            # For longer sequences, split them into subsequences and process each subsequence
            contact_prob_map = np.zeros((len(seq), len(seq)))  # Initialize a global contact map (zeros)
            interval = 500  # Define the interval for subsequence length
            i = math.ceil(len(seq) / interval)  # Calculate the number of subsequences
            
            # Iterate through subsequences of the protein sequence
            for s in range(i):
                start = s * interval  # Starting index for subsequence
                end = min((s + 2) * interval, len(seq))  # Ending index (ensure no out-of-bound error)
                sub_seq_len = end - start  # Length of the subsequence
                
                # Process the subsequence
                temp_seq = seq[start:end]
                temp_data = [(pro_id, temp_seq)]  # Prepare subsequence data
                batch_labels, batch_strs, batch_tokens = batch_converter(temp_data)  # Convert subsequence
                
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)  # Get contacts for subsequence
                
                # Insert the predicted contact map into the global contact map
                row, col = np.where(contact_prob_map[start:end, start:end] != 0)  # Get indices of non-zero values
                row = row + start  # Adjust row indices
                col = col + start  # Adjust column indices
                contact_prob_map[start:end, start:end] += results["contacts"][0].numpy()  # Add contact map values
                contact_prob_map[row, col] /= 2.0  # Average overlapping regions

                if end == len(seq):  # Break the loop once the whole sequence is processed
                    break

            target_graph[pro_id] = contact_prob_map  # Store the final contact map

        # Save the contact map as a .npy file
        np.save(save_dir + pro_id + '.npy', target_graph[pro_id])
        count += 1  # Increment the counter

# Main script for loading data and calling the protein graph construction function
if __name__ == '__main__':
    def save_obj(obj, name):
        """
        Save an object to a file using pickle.

        Parameters:
            obj: The object to be saved.
            name (str): The name of the file (without extension) to save the object.
        """
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)  # Save using highest protocol

    def load_obj(name):
        """
        Load an object from a pickle file.

        Parameters:
            name (str): The name of the file to load the object from (without extension).
        
        Returns:
            The loaded object.
        """
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)  # Load the object from the pickle file

    # Define the dataset to process
    dataset = 'EGFR'
    
    if dataset in ['kiba', 'davis', 'Metz', 'EGFR']:
        # Load the protein data from a JSON file
        proteins = json.load(open('dataset/' + dataset + '/proteins.txt'))  # Read protein sequences

    # Print dataset information
    print('Dataset:', dataset)
    print('Number of proteins:', len(proteins))

    # Define the directory to save the contact maps
    save_dir = 'dataset/' + dataset + '/pconsc4/'
    
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Call the function to construct the protein graphs (contact maps)
    protein_graph_construct(proteins, save_dir)
