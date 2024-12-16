
from squadds.database.HuggingFace import login_to_huggingface
from datasets import Dataset, load_dataset, concatenate_datasets

def contribute_experimental_data(data: dict, repo_id: str, dataset_name: str):
    """
    Contribute experimental data to SQuADDS_DB.

    Args:
        data (dict): Experimental data to be contributed.
        repo_id (str): The Hugging Face repo ID where the dataset is stored.
        dataset_name (str): Name of the dataset within the repo.

    Returns:
        None
    """
    # Validate data
    # ...existing code...
    
    # Format data as Hugging Face Dataset
    new_entry = Dataset.from_dict(data)
    
    # Load existing dataset
    dataset = load_dataset(repo_id, name=dataset_name)
    
    # Append new entry to dataset
    updated_dataset = concatenate_datasets([dataset, new_entry])
    
    # Push updated dataset to Hugging Face
    login_to_huggingface()
    updated_dataset.push_to_hub(repo_id, private=True)
    print(f"Experimental data contributed to {repo_id}/{dataset_name}.")