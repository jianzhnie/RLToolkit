import pickle5 as pickle
from huggingface_hub import hf_hub_download


def load_from_hub(repo_id: str, filename: str) -> str:
    """Download a model from Hugging Face Hub.

    :param repo_id: id of the model repository from the Hugging Face Hub
    :param filename: name of the model zip file from the repository
    """

    # Get the model from the Hub, download and cache the model on your local disk
    pickle_model = hf_hub_download(repo_id=repo_id, filename=filename)

    with open(pickle_model, 'rb') as f:
        downloaded_model_file = pickle.load(f)

    return downloaded_model_file


if __name__ == '__main__':
    model = load_from_hub(
        repo_id='unfinity/q-Taxi-v3', filename='q-learning.pkl')
    print(model)
