"""Script to load a locally saved RobertaTokenizerFast tokenizer and display its details."""

from transformers import RobertaTokenizerFast
import os

def main():
    """
    Loads a RobertaTokenizerFast from a local directory and prints diagnostic information.
    """
    # Path to your local 'model' directory
    model_dir = "model"

    # Print the current working directory
    print("Current working directory:", os.getcwd())

    # Check if the model directory exists
    print("Model dir exists?", os.path.exists(model_dir))

    # List the contents of the model directory if it exists
    if os.path.exists(model_dir):
        print("Model dir content:", os.listdir(model_dir))
    else:
        print("Model dir content: NOT FOUND")

    # Load the tokenizer from the local directory
    tok = RobertaTokenizerFast.from_pretrained(model_dir)
    print("Tokenizer loaded successfully.")
    print(tok)


if __name__ == "__main__":
    main()
