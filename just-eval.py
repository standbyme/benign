import pickle

from datasets import load_dataset

def main():
    ds = load_dataset("re-align/just-eval-instruct", "default")

    test_dataset = ds["test"]
    df = test_dataset.to_pandas()

    # get the row with category is regular
    regular = df[df["category"] == "regular"]

    # get the column instruction and save to a list
    instructions = regular["instruction"].tolist()

    # save the list to a pickle file
    with open("just-eval-instruct-regular.pkl", "wb") as f:
        pickle.dump(instructions, f)


if __name__ == "__main__":
    main()
