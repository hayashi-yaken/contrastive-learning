"""Fetch WordNet (nltk) and the STS-B dataset cache up front."""
from hypsimcse.data.wordnet import ensure_wordnet


def main():
    ensure_wordnet()
    print("WordNet ready.")
    try:
        from datasets import load_dataset
        load_dataset("glue", "stsb", split="validation")
        print("STS-B ready.")
    except Exception as e:  # noqa: BLE001
        print(f"STS-B fetch skipped/failed: {e}")


if __name__ == "__main__":
    main()
