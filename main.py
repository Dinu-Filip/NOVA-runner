from datasets import load_dataset
ds = load_dataset("parquet", data_files=f"hf://datasets/c-i-ber/Nova/data/nova-v1.parquet", split="train")

if __name__ == "__main__":
    print(ds.features)
