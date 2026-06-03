import os
import argparse
import io
import zipfile
import glob
import pandas as pd
from PIL import Image
import numpy as np
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, Features, ClassLabel, Image as HFImage
from huggingface_hub import hf_hub_download

def fetch_julienlucas(num_samples, seed=42):
    """
    Load julienlucas, dynamically pick num_samples (balanced 50/50 real/ai).
    """
    print(f"\n📡 Streaming julienlucas dataset to fetch {num_samples} balanced samples...")
    stream = load_dataset("julienlucas/midjourney-dalle-sd-nanobananapro-dataset", split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=5000)
    
    target_per_class = num_samples // 2
    
    def gen():
        real_count = 0
        ai_count = 0
        for ex in stream:
            if real_count >= target_per_class and ai_count >= target_per_class:
                break
                
            img = ex["image"]
            if not isinstance(img, Image.Image):
                if isinstance(img, bytes):
                    img = Image.open(io.BytesIO(img)).convert("RGB")
            else:
                img = img.convert("RGB")
                
            label = ex["label"]
            
            if label == 0 and real_count < target_per_class:
                real_count += 1
                yield {"image": img, "label": label}
                total = real_count + ai_count
                if total % 1000 == 0:
                    print(f"   ... fetched {total}/{num_samples} samples")
            elif label == 1 and ai_count < target_per_class:
                ai_count += 1
                yield {"image": img, "label": label}
                total = real_count + ai_count
                if total % 1000 == 0:
                    print(f"   ... fetched {total}/{num_samples} samples")
                
        print(f"✅ Fetched {real_count} real and {ai_count} AI julienlucas samples.")

    features = Features({
        "image": HFImage(),
        "label": ClassLabel(names=["real", "ai"])
    })
    
    ds = Dataset.from_generator(gen, features=features)
    ds = ds.shuffle(seed=seed)
    return ds

def fetch_ntire(num_samples, temp_dir, seed=42):
    """
    Downloads shard_0.zip from NTIRE-RobustAIGenDetection-train, extracts it,
    reads the CSV for labels, and loads num_samples (balanced).
    """
    print(f"\n📡 Downloading NTIRE shard_0.zip (this may take a while)...")
    os.makedirs(temp_dir, exist_ok=True)
    
    zip_path = hf_hub_download(
        repo_id="deepfakesMSU/NTIRE-RobustAIGenDetection-train",
        filename="shard_0.zip",
        repo_type="dataset",
        local_dir=temp_dir
    )
    
    extract_dir = os.path.join(temp_dir, "ntire_shard_0")
    if not os.path.exists(extract_dir):
        print(f"📦 Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
    # Find the CSV file
    csv_files = glob.glob(os.path.join(extract_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError("Could not find label CSV file in NTIRE shard_0.")
    csv_path = csv_files[0]
    print(f"📄 Found label CSV: {csv_path}")
    
    df = pd.read_csv(csv_path)
    # The CSV columns might be something like 'filename' or 'image' and 'label'
    # We try to heuristically find the image path column and label column
    img_col = next((c for c in df.columns if 'image' in c.lower() or 'file' in c.lower() or 'path' in c.lower()), df.columns[0])
    lbl_col = next((c for c in df.columns if 'label' in c.lower() or 'class' in c.lower() or 'target' in c.lower()), df.columns[-1])
    
    # Standardize labels: assuming 'real' or 0 is real, 'ai'/'fake' or 1 is AI.
    def parse_label(val):
        val_str = str(val).lower()
        if val_str in ['0', 'real', 'human', 'authentic']:
            return 0
        return 1

    df['standard_label'] = df[lbl_col].apply(parse_label)
    
    # Shuffle dataframe
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    target_per_class = num_samples // 2
    real_count = 0
    ai_count = 0
    
    def gen():
        nonlocal real_count, ai_count
        for _, row in df.iterrows():
            if real_count >= target_per_class and ai_count >= target_per_class:
                break
                
            img_name = str(row[img_col])
            # Find the actual image file path
            img_paths = glob.glob(os.path.join(extract_dir, "**", img_name), recursive=True)
            if not img_paths:
                continue
            
            try:
                img = Image.open(img_paths[0]).convert("RGB")
            except Exception:
                continue
                
            label = row['standard_label']
            
            if label == 0 and real_count < target_per_class:
                real_count += 1
                yield {"image": img, "label": label}
                total = real_count + ai_count
                if total % 1000 == 0:
                    print(f"   ... fetched {total}/{num_samples} samples")
            elif label == 1 and ai_count < target_per_class:
                ai_count += 1
                yield {"image": img, "label": label}
                total = real_count + ai_count
                if total % 1000 == 0:
                    print(f"   ... fetched {total}/{num_samples} samples")
                
        print(f"✅ Fetched {real_count} real and {ai_count} AI NTIRE samples.")

    features = Features({
        "image": HFImage(),
        "label": ClassLabel(names=["real", "ai"])
    })
    
    ds = Dataset.from_generator(gen, features=features)
    ds = ds.shuffle(seed=seed)
    return ds

def fetch_deepfakejudge(num_samples, seed=42, hf_token=None):
    """
    Stream MBZUAI/DeepfakeJudge-Dataset and dynamically pick num_samples (balanced).
    Requires a valid Hugging Face token with access to gated repos.
    """
    print(f"\n📡 Streaming MBZUAI/DeepfakeJudge-Dataset to fetch {num_samples} balanced samples...")
    if not hf_token and not os.environ.get("HF_TOKEN"):
        print("⚠️ Warning: No HF_TOKEN provided. Falling back to huggingface-cli cached token. This gated dataset might fail to load if not logged in.")
        hf_token = True # Instruct load_dataset to use the cached token
        
    # If hf_token is None, fallback to environ
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN", True)
        
    stream = load_dataset("MBZUAI/DeepfakeJudge-Dataset", split="train", streaming=True, token=hf_token)
    stream = stream.shuffle(seed=seed, buffer_size=5000)
    
    target_per_class = num_samples // 2
    
    def gen():
        real_count = 0
        ai_count = 0
        for ex in stream:
            if real_count >= target_per_class and ai_count >= target_per_class:
                break
                
            img = ex.get("image")
            if img is None:
                continue
                
            if not isinstance(img, Image.Image):
                if isinstance(img, bytes):
                    img = Image.open(io.BytesIO(img)).convert("RGB")
            else:
                img = img.convert("RGB")
                
            # Attempt to determine label from common keys
            # Often it's 'label', 'is_fake', 'class'
            label_val = None
            for key in ["label", "is_fake", "class", "target"]:
                if key in ex:
                    label_val = ex[key]
                    break
            
            if label_val is None:
                # Fallback: check file path if available
                path = str(ex.get("image_path", "")).lower()
                if "real" in path or "authentic" in path:
                    label_val = 0
                elif "fake" in path or "ai" in path or "generated" in path:
                    label_val = 1
                else:
                    label_val = 1 # Assume fake if we can't tell, though risky
            
            # Standardize
            val_str = str(label_val).lower()
            if val_str in ['0', 'real', 'human', 'authentic', 'false']:
                label = 0
            else:
                label = 1
            
            if label == 0 and real_count < target_per_class:
                real_count += 1
                yield {"image": img, "label": label}
                total = real_count + ai_count
                if total % 1000 == 0:
                    print(f"   ... fetched {total}/{num_samples} samples")
            elif label == 1 and ai_count < target_per_class:
                ai_count += 1
                yield {"image": img, "label": label}
                total = real_count + ai_count
                if total % 1000 == 0:
                    print(f"   ... fetched {total}/{num_samples} samples")
                
        print(f"✅ Fetched {real_count} real and {ai_count} AI DeepfakeJudge samples.")

    features = Features({
        "image": HFImage(),
        "label": ClassLabel(names=["real", "ai"])
    })
    
    ds = Dataset.from_generator(gen, features=features)
    ds = ds.shuffle(seed=seed)
    return ds

def split_dataset_balanced(ds, seed=42):
    """
    Splits the dataset into 80/10/10 train/val/test while keeping the 50/50 real/ai balance.
    """
    print("\n⚖️ Splitting dataset 80/10/10 with balanced classes...")
    real_ds = ds.filter(lambda x: x["label"] == 0)
    ai_ds = ds.filter(lambda x: x["label"] == 1)
    
    def split_single_class(class_ds):
        train_test = class_ds.train_test_split(test_size=0.2, seed=seed)
        train_ds = train_test["train"]
        
        val_test = train_test["test"].train_test_split(test_size=0.5, seed=seed)
        val_ds = val_test["train"]
        test_ds = val_test["test"]
        
        return train_ds, val_ds, test_ds

    real_train, real_val, real_test = split_single_class(real_ds)
    ai_train, ai_val, ai_test = split_single_class(ai_ds)
    
    train_ds = concatenate_datasets([real_train, ai_train]).shuffle(seed=seed)
    val_ds = concatenate_datasets([real_val, ai_val]).shuffle(seed=seed)
    test_ds = concatenate_datasets([real_test, ai_test]).shuffle(seed=seed)
    
    return DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds
    })

def build_datasets(num_julienlucas=10000, num_ntire=15000, num_deepfakejudge=10000, out_dir="../../data/visual", hf_token=None):
    from datasets import load_from_disk
    
    # Paths
    julienlucas_dir = os.path.join(out_dir, "julienlucas")
    ntire_dir = os.path.join(out_dir, "ntire")
    deepfakejudge_dir = os.path.join(out_dir, "deepfakejudge")
    combined_dir = os.path.join(out_dir, "combined_dataset")
    temp_dir = os.path.join(out_dir, "temp_downloads")
    
    os.makedirs(julienlucas_dir, exist_ok=True)
    os.makedirs(ntire_dir, exist_ok=True)
    os.makedirs(deepfakejudge_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    # 1. Julienlucas
    if os.path.exists(os.path.join(julienlucas_dir, "dataset_info.json")):
        print(f"📦 Loading julienlucas subset from {julienlucas_dir} (already exists)")
        julienlucas_ds = load_from_disk(julienlucas_dir)
    else:
        julienlucas_ds = fetch_julienlucas(num_julienlucas)
        print(f"💾 Saving julienlucas subset to {julienlucas_dir}")
        julienlucas_ds.save_to_disk(julienlucas_dir)

    # 2. NTIRE
    if os.path.exists(os.path.join(ntire_dir, "dataset_info.json")):
        print(f"📦 Loading NTIRE subset from {ntire_dir} (already exists)")
        ntire_ds = load_from_disk(ntire_dir)
    else:
        ntire_ds = fetch_ntire(num_ntire, temp_dir)
        print(f"💾 Saving NTIRE subset to {ntire_dir}")
        ntire_ds.save_to_disk(ntire_dir)
    
    # 3. DeepfakeJudge
    if os.path.exists(os.path.join(deepfakejudge_dir, "dataset_info.json")):
        print(f"📦 Loading DeepfakeJudge subset from {deepfakejudge_dir} (already exists)")
        dfj_ds = load_from_disk(deepfakejudge_dir)
    else:
        dfj_ds = fetch_deepfakejudge(num_deepfakejudge, hf_token=hf_token)
        print(f"💾 Saving DeepfakeJudge subset to {deepfakejudge_dir}")
        dfj_ds.save_to_disk(deepfakejudge_dir)

    # Combine
    print("\n🔗 Combining datasets into one big dataset...")
    big_dataset = concatenate_datasets([julienlucas_ds, ntire_ds, dfj_ds]).shuffle(seed=42)
    print(f"Total combined size: {len(big_dataset)} samples.")

    # Split 80/10/10 with 50/50 balance
    final_dataset_dict = split_dataset_balanced(big_dataset)
    
    print(f"\n💾 Saving final combined dataset to {combined_dir}")
    final_dataset_dict.save_to_disk(combined_dir)
    
    print("\n🎉 Done! Dataset ready.")
    print(f"Train: {len(final_dataset_dict['train'])}")
    print(f"Validation: {len(final_dataset_dict['validation'])}")
    print(f"Test: {len(final_dataset_dict['test'])}")
    
    return final_dataset_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build balanced combined visual dataset.")
    parser.add_argument("--num_julienlucas", type=int, default=10000, help="Number of julienlucas samples to fetch.")
    parser.add_argument("--num_ntire", type=int, default=15000, help="Number of NTIRE samples to fetch.")
    parser.add_argument("--num_deepfakejudge", type=int, default=10000, help="Number of DeepfakeJudge samples to fetch.")
    parser.add_argument("--out_dir", type=str, default="../../data/visual", help="Base output directory.")
    args = parser.parse_args()
    build_datasets(args.num_julienlucas, args.num_ntire, args.num_deepfakejudge, args.out_dir)
