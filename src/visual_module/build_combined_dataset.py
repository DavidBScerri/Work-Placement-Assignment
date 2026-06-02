import os
import argparse
import io
from PIL import Image
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets

def extract_genimage_label(example):
    """Derive binary label from GenImage image_path: 0 for real, 1 for AI."""
    path = (example.get("image_path") or "").lower()
    if "/nature/" in path or "/real/" in path:
        example["label"] = 0
    else:
        example["label"] = 1
    return example

def fetch_genimage(num_samples, seed=42):
    """
    Stream GenImage, dynamically pick num_samples (balanced 50/50 real/ai),
    and keep only 'image' and 'label' columns.
    """
    print(f"\n📡 Streaming GenImage to fetch {num_samples} balanced samples...")
    stream = load_dataset("nebula/GenImage-arrow", split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=5000)
    
    target_per_class = num_samples // 2
    
    def gen():
        real_count = 0
        ai_count = 0
        for ex in stream:
            if real_count >= target_per_class and ai_count >= target_per_class:
                break
                
            ex = extract_genimage_label(ex)
            label = ex["label"]
            
            img = ex["image"]
            if isinstance(img, bytes):
                img = Image.open(io.BytesIO(img)).convert("RGB")
                
            if label == 0 and real_count < target_per_class:
                real_count += 1
                yield {"image": img, "label": label}
            elif label == 1 and ai_count < target_per_class:
                ai_count += 1
                yield {"image": img, "label": label}
                
        print(f"✅ Fetched {real_count} real and {ai_count} AI GenImage samples.")

    from datasets import Features, ClassLabel, Image as HFImage
    features = Features({
        "image": HFImage(),
        "label": ClassLabel(names=["real", "ai"])
    })
    
    ds = Dataset.from_generator(gen, features=features)
    ds = ds.shuffle(seed=seed)
    return ds

def fetch_julienlucas(num_samples, seed=42):
    """
    Load julienlucas, dynamically pick num_samples (balanced 50/50 real/ai),
    and keep only 'image' and 'label' columns.
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
            elif label == 1 and ai_count < target_per_class:
                ai_count += 1
                yield {"image": img, "label": label}
                
        print(f"✅ Fetched {real_count} real and {ai_count} AI julienlucas samples.")

    from datasets import Features, ClassLabel, Image as HFImage
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
        # First split off 20% for val+test
        train_test = class_ds.train_test_split(test_size=0.2, seed=seed)
        train_ds = train_test["train"]
        
        # Then split that 20% into two 10% (i.e. half of 20%)
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

def main():
    parser = argparse.ArgumentParser(description="Build balanced combined visual dataset.")
    parser.add_argument("--num_genimage", type=int, default=1000, help="Number of GenImage samples to fetch.")
    parser.add_argument("--num_julienlucas", type=int, default=1000, help="Number of julienlucas samples to fetch.")
    parser.add_argument("--out_dir", type=str, default="../../data/visual", help="Base output directory.")
    args = parser.parse_args()

    # Paths
    genimage_dir = os.path.join(args.out_dir, "genimage")
    julienlucas_dir = os.path.join(args.out_dir, "julienlucas")
    combined_dir = os.path.join(args.out_dir, "combined_dataset")
    
    os.makedirs(genimage_dir, exist_ok=True)
    os.makedirs(julienlucas_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    # 1. GenImage
    genimage_ds = fetch_genimage(args.num_genimage)
    print(f"💾 Saving GenImage subset to {genimage_dir}")
    genimage_ds.save_to_disk(genimage_dir)

    # 2. Julienlucas
    julienlucas_ds = fetch_julienlucas(args.num_julienlucas)
    print(f"💾 Saving julienlucas subset to {julienlucas_dir}")
    julienlucas_ds.save_to_disk(julienlucas_dir)

    # 3. Combine
    print("\n🔗 Combining datasets into one big dataset...")
    big_dataset = concatenate_datasets([genimage_ds, julienlucas_ds]).shuffle(seed=42)
    print(f"Total combined size: {len(big_dataset)} samples.")

    # 4. Split 80/10/10 with 50/50 balance
    final_dataset_dict = split_dataset_balanced(big_dataset)
    
    print(f"\n💾 Saving final combined dataset to {combined_dir}")
    final_dataset_dict.save_to_disk(combined_dir)
    
    print("\n🎉 Done! Dataset ready.")
    print(f"Train: {len(final_dataset_dict['train'])}")
    print(f"Validation: {len(final_dataset_dict['validation'])}")
    print(f"Test: {len(final_dataset_dict['test'])}")

if __name__ == "__main__":
    main()
