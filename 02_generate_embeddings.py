#!/usr/bin/env python3
"""
02_generate_embeddings.py

Generates ESM-2 protein embeddings for all sequences in the MIBiG CSV.
Uses the lightweight esm2_t6_8M_UR50D model (runs on CPU in reasonable time).

Usage:
    python 02_generate_embeddings.py --input data/raw/mibig_proteins.csv --output data/processed/
    python 02_generate_embeddings.py --input data/raw/mibig_proteins.csv --output data/processed/ --max-len 512 --batch-size 8

Outputs:
    data/processed/embeddings.npy      float32 array (N, 320)
    data/processed/metadata.csv        bgc_id, protein_id, role, bgc_type, organism per row
"""

import argparse
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_model(device: torch.device):
    """Load ESM-2 smallest model — 8M params, 320-dim, runs on CPU."""
    print("  Loading ESM-2 (esm2_t6_8M_UR50D)...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    print(f"  Model loaded on {device}.")
    return model, alphabet, batch_converter


def clean_sequence(seq: str, max_len: int) -> str:
    """Remove non-standard amino acids and truncate."""
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    seq = "".join(c if c in valid else "X" for c in seq.upper())
    return seq[:max_len]


def get_embedding(
    model,
    batch_converter,
    sequences: list[tuple[str, str]],  # [(label, seq), ...]
    device: torch.device,
    repr_layer: int = 6,
) -> np.ndarray:
    """
    Run ESM-2 forward pass and return mean-pooled residue embeddings.
    Returns array of shape (len(sequences), 320).
    """
    _, _, tokens = batch_converter(sequences)
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[repr_layer], return_contacts=False)

    # results["representations"][repr_layer]: (batch, seq_len+2, embed_dim)
    # +2 for <cls> and <eos> tokens — exclude them
    token_reps = results["representations"][repr_layer][:, 1:-1, :]

    # Mean pool over sequence length → (batch, embed_dim)
    embeddings = token_reps.mean(dim=1).cpu().numpy().astype(np.float32)
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Generate ESM-2 embeddings from MIBiG proteins.")
    parser.add_argument("--input",      default="data/raw/mibig_proteins.csv")
    parser.add_argument("--output",     default="data/processed/")
    parser.add_argument("--max-len",    type=int, default=512,  help="Truncate sequences to this length")
    parser.add_argument("--batch-size", type=int, default=16,   help="Sequences per forward pass (reduce if OOM)")
    parser.add_argument("--min-per-role", type=int, default=20, help="Drop roles with fewer sequences than this")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Step 1: Load data ===")
    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df)} proteins.")

    # Drop roles with too few examples
    role_counts = df["role"].value_counts()
    valid_roles = role_counts[role_counts >= args.min_per_role].index.tolist()
    df = df[df["role"].isin(valid_roles)].reset_index(drop=True)
    print(f"  After filtering rare roles: {len(df)} proteins.")
    print(f"  Role counts:\n{df['role'].value_counts().to_string()}")

    # Clean sequences
    df["sequence"] = df["sequence"].apply(lambda s: clean_sequence(str(s), args.max_len))
    df = df[df["sequence"].str.len() >= 30].reset_index(drop=True)

    print("\n=== Step 2: Load ESM-2 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet, batch_converter = load_model(device)

    print(f"\n=== Step 3: Generate embeddings (batch_size={args.batch_size}) ===")
    all_embeddings = []
    n = len(df)

    for start in tqdm(range(0, n, args.batch_size), desc="  Embedding"):
        batch_df  = df.iloc[start : start + args.batch_size]
        sequences = [
            (row["protein_id"], row["sequence"])
            for _, row in batch_df.iterrows()
        ]
        embs = get_embedding(model, batch_converter, sequences, device)
        all_embeddings.append(embs)

    embeddings = np.vstack(all_embeddings)  # (N, 320)
    print(f"  Embeddings shape: {embeddings.shape}")

    print("\n=== Step 4: Save ===")
    emb_path  = out_dir / "embeddings.npy"
    meta_path = out_dir / "metadata.csv"

    np.save(emb_path, embeddings)
    df[["bgc_id", "protein_id", "role", "bgc_type", "organism"]].to_csv(meta_path, index=False)

    print(f"  Saved embeddings → {emb_path}")
    print(f"  Saved metadata   → {meta_path}")
    print("\nDone. Run 03_train_classifier.py next.")


if __name__ == "__main__":
    main()
