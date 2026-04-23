#!/usr/bin/env python3
"""
01_download_mibig.py

Downloads MIBiG 4.0 protein sequences and JSON annotations,
parses protein functional roles (core / regulatory / accessory / transport / other),
and outputs a clean CSV filtered to Actinobacteria.

Usage:
    python 01_download_mibig.py --output data/raw/
    python 01_download_mibig.py --output data/raw/ --all-taxa

Outputs:
    data/raw/mibig_proteins.csv   columns: bgc_id, protein_id, sequence, role, bgc_type, organism, taxonomy
"""

import argparse
import json
import os
import re
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

MIBIG_FASTA_URL = "https://dl.secondarymetabolites.org/mibig/mibig_prot_seqs_4.0.fasta"
MIBIG_JSON_URL  = "https://dl.secondarymetabolites.org/mibig/mibig_json_4.0.tar.gz"

# Actinobacteria genera — used to filter taxonomy.name in MIBiG 4.0
# (4.0 dropped lineage strings; we match organism name instead)
ACTINOBACTERIA_GENERA = {
    "streptomyces", "amycolatopsis", "saccharopolyspora", "micromonospora",
    "nocardia", "kitasatospora", "actinoplanes", "streptosporangium",
    "nonomuraea", "pseudonocardia", "lentzea", "actinosynnema",
    "kutzneria", "streptoalloteichus", "salinispora", "verrucosispora",
    "catenulispora", "actinokineospora", "couchioplanes", "actinomadura",
    "planomonospora", "dactylosporangium", "glycomyces", "actinobacteria",
    "mycobacterium", "corynebacterium", "propionibacterium", "frankia",
}

# Keywords for heuristic role classification from protein annotations
ROLE_KEYWORDS = {
    "core": [
        "synthase", "synthetase", "polyketide", "nrps", "pks", "cyclase",
        "oxidase", "reductase", "hydroxylase", "methyltransferase", "glycosyltransferase",
        "halogenase", "dehydratase", "acyltransferase", "ketoreductase",
        "enoylreductase", "thioesterase", "adenylation", "condensation",
        "thiolation", "ketosynthase",
    ],
    "regulatory": [
        "regulator", "transcription", "repressor", "activator", "sensor",
        "response regulator", "two-component", "sigma", "luxr", "lasr",
        "pathway-specific", "cluster-situated",
    ],
    "transport": [
        "transporter", "efflux", "abc transporter", "permease", "exporter",
        "importer", "mfs", "resistance",
    ],
    "accessory": [
        "isomerase", "epimerase", "mutase", "ligase", "hydrolase",
        "dehydrogenase", "aminotransferase", "phosphatase", "kinase",
        "esterase", "lactonase", "phosphotransferase",
    ],
}


def download_file(url: str, dest: Path, desc: str) -> Path:
    """Download a file with progress bar if not already cached."""
    if dest.exists():
        print(f"  [cache] {dest.name} already exists, skipping download.")
        return dest
    print(f"  Downloading {desc}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")
    return dest


def classify_role(annotation: str) -> str:
    """Heuristic role classification from a free-text protein annotation."""
    ann = annotation.lower()
    for role, keywords in ROLE_KEYWORDS.items():
        if any(kw in ann for kw in keywords):
            return role
    return "other"


def parse_json_annotations(json_dir: Path) -> dict:
    """
    Parse all MIBiG 4.0 JSON files.
    MIBiG 4.0 uses a flat structure (no 'cluster' wrapper).
    Returns dict: bgc_id -> {bgc_type, organism, taxonomy, genes: {locus_tag: role}}
    """
    records = {}
    json_files = list(json_dir.glob("*.json"))
    print(f"  Parsing {len(json_files)} JSON files...")

    role_map = {
        "biosynthetic":            "core",
        "biosynthetic-additional": "accessory",
        "regulatory":              "regulatory",
        "transport":               "transport",
        "other":                   "other",
    }

    for jf in tqdm(json_files, desc="  JSON"):
        try:
            with open(jf) as f:
                data = json.load(f)

            # MIBiG 4.0: fields are at top level
            bgc_id   = data.get("accession", jf.stem)

            # biosynthesis.classes is a list of dicts with 'class' key
            classes  = data.get("biosynthesis", {}).get("classes", [])
            bgc_type = "/".join(
                c.get("class", "unknown") for c in classes
            ) if classes else "unknown"

            # taxonomy.name is organism name; no lineage string in 4.0
            tax      = data.get("taxonomy", {}) or {}
            organism = tax.get("name", "unknown")

            # genes.annotations — same structure as before
            genes = {}
            for gene in data.get("genes", {}).get("annotations", []):
                locus = gene.get("id", "")
                func  = gene.get("functions", [])
                if func:
                    category = func[0].get("category", "").lower()
                    role = role_map.get(category, "other")
                else:
                    role = None  # fall back to FASTA header heuristic
                if locus:
                    genes[locus] = role

            records[bgc_id] = {
                "bgc_type": bgc_type,
                "organism": organism,
                "genes":    genes,
            }
        except Exception:
            continue

    return records


def is_actinobacteria(organism: str) -> bool:
    """Check organism name against known Actinobacteria genera."""
    org_lower = organism.lower()
    return any(genus in org_lower for genus in ACTINOBACTERIA_GENERA)


def parse_fasta(fasta_path: Path, annotations: dict, actinobacteria_only: bool) -> pd.DataFrame:
    """
    Parse MIBiG 4.0 FASTA and merge with JSON annotations.

    MIBiG 4.0 header format:
      >BGC0000001.5|c1|1-1083|+|AEK75490.1|protein_methyltransferase|AEK75490.1
        field 0: BGC_ID.version
        field 4: protein_id
        field 5: annotation text  ← used for heuristic role classification
        field 6: accession
    """
    rows = []
    print(f"  Parsing FASTA sequences...")

    for record in tqdm(SeqIO.parse(fasta_path, "fasta"), desc="  FASTA"):
        seq = str(record.seq)
        if len(seq) < 30:
            continue

        # Parse header fields
        parts = record.id.split("|")
        # field 0 is BGC_ID.version → strip version
        bgc_id = parts[0].split(".")[0] if parts else record.id

        protein_id = parts[4] if len(parts) > 4 else record.id
        annotation = parts[5] if len(parts) > 5 else record.description

        meta     = annotations.get(bgc_id, {})
        organism = meta.get("organism", "unknown")

        # Actinobacteria filter using organism name
        if actinobacteria_only and not is_actinobacteria(organism):
            continue

        # Role: prefer JSON annotation per gene, fall back to FASTA annotation heuristic
        gene_roles = meta.get("genes", {})
        role = None
        for locus, r in gene_roles.items():
            if locus in protein_id or locus in (parts[6] if len(parts) > 6 else ""):
                role = r
                break
        if role is None:
            role = classify_role(annotation)

        rows.append({
            "bgc_id":     bgc_id,
            "protein_id": protein_id,
            "sequence":   seq,
            "role":       role,
            "bgc_type":   meta.get("bgc_type", "unknown"),
            "organism":   organism,
            "annotation": annotation,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Download and parse MIBiG 4.0 data.")
    parser.add_argument("--output",    default="data/raw/", help="Output directory")
    parser.add_argument("--all-taxa",  action="store_true",  help="Include all taxa (default: Actinobacteria only)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Step 1: Download ===")
    fasta_path   = download_file(MIBIG_FASTA_URL, out_dir / "mibig_prot_seqs_4.0.fasta", "MIBiG FASTA")
    tarball_path = download_file(MIBIG_JSON_URL,  out_dir / "mibig_json_4.0.tar.gz",     "MIBiG JSON")

    print("\n=== Step 2: Extract JSONs ===")
    json_dir = out_dir / "mibig_json_4.0"
    if not json_dir.exists():
        print(f"  Extracting to {json_dir}...")
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(out_dir)
    else:
        print(f"  [cache] {json_dir} already extracted.")

    print("\n=== Step 3: Parse annotations ===")
    annotations = parse_json_annotations(json_dir)

    print("\n=== Step 4: Build protein table ===")
    actino_only = not args.all_taxa
    df = parse_fasta(fasta_path, annotations, actinobacteria_only=actino_only)

    print(f"\n  Total proteins: {len(df)}")
    print(f"  Role distribution:\n{df['role'].value_counts().to_string()}")
    print(f"  BGC types:\n{df['bgc_type'].value_counts().head(10).to_string()}")

    out_csv = out_dir / "mibig_proteins.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n  Total proteins parsed: {len(df)}")
    if len(df) == 0:
        print("\n  WARNING: 0 proteins matched the filter.")
        print("  Try running with --all-taxa to see if data loads correctly,")
        print("  then inspect the 'organism' column to verify genus names.")
    else:
        print(f"  Role distribution:\n{df['role'].value_counts().to_string()}")
        print(f"  BGC types (top 10):\n{df['bgc_type'].value_counts().head(10).to_string()}")
        print(f"  Example organisms:\n{df['organism'].value_counts().head(5).to_string()}")

    print(f"\n  Saved: {out_csv}")
    print("\nDone. Run 02_generate_embeddings.py next.")


if __name__ == "__main__":
    main()
