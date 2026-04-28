#!/usr/bin/env python3
"""
05b_antismash_graph.py

Builds a genomic neighborhood knowledge graph from antiSMASH JSON outputs.
Accepts 1–N genomes and combines them into a single graph.

Node types:
  - gene     : CDS within a predicted BGC, colored by functional role
  - genome   : source organism (square node, connects to its BGC genes)

Edge types:
  - adjacent        : consecutive genes within the same BGC (solid)
  - shares_function : same functional keyword, different genomes (dashed, purple)

Role assignment (in priority order):
  1. CDS overlaps proto_core boundary → "core"
  2. qualifiers.gene_kind if present
  3. qualifiers.gene_functions if present
  4. Heuristic keyword match on qualifiers.product

Usage:
    Option A: explicit files
    python 05b_antismash_graph.py \\
        --jsons AL645882.json CP009124.json CP029197_1.json \\
        --output figures/ --results results/

    Option B: whole folder
    python 05b_antismash_graph.py \\
        --json-dir data/raw/antismash \\
        --output figures/ --results results/

"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network
from tqdm import tqdm

# ── Visual identity ────────────────────────────────────────────────────────────

ROLE_COLORS = {
    "core":       "#1D9E75",
    "regulatory": "#7F77DD",
    "accessory":  "#EF9F27",
    "transport":  "#378ADD",
    "other":      "#888780",
}

GENOME_COLOR = "#D85A30"

EDGE_COLORS = {
    "adjacent":        "rgba(44,44,42,0.55)",
    "shares_function": "rgba(127,119,221,0.75)",
}

ROLE_KEYWORDS = {
    "core": [
        "synthase", "synthetase", "polyketide", "nrps", "pks", "cyclase",
        "oxidase", "reductase", "hydroxylase", "methyltransferase",
        "glycosyltransferase", "halogenase", "dehydratase", "acyltransferase",
        "ketoreductase", "thioesterase", "adenylation", "condensation",
        "ketosynthase", "enoylreductase",
    ],
    "regulatory": [
        "regulator", "transcription", "repressor", "activator",
        "two-component", "sigma", "sensor", "response regulator",
    ],
    "transport": [
        "transporter", "efflux", "permease", "abc transporter",
        "exporter", "importer", "mfs",
    ],
    "accessory": [
        "isomerase", "epimerase", "mutase", "ligase", "hydrolase",
        "dehydrogenase", "aminotransferase", "phosphatase", "kinase",
        "esterase", "lactonase",
    ],
}

FUNCTION_KEYWORDS = [
    "methyltransferase", "hydroxylase", "oxidase", "reductase", "synthase",
    "synthetase", "thioesterase", "cyclase", "isomerase", "dehydrogenase",
    "transporter", "permease", "regulator", "repressor", "activator",
    "kinase", "phosphatase", "hydrolase", "ligase", "glycosyltransferase",
    "halogenase", "acyltransferase", "dehydratase", "aminotransferase",
]


# ── Location parsing ───────────────────────────────────────────────────────────

LOC_RE = re.compile(r'\[?(\d+):(\d+)\]?')

def parse_loc(loc_str: str) -> tuple[int, int] | None:
    """Parse antiSMASH location string '[start:end](strand)' → (start, end)."""
    m = LOC_RE.search(str(loc_str))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def overlaps(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end


# ── Role assignment ────────────────────────────────────────────────────────────

def role_from_keyword(product: str) -> str:
    p = product.lower()
    for role, keywords in ROLE_KEYWORDS.items():
        if any(kw in p for kw in keywords):
            return role
    return "other"


def assign_role(qualifiers: dict, gene_start: int, gene_end: int,
                core_regions: list[tuple[int, int]]) -> str:
    # 1. Position-based: overlaps proto_core → core
    for cs, ce in core_regions:
        if overlaps(gene_start, gene_end, cs, ce):
            return "core"

    # 2. gene_kind qualifier
    gk = qualifiers.get("gene_kind")
    if gk:
        gk = gk[0] if isinstance(gk, list) else gk
        mapping = {"biosynthetic": "core", "biosynthetic-additional": "accessory",
                   "regulatory": "regulatory", "transport": "transport",
                   "other": "other"}
        if gk.lower() in mapping:
            return mapping[gk.lower()]

    # 3. gene_functions qualifier
    gf = qualifiers.get("gene_functions", [])
    if gf:
        val = (gf[0] if isinstance(gf, list) else gf).lower()
        if "biosynthetic" in val:    return "core"
        if "regulatory"  in val:    return "regulatory"
        if "transport"   in val:    return "transport"
        if "additional"  in val:    return "accessory"

    # 4. Heuristic on product name
    product = qualifiers.get("product", "")
    if isinstance(product, list):
        product = " ".join(product)
    return role_from_keyword(product)


def extract_keyword(product: str) -> str | None:
    p = product.lower()
    for kw in FUNCTION_KEYWORDS:
        if kw in p:
            return kw
    return None


# ── Parsing one antiSMASH JSON ────────────────────────────────────────────────

def parse_antismash_json(json_path: Path) -> list[dict]:
    """
    Parse one antiSMASH JSON file.
    Returns list of gene dicts: {protein_id, genome_id, bgc_id, bgc_type,
                                  start, end, product, role, keyword}
    """
    data    = json.loads(json_path.read_text())
    record  = data["records"][0]
    genome_id = record["id"]
    features  = record["features"]

    # Index proto_core boundaries (defines core gene positions)
    core_regions = []
    for f in features:
        if f["type"] == "proto_core":
            loc = parse_loc(f["location"])
            if loc:
                core_regions.append(loc)

    # Index area (BGC) boundaries
    areas = []
    for area in record.get("areas", []):
        s, e     = area["start"], area["end"]
        products = area.get("products", ["unknown"])
        bgc_type = "/".join(products) if products else "unknown"
        areas.append({"start": s, "end": e, "bgc_type": bgc_type})

    if not areas:
        print(f"  Warning: no BGC areas found in {json_path.name}")
        return []

    # Parse CDS features
    genes = []
    for f in features:
        if f["type"] != "CDS":
            continue

        loc = parse_loc(f["location"])
        if not loc:
            continue
        g_start, g_end = loc

        # Find which BGC area this gene belongs to
        bgc = None
        for i, area in enumerate(areas):
            if overlaps(g_start, g_end, area["start"], area["end"]):
                bgc = area
                bgc_idx = i
                break
        if bgc is None:
            continue  # gene not in any BGC

        q          = f.get("qualifiers", {})
        product    = q.get("product", "")
        if isinstance(product, list):
            product = " ".join(product)
        protein_id = q.get("protein_id", "")
        if isinstance(protein_id, list):
            protein_id = protein_id[0]
        locus_tag  = q.get("locus_tag", "")
        if isinstance(locus_tag, list):
            locus_tag = locus_tag[0]

        node_id = f"{genome_id}|{protein_id or locus_tag or f'{g_start}-{g_end}'}"
        bgc_id  = f"{genome_id}_BGC{bgc_idx+1:02d}"
        role    = assign_role(q, g_start, g_end, core_regions)
        keyword = extract_keyword(product)

        genes.append({
            "node_id":   node_id,
            "protein_id": protein_id or locus_tag,
            "genome_id": genome_id,
            "bgc_id":    bgc_id,
            "bgc_type":  bgc["bgc_type"],
            "start":     g_start,
            "end":       g_end,
            "product":   product,
            "role":      role,
            "keyword":   keyword,
        })

    return genes


# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph(all_genes: list[dict]) -> nx.Graph:
    df = pd.DataFrame(all_genes)
    print(f"  {len(df)} genes across {df['genome_id'].nunique()} genomes "
          f"and {df['bgc_id'].nunique()} BGCs")
    print(f"  Role distribution:\n{df['role'].value_counts().to_string()}")

    G = nx.Graph()

    # ── Genome concept nodes ───────────────────────────────────────────────
    for genome_id in df["genome_id"].unique():
        G.add_node(f"GENOME::{genome_id}",
                   node_type="genome",
                   label=genome_id,
                   display_label=genome_id,
                   title=f"Genome: {genome_id}")

    # ── Gene nodes ─────────────────────────────────────────────────────────
    for _, row in df.iterrows():
        G.add_node(row["node_id"],
                   node_type="gene",
                   role=row["role"],
                   genome_id=row["genome_id"],
                   bgc_id=row["bgc_id"],
                   bgc_type=row["bgc_type"],
                   product=row["product"],
                   keyword=row["keyword"],
                   start=row["start"],
                   end=row["end"],
                   label=row["product"][:25] if row["product"] else row["protein_id"][:20],
                   title=(f"{row['protein_id']}\n"
                          f"Role: {row['role']}\n"
                          f"BGC: {row['bgc_id']} ({row['bgc_type']})\n"
                          f"Genome: {row['genome_id']}\n"
                          f"Position: {row['start']}–{row['end']}\n"
                          f"Product: {row['product']}"))

    # ── Adjacency edges (genomic order within BGC) ─────────────────────────
    adj_count = 0
    for bgc_id, group in df.groupby("bgc_id"):
        ordered = group.sort_values("start")["node_id"].tolist()
        for i in range(len(ordered) - 1):
            u, v = ordered[i], ordered[i + 1]
            G.add_edge(u, v, edge_type="adjacent",
                       bgc_id=bgc_id,
                       title=f"Adjacent in {bgc_id}",
                       weight=2)
            adj_count += 1
    print(f"  Adjacency edges: {adj_count}")

    # ── Cross-genome functional similarity edges ───────────────────────────
    keyword_nodes = defaultdict(list)
    for _, row in df.iterrows():
        if row["keyword"]:
            keyword_nodes[row["keyword"]].append(
                (row["node_id"], row["genome_id"])
            )

    share_count = 0
    for kw, entries in keyword_nodes.items():
        genomes_represented = set(g for _, g in entries)
        if len(genomes_represented) < 2:
            continue
        # One representative per genome per keyword
        seen_genomes = set()
        subset = []
        for node_id, genome_id in entries:
            if genome_id not in seen_genomes:
                subset.append((node_id, genome_id))
                seen_genomes.add(genome_id)

        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                u, gid_u = subset[i]
                v, gid_v = subset[j]
                if gid_u == gid_v:
                    continue
                if G.has_node(u) and G.has_node(v) and not G.has_edge(u, v):
                    G.add_edge(u, v,
                               edge_type="shares_function",
                               shared_function=kw,
                               title=f"Shared across genomes: {kw}",
                               weight=1)
                    share_count += 1
    print(f"  Cross-genome functional edges: {share_count}")
    print(f"  Total edges: {G.number_of_edges()}")
    return G


# ── Export ─────────────────────────────────────────────────────────────────────

def export_pyvis(G: nx.Graph, out_path: Path):
    net = Network(height="850px", width="100%",
                  bgcolor="#FAFAF8", font_color="#2C2C2A", notebook=False)
    net.force_atlas_2based(gravity=-60, central_gravity=0.005,
                           spring_length=90, spring_strength=0.08, damping=0.4)

    for node, data in G.nodes(data=True):
        ntype = data.get("node_type", "gene")
        if ntype == "genome":
            net.add_node(node, label=data["label"], color=GENOME_COLOR,
                         shape="square", size=25, font={"size": 13, "bold": True},
                         title=data.get("title", node))
        else:
            role  = data.get("role", "other")
            color = ROLE_COLORS.get(role, "#888780")
            size  = max(7, min(25, 5 + G.degree(node) * 2))
            net.add_edge
            net.add_node(node, label=data.get("label", ""), color=color,
                         size=size, font={"size": 8},
                         title=data.get("title", node))

    for u, v, data in G.edges(data=True):
        etype = data.get("edge_type", "adjacent")
        net.add_edge(u, v,
                     color=EDGE_COLORS.get(etype, "rgba(136,135,128,0.3)"),
                     width=2.0 if etype == "adjacent" else 1.2,
                     title=data.get("title", etype),
                     dashes=(etype == "shares_function"))

    net.set_options('{"nodes":{"borderWidth":0.5},'
                    '"edges":{"smooth":{"type":"continuous"}},'
                    '"physics":{"maxVelocity": 50, "stabilization":{"iterations":300}}}')
    net.save_graph(str(out_path))
    print(f"  Saved interactive → {out_path}")


def export_static(G: nx.Graph, out_path: Path):
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("white")

    pos = nx.spring_layout(G, k=2.5 / np.sqrt(max(G.number_of_nodes(), 1)),
                           seed=42, iterations=120)

    adj_edges   = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "adjacent"]
    share_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "shares_function"]

    nx.draw_networkx_edges(G, pos, edgelist=adj_edges,   ax=ax,
                           edge_color="#2C2C2A", alpha=0.40, width=1.0)
    nx.draw_networkx_edges(G, pos, edgelist=share_edges, ax=ax,
                           edge_color="#7F77DD", alpha=0.65, width=1.2, style="dashed")

    degrees = dict(G.degree())
    gene_nodes   = [n for n, d in G.nodes(data=True) if d.get("node_type") == "gene"]
    genome_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "genome"]

    for role, color in ROLE_COLORS.items():
        nodes = [n for n in gene_nodes if G.nodes[n].get("role") == role]
        if not nodes:
            continue
        sizes = [max(40, min(350, degrees[n] * 25)) for n in nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax,
                               node_color=color, node_size=sizes,
                               alpha=0.85, linewidths=0.3, edgecolors="white")

    if genome_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=genome_nodes, ax=ax,
                               node_color=GENOME_COLOR, node_size=800,
                               node_shape="s", alpha=0.95)
        nx.draw_networkx_labels(G, pos,
                                labels={n: G.nodes[n]["label"] for n in genome_nodes},
                                ax=ax, font_size=8, font_color="white", font_weight="bold")

    threshold = np.percentile([degrees[n] for n in gene_nodes], 85) if gene_nodes else 0
    gene_labels = {n: G.nodes[n].get("product", "")[:20]
                   for n in gene_nodes if degrees[n] >= threshold}
    nx.draw_networkx_labels(G, pos, labels=gene_labels, ax=ax,
                            font_size=6, font_color="#1A1A18")

    patches = [mpatches.Patch(color=c, label=f"{r} gene") for r, c in ROLE_COLORS.items()]
    patches += [mpatches.Patch(color=GENOME_COLOR, label="genome"),
                plt.Line2D([0], [0], color="#2C2C2A", lw=1.0, label="genomic adjacency"),
                plt.Line2D([0], [0], color="#7F77DD", lw=1.2, linestyle="dashed",
                           label="shared function (cross-genome)")]
    ax.legend(handles=patches, loc="upper left", fontsize=8,
              framealpha=0.9, title="Legend", title_fontsize=8)

    n_genomes = len(genome_nodes)
    n_bgcs    = len(set(nx.get_node_attributes(G, "bgc_id").values()) - {None})
    ax.set_title(
        f"antiSMASH BGC genomic neighborhood graph\n"
        f"{len(gene_nodes)} genes · {n_bgcs} BGCs · {n_genomes} Streptomyces genomes · "
        f"{len(adj_edges)} adjacency · {len(share_edges)} cross-genome functional links",
        fontsize=11, pad=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved static → {out_path}")


# ── Stats ──────────────────────────────────────────────────────────────────────

def compute_stats(G: nx.Graph) -> dict:
    gene_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "gene"]
    degrees    = {n: G.degree(n) for n in gene_nodes}

    adj_role_pairs = defaultdict(int)
    func_counts    = defaultdict(int)
    for u, v, d in G.edges(data=True):
        if d.get("edge_type") == "adjacent":
            r1 = G.nodes[u].get("role", "other")
            r2 = G.nodes[v].get("role", "other")
            adj_role_pairs[" — ".join(sorted([r1, r2]))] += 1
        elif d.get("edge_type") == "shares_function":
            func_counts[d.get("shared_function", "?")] += 1

    return {
        "n_gene_nodes":  len(gene_nodes),
        "n_edges_total": G.number_of_edges(),
        "avg_degree":    round(sum(degrees.values()) / len(degrees), 2) if degrees else 0,
        "adjacency_role_pairs": dict(
            sorted(adj_role_pairs.items(), key=lambda x: x[1], reverse=True)),
        "most_shared_cross_genome_functions": dict(
            sorted(func_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--jsons", nargs="+",
                       help="antiSMASH JSON files (1 per genome)")
    group.add_argument("--json-dir",
                       help="Directory containing antiSMASH JSON files")

    parser.add_argument("--output",  default="figures/")
    parser.add_argument("--results", default="results/")
    args = parser.parse_args()

    out_dir     = Path(args.output);  out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results); results_dir.mkdir(parents=True, exist_ok=True)

    # Resolve input files
    if args.json_dir:
        json_paths = list(Path(args.json_dir).glob("*.json"))
        # json_paths = sorted(Path(args.json_dir).glob("*.json")) # consider sorting for reproducibility:
    else:
        json_paths = [Path(p) for p in args.jsons]

    all_genes = []
    for p in json_paths:
        print(f"\n=== Parsing {p.name} ===")
        genes = parse_antismash_json(p)
        print(f"  → {len(genes)} BGC genes extracted")
        all_genes.extend(genes)

    if not all_genes:
        print("No genes found. Check that your JSON files are valid antiSMASH outputs.")
        return

    print(f"\n=== Building graph ({len(all_genes)} total genes) ===")
    G = build_graph(all_genes)

    print("\n=== Computing stats ===")
    stats = compute_stats(G)
    stats_path = results_dir / "antismash_graph_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("  Adjacency role pairs (what sits next to what):")
    for k, v in list(stats["adjacency_role_pairs"].items())[:6]:
        print(f"    {k}: {v}")
    print("  Most shared cross-genome functions:")
    for k, v in list(stats["most_shared_cross_genome_functions"].items())[:5]:
        print(f"    {k}: {v}")

    print("\n=== Exporting interactive graph ===")
    export_pyvis(G, out_dir / "antismash_graph.html")

    print("\n=== Exporting static graph ===")
    export_static(G, out_dir / "antismash_graph.png")

    print(f"\n  Stats → {stats_path}")
    print("Done. Open figures/antismash_graph.html in your browser.")

if __name__ == "__main__":
    main()
