#!/usr/bin/env python3
"""
05_knowledge_graph.py

Builds a genomic neighborhood knowledge graph of BGC genes.
Data source: MIBiG 4.0 FASTA headers + mibig_proteins.csv (local, no API needed).

FASTA header format:
  >BGC0000001.1|c1|start-end|strand|protein_id|annotation|accession

Graph:
  Nodes : genes, colored by functional role (core/regulatory/accessory/transport)
  Edges : genomic adjacency (consecutive genes sorted by coordinate within a BGC)
        + cross-BGC functional similarity (same keyword, different BGCs — dashed)

Usage:
    python 05_knowledge_graph.py \\
        --fasta  data/raw/mibig_prot_seqs_4.0.fasta \\
        --csv    data/raw/mibig_proteins.csv \\
        --output figures/ --results results/ --n-bgcs 30
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from Bio import SeqIO
from pyvis.network import Network
from tqdm import tqdm

ROLE_COLORS = {
    "core":       "#1D9E75",
    "regulatory": "#7F77DD",
    "accessory":  "#EF9F27",
    "transport":  "#378ADD",
    "other":      "#888780",
}
EDGE_COLORS = {
    "adjacent":        "rgba(44,44,42,0.50)",
    "shares_function": "rgba(127,119,221,0.70)",
}
FUNCTION_KEYWORDS = [
    "methyltransferase","hydroxylase","oxidase","reductase","synthase",
    "synthetase","thioesterase","cyclase","isomerase","dehydrogenase",
    "transporter","permease","regulator","repressor","activator",
    "kinase","phosphatase","hydrolase","ligase","glycosyltransferase",
    "halogenase","acyltransferase","dehydratase","aminotransferase",
]


def parse_fasta_coords(fasta_path):
    rows = []
    for record in tqdm(SeqIO.parse(fasta_path, "fasta"), desc="  Parsing FASTA"):
        parts = record.id.split("|")
        if len(parts) < 6:
            continue
        bgc_id     = parts[0].split(".")[0]
        coords     = parts[2]
        strand     = parts[3]
        protein_id = parts[4]
        annotation = parts[5].replace("_", " ")
        try:
            start, end = map(int, coords.split("-"))
        except ValueError:
            continue
        rows.append({"bgc_id": bgc_id, "protein_id": protein_id,
                     "start": start, "end": end, "strand": strand,
                     "annotation": annotation})
    return pd.DataFrame(rows)


def extract_keyword(annotation):
    ann = annotation.lower()
    for kw in FUNCTION_KEYWORDS:
        if kw in ann:
            return kw
    return None


def build_graph(coords_df, roles_df, n_bgcs):
    df = coords_df.merge(
        roles_df[["protein_id","role","bgc_type","organism"]],
        on="protein_id", how="left"
    )
    df["role"]     = df["role"].fillna("other")
    df["bgc_type"] = df["bgc_type"].fillna("unknown")
    df["organism"] = df["organism"].fillna("unknown")

    bgc_counts = df.groupby("bgc_id")["protein_id"].count()
    top_bgcs   = bgc_counts[bgc_counts >= 3].nlargest(n_bgcs).index.tolist()
    df         = df[df["bgc_id"].isin(top_bgcs)].copy()

    print(f"  {len(top_bgcs)} BGCs · {len(df)} genes")
    print(f"  Roles:\n{df['role'].value_counts().to_string()}")

    G = nx.Graph()

    for _, row in df.iterrows():
        ann = str(row["annotation"] or "")
        G.add_node(row["protein_id"],
            role=row["role"], bgc_id=row["bgc_id"],
            bgc_type=row["bgc_type"], organism=row["organism"],
            annotation=ann, start=row["start"], end=row["end"],
            label=ann[:28] if ann else row["protein_id"][:20],
            title=(f"{row['protein_id']}\nRole: {row['role']}\n"
                   f"BGC: {row['bgc_id']} ({row['bgc_type']})\n"
                   f"Organism: {row['organism']}\n"
                   f"Position: {row['start']}–{row['end']} ({row['strand']})\n"
                   f"Function: {ann}"))

    adj_count = 0
    for bgc_id, group in df.groupby("bgc_id"):
        ordered = group.sort_values("start")["protein_id"].tolist()
        for i in range(len(ordered) - 1):
            u, v = ordered[i], ordered[i+1]
            if G.has_node(u) and G.has_node(v):
                G.add_edge(u, v, edge_type="adjacent", bgc_id=bgc_id,
                           title=f"Adjacent in {bgc_id}", weight=2)
                adj_count += 1
    print(f"  Adjacency edges: {adj_count}")

    keyword_proteins = defaultdict(list)
    for _, row in df.iterrows():
        kw = extract_keyword(str(row["annotation"] or ""))
        if kw:
            keyword_proteins[kw].append((row["protein_id"], row["bgc_id"]))

    share_count = 0
    for kw, entries in keyword_proteins.items():
        if len(set(b for _,b in entries)) < 2:
            continue
        subset = entries[:4]
        for i in range(len(subset)):
            for j in range(i+1, len(subset)):
                pid_i, bgc_i = subset[i]
                pid_j, bgc_j = subset[j]
                if bgc_i == bgc_j:
                    continue
                if G.has_node(pid_i) and G.has_node(pid_j) and not G.has_edge(pid_i, pid_j):
                    G.add_edge(pid_i, pid_j, edge_type="shares_function",
                               shared_function=kw,
                               title=f"Shared: {kw}", weight=1)
                    share_count += 1
    print(f"  Cross-BGC functional edges: {share_count}")
    return G


def export_pyvis(G, out_path):
    net = Network(height="800px", width="100%",
                  bgcolor="#FAFAF8", font_color="#2C2C2A", notebook=False)
    net.force_atlas_2based(gravity=-50, central_gravity=0.005,
                           spring_length=80, spring_strength=0.1, damping=0.4)
    for node, data in G.nodes(data=True):
        role  = data.get("role", "other")
        color = ROLE_COLORS.get(role, "#888780")
        size  = max(8, min(30, 6 + G.degree(node) * 3))
        net.add_node(node, label=data.get("label",""), color=color,
                     size=size, title=data.get("title", node), font={"size": 8})
    for u, v, data in G.edges(data=True):
        etype = data.get("edge_type","adjacent")
        net.add_edge(u, v, color=EDGE_COLORS.get(etype,"rgba(136,135,128,0.3)"),
                     width=2.5 if etype=="adjacent" else 1.2,
                     title=data.get("title", etype),
                     dashes=(etype=="shares_function"))
    net.set_options('{"nodes":{"borderWidth":0.5},"edges":{"smooth":{"type":"continuous"}},'
                    '"physics":{"maxVelocity": 50, "stabilization":{"iterations":250}}}')
    net.save_graph(str(out_path))
    print(f"  Saved → {out_path}")


def export_static(G, out_path):
    fig, ax = plt.subplots(figsize=(14,11))
    fig.patch.set_facecolor("white")
    pos = nx.spring_layout(G, k=2.0/np.sqrt(max(G.number_of_nodes(),1)),
                           seed=42, iterations=100)
    adj_edges   = [(u,v) for u,v,d in G.edges(data=True) if d.get("edge_type")=="adjacent"]
    share_edges = [(u,v) for u,v,d in G.edges(data=True) if d.get("edge_type")=="shares_function"]
    nx.draw_networkx_edges(G, pos, edgelist=adj_edges,   ax=ax, edge_color="#2C2C2A", alpha=0.45, width=1.2)
    nx.draw_networkx_edges(G, pos, edgelist=share_edges, ax=ax, edge_color="#7F77DD", alpha=0.55, width=1.0, style="dashed")
    degrees = dict(G.degree())
    for role, color in ROLE_COLORS.items():
        nodes = [n for n,d in G.nodes(data=True) if d.get("role")==role]
        if not nodes: continue
        sizes = [max(60, min(400, degrees[n]*30)) for n in nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax,
                               node_color=color, node_size=sizes, alpha=0.85,
                               linewidths=0.4, edgecolors="white")
    threshold = np.percentile(list(degrees.values()), 80)
    labels = {n: G.nodes[n].get("annotation","")[:22]
              for n in G.nodes() if degrees[n] >= threshold}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6, font_color="#1A1A18")
    patches = [mpatches.Patch(color=c, label=f"{r} protein") for r,c in ROLE_COLORS.items()]
    patches += [plt.Line2D([0],[0], color="#2C2C2A", lw=1.2, label="genomic adjacency"),
                plt.Line2D([0],[0], color="#7F77DD", lw=1.0, linestyle="dashed", label="shared function (cross-BGC)")]
    ax.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.9,
              title="Legend", title_fontsize=8)
    n_bgcs = len(set(nx.get_node_attributes(G,"bgc_id").values()))
    ax.set_title(f"BGC genomic neighborhood graph\n"
                 f"{G.number_of_nodes()} genes · {len(adj_edges)} adjacency · "
                 f"{len(share_edges)} cross-BGC links · {n_bgcs} BGCs",
                 fontsize=11, pad=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


def compute_stats(G):
    adj_role_pairs = defaultdict(int)
    func_counts    = defaultdict(int)
    for u, v, d in G.edges(data=True):
        if d.get("edge_type") == "adjacent":
            r1, r2 = G.nodes[u].get("role","other"), G.nodes[v].get("role","other")
            adj_role_pairs[" — ".join(sorted([r1,r2]))] += 1
        elif d.get("edge_type") == "shares_function":
            func_counts[d.get("shared_function","?")] += 1
    degrees = dict(G.degree())
    return {
        "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
        "avg_degree": round(sum(degrees.values())/len(degrees),2) if degrees else 0,
        "adjacency_role_pairs": dict(sorted(adj_role_pairs.items(), key=lambda x:x[1], reverse=True)),
        "most_shared_functions": dict(sorted(func_counts.items(), key=lambda x:x[1], reverse=True)[:10]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta",   default="data/raw/mibig_prot_seqs_4.0.fasta")
    parser.add_argument("--csv",     default="data/raw/mibig_proteins.csv")
    parser.add_argument("--output",  default="figures/")
    parser.add_argument("--results", default="results/")
    parser.add_argument("--n-bgcs",  type=int, default=30)
    args = parser.parse_args()

    out_dir     = Path(args.output);  out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results); results_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Step 1: Parse FASTA coordinates ===")
    coords_df = parse_fasta_coords(Path(args.fasta))
    print(f"  {len(coords_df)} genes with coordinates")

    print("\n=== Step 2: Load role annotations ===")
    roles_df = pd.read_csv(args.csv)
    print(f"  {len(roles_df)} proteins")

    print("\n=== Step 3: Build graph ===")
    G = build_graph(coords_df, roles_df, n_bgcs=args.n_bgcs)

    print("\n=== Step 4: Stats ===")
    stats = compute_stats(G)
    stats_path = results_dir / "graph_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("  Adjacency role pairs:")
    for k,v in list(stats["adjacency_role_pairs"].items())[:6]:
        print(f"    {k}: {v}")

    print("\n=== Step 5: Interactive graph ===")
    export_pyvis(G, out_dir / "knowledge_graph.html")

    print("\n=== Step 6: Static graph ===")
    export_static(G, out_dir / "knowledge_graph.png")

    print(f"\nDone. Open figures/knowledge_graph.html in your browser.")

if __name__ == "__main__":
    main()
