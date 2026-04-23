#!/usr/bin/env nextflow
/*
 * pipeline/main.nf
 *
 * Nextflow pipeline wrapping the four Python scripts.
 * Run AFTER the scripts work individually.
 *
 * Usage:
 *   nextflow run pipeline/main.nf --outdir results_nf/
 *   nextflow run pipeline/main.nf --outdir results_nf/ --all_taxa
 */

nextflow.enable.dsl = 2

params.outdir    = "results_nf"
params.max_len   = 512
params.batch_size = 16
params.cv        = 5
params.all_taxa  = false

process DOWNLOAD_MIBIG {
    publishDir "${params.outdir}/raw", mode: "copy"

    output:
    path "mibig_proteins.csv", emit: proteins_csv

    script:
    def taxa_flag = params.all_taxa ? "--all-taxa" : ""
    """
    python ${projectDir}/01_download_mibig.py \
        --output . \
        ${taxa_flag}
    """
}

process GENERATE_EMBEDDINGS {
    publishDir "${params.outdir}/processed", mode: "copy"

    input:
    path proteins_csv

    output:
    path "embeddings.npy", emit: embeddings
    path "metadata.csv",   emit: metadata

    script:
    """
    python ${projectDir}/02_generate_embeddings.py \
        --input ${proteins_csv} \
        --output . \
        --max-len ${params.max_len} \
        --batch-size ${params.batch_size}
    """
}

process TRAIN_CLASSIFIER {
    publishDir "${params.outdir}/results", mode: "copy"

    input:
    path embeddings
    path metadata

    output:
    path "predictions.csv",         emit: predictions
    path "metrics_summary.json",    emit: metrics
    path "classification_report.txt"
    path "confusion_matrix.csv"
    path "best_model.pkl"

    script:
    """
    python ${projectDir}/03_train_classifier.py \
        --input . \
        --output . \
        --cv ${params.cv}
    """
}

process VISUALIZE {
    publishDir "${params.outdir}/figures", mode: "copy"

    input:
    path embeddings
    path metadata
    path predictions

    output:
    path "*.png"

    script:
    """
    python ${projectDir}/04_visualize.py \
        --embeddings ${embeddings} \
        --metadata ${metadata} \
        --predictions ${predictions} \
        --output .
    """
}

workflow {
    DOWNLOAD_MIBIG()
    GENERATE_EMBEDDINGS(DOWNLOAD_MIBIG.out.proteins_csv)
    TRAIN_CLASSIFIER(
        GENERATE_EMBEDDINGS.out.embeddings,
        GENERATE_EMBEDDINGS.out.metadata
    )
    VISUALIZE(
        GENERATE_EMBEDDINGS.out.embeddings,
        GENERATE_EMBEDDINGS.out.metadata,
        TRAIN_CLASSIFIER.out.predictions
    )
}
