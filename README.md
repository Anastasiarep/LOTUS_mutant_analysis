# LOTUS_mutant_analysis
This repository contains analysis code for quantifying spatial germ plasm component enrichment and colocalisation from fluorescence micrographs of Drosophila oocytes and embryos in Oskar LOTUS mutants

# Germ Plasm Quantification and Imaging Analysis

This repository contains analysis code for quantifying spatial molecule enrichment and colocalisation from fluorescence micrographs of Drosophila oocytes and embryos. It supports the extraction of z-score-based linescans, anterior enrichment metrics, and Pearson correlation coefficients for multichannel imaging data.

## Repository Contents

- `*.py` script: Core analysis functions and figure generation

## Features

- Z-score normalisation and anterior-posterior intensity profiling
- Calculation of integrated anterior enrichment values
- Smoothed pixelwise Pearson correlation analysis in high-signal regions
- Bootstrap and Fisher's exact statistical testing
- Publication-ready boxplots and scatterplots
- Analysis and visualisation of pole cell induction and specification defects

## Running the Code
Prepare your data in Masks/ and Rotated/ subdirectories for each experiment.
Edit the script to point to the correct sample_dir and set the relevant channel, tissue, and figure parameters.
Run: python analysis_script.py segmentally based on desired analysis

## Figures Generated
Linescans: Normalised z-score profiles across A–P axis
Boxplots: Anterior enrichment and Pearson's correlation across genotypes
Stacked bar plot: Pole cell induction and attempt rates (Figure 2)
Scatterplots: Anterior/posterior pole cell count distributions (Figure 2)

## License
This repository is shared for academic use. Please cite appropriately if used in publications. A formal license can be added on request.

## Contact
For questions or collaboration, please contact:
Anastasia Repouliou (author) or
Cassandra G. Extavour (lead manuscript contact)

## Requirements

This code uses:
- Python 3.8+
- `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-image`

Install dependencies via:
```bash
pip install numpy scipy pandas matplotlib scikit-image

