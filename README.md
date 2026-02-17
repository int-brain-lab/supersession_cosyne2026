# Supersession Cosyne 2026 --- Brain-Wide Map Tutorial

This repository contains code for the Cosyne 2026 supersession tutorial
on analyzing the International Brain Laboratory (IBL) Brain-Wide Map
(BWM) dataset using trial-averaged neuronal responses (concatenated
PETHs). (Slides: https://docs.google.com/presentation/d/1i2o6p9f-Ab5e5ilnPh7_cGxCQTZFWsZr6Eft3F4qLRU/edit?usp=sharing)

The tutorial demonstrates how to explore large-scale neural population
structure across the mouse brain using clustering, similarity-based
embeddings, and anatomical organization. 
(Based on https://www.biorxiv.org/content/10.1101/2025.07.30.667641v1)

------------------------------------------------------------------------

## 1. Download the trial-averaged dataset

Download the precomputed trial-averaged BWM feature matrices (each \~1
GB):

**Cross-validation OFF**\
`concat_cvFalse_ephysFalse.npy`\
https://drive.google.com/file/d/1_TEqHAbzwKqxqESLVR5MTkkC4vfh4Z-l/view?usp=sharing

**Cross-validation ON**\
`concat_cvTrue_ephysFalse.npy`\
https://drive.google.com/file/d/1afkr1UQHGLOif-khmIQO9--PWElHUSj-/view?usp=sharing

After download, place the files into your local IBL cache directory:

    <ONE cache>/dmn/res/

Example (Linux):

    ~/FlatIron/dmn/res/

These files contain trial-averaged concatenated peri-event time
histograms (PETHs) for \>50,000 neurons across the Brain-Wide Map
dataset.

------------------------------------------------------------------------

## 2. Quick start

### Plot example neurons

Visualize a few neurons' concatenated PETH feature vectors:

``` python
plot_example_neurons()
```

------------------------------------------------------------------------

## 3. Population structure visualizations

### K-means clustering

View all neurons ordered by k-means cluster identity:

``` python
plot_rastermap(
    mapping='kmeans',
    cv=False,
    bg=True,
    sort_method='acs',
    bounds=True,
    exa=True
)
```

------------------------------------------------------------------------

### Similarity-based ordering (Rastermap)

Order neurons by functional similarity using a 1D Rastermap embedding.

With cross-validation: - Even trials: fit embedding\
- Odd trials: visualize responses

``` python
plot_rastermap(mapping='rm', cv=True)
```

------------------------------------------------------------------------

### Anatomical ordering (Cosmos)

Order neurons by Cosmos brain region:

``` python
plot_rastermap(
    mapping='Cosmos',
    cv=False,
    bg=True,
    sort_method='acs',
    bounds=True,
    exa=True
)
```

------------------------------------------------------------------------

## 4. About the dataset

For details on downloading and working with the raw Brain-Wide Map data:

https://docs.internationalbrainlab.org/notebooks_external/2025_data_release_brainwidemap.html

------------------------------------------------------------------------

## 5. Related publication

Basic analyses of this dataset are described in:

https://www.nature.com/articles/s41586-025-09235-0

------------------------------------------------------------------------

## 6. Scope of this tutorial

This tutorial is intended for:

-   Experimentalists seeking ready-to-use population-level summaries of
    the BWM dataset\
-   Computational neuroscientists interested in large-scale,
    standardized neural feature spaces for modeling and benchmarking

The provided matrices represent each neuron as a concatenation of
trial-averaged task responses (PETHs), enabling fast exploration of
functional organization across the brain without downloading raw spike
data.

------------------------------------------------------------------------

## 7. Notes

-   Each dataset file is \~1 GB.
-   The repository contains analysis code only; large data files are
    hosted externally.
-   First runs may take time due to caching and embedding computations.
