# llm-project
# Music Recommendation Demo

## Overview

`llm.py` is a small command-line music search and recommendation demo. It matches a natural-language description of what someone wants to hear against a built-in collection of tracks. For example, a query can describe a genre, era, mood, or combination of musical qualities.

The program does not generate music or search a live streaming catalog. Track details and descriptions are stored directly in the `TRACKS` list in `llm.py`.

## How It Works

1. Each track is represented by its title, artist, genre, year, mood, and a short written description.
2. `build_text()` combines those fields into one text passage for the model to compare.
3. `main()` loads the pretrained `cross-encoder/ms-marco-MiniLM-L-6-v2` model and gathers one or more search queries.
4. `rank_items()` pairs each query with every track description and asks the CrossEncoder to score the pairs. It sorts tracks by score, highest first, and returns the top five.
5. `print_results()` displays each recommended track, its score, and its description.
6. After each query, the program prints manual assessment notes. These are static examples selected by exact query text; they are not calculated from the recommendations or scores.

A CrossEncoder evaluates a query and a candidate together, so this program scores every track for each query. The displayed relevance score is useful for comparing results within a run, but it is not a probability or a guarantee that a recommendation is correct.

## Setup

Use Python with the packages listed in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

The model may need to be downloaded the first time the program runs, which requires an internet connection. Later runs can generally use the locally cached model.

## Run It

Start the program without arguments to enter a query interactively. The script then runs several additional example queries:

```powershell
python llm.py
```

Pass a query as command-line arguments to run only that query:

```powershell
python llm.py I want mellow late-night jazz
```

The program returns up to five tracks for each query.

## Main Code Areas

- `MODEL_NAME`: the pretrained CrossEncoder identifier.
- `DATASET_DESCRIPTION`: a short label printed before the results.
- `TRACKS`: the in-code catalog. Add or edit dictionaries here to change the available tracks; each item should provide the fields used by `build_text()`.
- `build_text(item)`: formats a track's metadata for model input.
- `rank_items(query, model, items, k=5)`: scores, sorts, and limits candidate tracks.
- `print_results(query, results)`: formats recommendation output.
- `main()`: loads the model, handles input, runs queries, and prints results.

## Notes and Limitations

- Recommendations depend on the written metadata and descriptions, not analysis of audio files.
- Broad or ambiguous queries can return tracks that match the mood but not every requested genre or era.
- The printed manual assessments are illustrative notes and may not correspond to the results for a changed catalog or query.
- The dataset description is a label; the actual available catalog is the `TRACKS` list in `llm.py`.
