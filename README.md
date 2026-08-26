# Call Record Transcript Word Play

A reusable Python codebook for transcript word-frequency, n-gram, sentiment and word-cloud analysis.

## Setup

The repository includes a completely synthetic `demo_data/` dataset, so
the code runs against the demo data by default.

For your own data, change:

```python
DATA_PATH = Path("demo_data")
```

to the folder containing your `.txt` files.

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the VADER lexicon once:

```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

Run:

```bash
python call_record_transcript_word_play.py
```

The script will use `demo_data/` automatically. No private transcripts
are required to test the project.

## Expected input

The loader recognises:

- `*_transcript.share.txt`
- `*_summary.txt`

## Speaker labels

The code uses only:

- `Speaker 1`
- `Speaker 2`
- `Both`
- `Neutral`

No dataset-specific names are hard-coded.

## GitHub

Recommended repository:

```text
call-record-transcript-word-play/
├── README.md
├── call_record_transcript_word_play.py
├── call_record_transcript_word_play.json
├── requirements.txt
└── .gitignore
```

Do **not** commit private transcripts or transcript-derived CSV files to a public repository.

## What it does

- Loads transcript files
- Matches transcript/summary files
- Counts words
- Calculates bigrams and trigrams
- Performs VADER sentence sentiment
- Produces sentiment n-grams
- Groups available speaker metadata
- Produces speaker × sentiment tables
- Generates word clouds
- Exports CSV analysis tables

## Limitations

Speaker attribution is only available when the input summary contains the generic speaker labels. Otherwise the record is treated as `Neutral`.

VADER is an automated lexical sentiment method. Scores are exploratory and can be affected by transcription errors, repetition, context and speech-to-text artefacts.

Review results before drawing substantive conclusions.
