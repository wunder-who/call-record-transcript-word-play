# Call Record Transcript Word Play
# Replace your text here with the transcript/summary folder.

from pathlib import Path
from collections import Counter
import re, random
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer

DATA_PATH = Path(r"demo_data_synthetic")
TOP_N = 50

# Load
files = list(DATA_PATH.glob("*.txt")) - missing file
transcripts = {
    p.name: p.read_text(encoding="utf-8", errors="ignore")
    for p in files if "transcript" in p.name.lower()
}
summaries = {
    p.name: p.read_text(encoding="utf-8", errors="ignore")
    for p in files if "summary" in p.name.lower()
}
transcripts_df = pd.DataFrame(
    [{"file": k, "text": v} for k, v in transcripts.items()]
)

print("Summary files:", len(summaries))
print("Transcript files:", len(transcripts))
print(DATA_PATH.exists())
print(list(DATA_PATH.glob("*.txt")))

if not transcripts:
    raise FileNotFoundError(
        f"No transcript files found in {DATA_PATH.resolve()}. "
        "Set DATA_PATH to a folder containing *_transcript.share.txt files."
    )

# Match summaries to transcripts
def base_name(name):
    name = re.sub(r"_summary\.txt$", "", name, flags=re.I)
    return re.sub(r"_transcript\.share\.txt$", "", name, flags=re.I)

sm, tm = {base_name(k): k for k in summaries}, {base_name(k): k for k in transcripts}
matched = []

for key in sorted(set(sm) & set(tm)):
    sf, tf = sm[key], tm[key]
    s = summaries[sf].lower()
    speaker = "Both" if "both" in s else (
        "Speaker 1" if "speaker 1" in s else
        "Speaker 2" if "speaker 2" in s else
        "Neutral"
    )
    matched.append({"file": sf, "speaker": speaker, "transcript_file": tf,
                    "transcript_text": transcripts[tf]})

df_matched = pd.DataFrame(matched)
print("Matched pairs:", len(df_matched))
if not df_matched.empty:
    print(df_matched["speaker"].value_counts())

# Word frequency
all_text = " ".join(transcripts.values()).lower()
words = re.findall(r"\b[\w']+\b", all_text)
stopwords = set("""the and to a of is it i you are am was we that this in on for with
my your me do did what have has had be can will would from as at or an but so if
then there they he she""".split())
content_words = [w for w in words if w not in stopwords]
word_counts = Counter(content_words)

top_words = pd.DataFrame(
    word_counts.most_common(TOP_N), columns=["word", "count"]
)
print("Total usable words:", len(content_words))
print(top_words)

# N-grams
def ngrams(text, n):
    w = re.findall(r"\b[\w']+\b", text.lower())
    return list(zip(*(w[i:] for i in range(n))))

def ngram_table(text, n, limit=30):
    c = Counter(ngrams(text, n))
    return pd.DataFrame(
        [(" ".join(p), n) for p, n in c.most_common(limit)],
        columns=["phrase", "count"]
    )

bigrams = ngram_table(all_text, 2)
trigrams = ngram_table(all_text, 3)

# Sentiment
sia = SentimentIntensityAnalyzer()
rows = []

for _, row in transcripts_df.iterrows():
    for sentence in re.split(r"(?<=[.!?])\s+", row["text"]):
        sentence = sentence.strip()
        if not sentence:
            continue
        compound = sia.polarity_scores(sentence)["compound"]
        sentiment = "Positive" if compound >= .05 else "Negative" if compound <= -.05 else "Neutral"
        rows.append({"file": row["file"], "sentence": sentence,
                     "compound": compound, "sentiment": sentiment})

transcript_sentences_df = pd.DataFrame(
    rows, columns=["file", "sentence", "compound", "sentiment"]
)
print(transcript_sentences_df["sentiment"].value_counts())
print(transcript_sentences_df["sentiment"].value_counts(normalize=True).mul(100).round(1))

# Sentiment trigrams
sentiment_trigrams = {s: Counter() for s in ["Positive", "Neutral", "Negative"]}
for _, row in transcript_sentences_df.iterrows():
    sentiment_trigrams[row["sentiment"]].update(ngrams(row["sentence"], 3))

for sentiment, c in sentiment_trigrams.items():
    print("\n", sentiment)
    for phrase, count in c.most_common(15):
        print(f"{' '.join(phrase):30} {count}")

# Speaker attribution from matched files
if not df_matched.empty:
    lookup = df_matched.set_index("transcript_file")["speaker"].to_dict()
    transcript_sentences_df["speaker"] = transcript_sentences_df["file"].map(lookup).fillna("Neutral")

    speaker_sentiment = pd.crosstab(
        transcript_sentences_df["speaker"], transcript_sentences_df["sentiment"]
    )
    speaker_sentiment_pct = speaker_sentiment.div(
        speaker_sentiment.sum(axis=1), axis=0
    ).mul(100).round(1)

    print("\nSpeaker × sentiment")
    print(speaker_sentiment)
    print(speaker_sentiment_pct)

# Word cloud helper
def show_wordcloud(freq, title, max_words=50, color_func=None):
    wc = WordCloud(width=1400, height=800, background_color="white",
                   max_words=max_words, random_state=42).generate_from_frequencies(freq)
    if color_func:
        wc = wc.recolor(color_func=color_func, random_state=42)
    plt.figure(figsize=(15, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()

# Top words: top 4 pink/blue, rest rainbow
pink_blue = ["#E91E63", "#1565C0", "#F06292", "#1976D2"]
rainbow = ["#E53935", "#FB8C00", "#FDD835", "#43A047",
           "#00ACC1", "#1E88E5", "#8E24AA", "#D81B60"]
top4 = {w: i for i, (w, _) in enumerate(word_counts.most_common(4))}

def word_color(word, *args, **kwargs):
    return pink_blue[top4[word]] if word in top4 else random.choice(rainbow)

show_wordcloud(dict(word_counts.most_common(TOP_N)), "Top 50 Words", TOP_N, word_color)

# Trigram word cloud
tri_freq = {" ".join(p): c for p, c in Counter(ngrams(all_text, 3)).most_common(30)}
show_wordcloud(tri_freq, "Top 30 Three-Word Phrases", 30)

# Speaker-attributed negative sentences
if "speaker" in transcript_sentences_df:
    negative_speaker_sentences = transcript_sentences_df[
        (transcript_sentences_df["sentiment"] == "Negative") &
        transcript_sentences_df["speaker"].isin(["Speaker 1", "Speaker 2"])
    ][["speaker", "sentence", "compound"]].sort_values(["speaker", "compound"])
    print("\nSpeaker-attributed negative sentences")
    print(negative_speaker_sentences)

# Export
top_words.to_csv("top_words.csv", index=False)
bigrams.to_csv("top_bigrams.csv", index=False)
trigrams.to_csv("top_trigrams.csv", index=False)
transcript_sentences_df.to_csv("transcript_sentence_sentiment.csv", index=False)

if "speaker_sentiment" in locals():
    speaker_sentiment.to_csv("speaker_sentiment_counts.csv")
    speaker_sentiment_pct.to_csv("speaker_sentiment_percentages.csv")

if "negative_speaker_sentences" in locals():
    negative_speaker_sentences.to_csv("deidentified_speaker_negative_sentences.csv", index=False)

print("Analysis complete.")
