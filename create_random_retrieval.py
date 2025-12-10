import pandas as pd
import random
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)

# Read the original dataset
print("Reading phoneme_training_dataset.csv...")
df = pd.read_csv('phoneme_training_dataset.csv')

# Extract all unique phoneme-sentence pairs (without scores for uniqueness)
phoneme_pairs_set = set()

print("Extracting all phoneme-sentence pairs...")
# Add target phoneme-sentence pairs
for _, row in df.iterrows():
    phoneme_pairs_set.add((row['target_phoneme'], row['target_text']))

    # Add similar phoneme-sentence pairs
    for i in range(1, 8):
        phoneme_col = f'similar_phoneme_{i}'
        text_col = f'similar_text_{i}'
        if pd.notna(row[phoneme_col]) and pd.notna(row[text_col]):
            phoneme_pairs_set.add((row[phoneme_col], row[text_col]))

# Convert to list for random sampling
phoneme_sentence_pairs = list(phoneme_pairs_set)
print(f"Total unique phoneme-sentence pairs: {len(phoneme_sentence_pairs)}")

# Create random retrieval dataset
random_data = []

print("Creating random retrieval dataset...")
for _, row in df.iterrows():
    target_phoneme = row['target_phoneme']
    target_text = row['target_text']

    # Filter out the target pair itself
    available_pairs = [p for p in phoneme_sentence_pairs if p != (target_phoneme, target_text)]

    sampled_pairs = random.sample(available_pairs, 7)

    # Create the new row with random similarity scores
    new_row = {
        'target_phoneme': target_phoneme,
        'target_text': target_text,
        'similar_phoneme_1': sampled_pairs[0][0],
        'similar_text_1': sampled_pairs[0][1],
        'similarity_score_1': random.random(),
        'similar_phoneme_2': sampled_pairs[1][0],
        'similar_text_2': sampled_pairs[1][1],
        'similarity_score_2': random.random(),
        'similar_phoneme_3': sampled_pairs[2][0],
        'similar_text_3': sampled_pairs[2][1],
        'similarity_score_3': random.random(),
        'similar_phoneme_4': sampled_pairs[3][0],
        'similar_text_4': sampled_pairs[3][1],
        'similarity_score_4': random.random(),
        'similar_phoneme_5': sampled_pairs[4][0],
        'similar_text_5': sampled_pairs[4][1],
        'similarity_score_5': random.random(),
        'similar_phoneme_6': sampled_pairs[5][0],
        'similar_text_6': sampled_pairs[5][1],
        'similarity_score_6': random.random(),
        'similar_phoneme_7': sampled_pairs[6][0],
        'similar_text_7': sampled_pairs[6][1],
        'similarity_score_7': random.random()
    }

    random_data.append(new_row)

# Create DataFrame and save
random_df = pd.DataFrame(random_data)
output_file = 'phoneme_training_dataset_random_retrieval.csv'
random_df.to_csv(output_file, index=False)

print(f"\nRandom retrieval dataset created successfully!")
print(f"Output file: {output_file}")
print(f"Number of rows: {len(random_df)}")
print(f"\nFirst few rows:")
print(random_df.head())
