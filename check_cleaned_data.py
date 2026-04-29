"""
Script to verify the quality of cleaned data in metadata.csv
"""

import pandas as pd
import numpy as np

def analyze_cleaned_data(csv_path='outputs/nlp/metadata.csv', sample_size=20):
    """
    Analyze the cleaned data to verify quality.
    
    Args:
        csv_path: Path to metadata.csv
        sample_size: Number of random samples to display
    """
    print("=" * 80)
    print("CLEANED DATA ANALYSIS")
    print("=" * 80)
    
    # Load the metadata
    df = pd.read_csv(csv_path)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total recipes processed: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    
    # Check for empty or null ingredients
    empty_count = df['ingredients'].isna().sum()
    empty_str_count = (df['ingredients'] == '').sum()
    
    print(f"\n🔍 Data Quality:")
    print(f"   Null ingredients: {empty_count}")
    print(f"   Empty string ingredients: {empty_str_count}")
    print(f"   Valid ingredients: {len(df) - empty_count - empty_str_count:,}")
    
    # Analyze ingredient text length
    df['ingredient_length'] = df['ingredients'].fillna('').str.len()
    df['ingredient_word_count'] = df['ingredients'].fillna('').str.split().str.len()
    
    print(f"\n📏 Text Statistics:")
    print(f"   Average character length: {df['ingredient_length'].mean():.1f}")
    print(f"   Average word count: {df['ingredient_word_count'].mean():.1f}")
    print(f"   Min word count: {df['ingredient_word_count'].min():.0f}")
    print(f"   Max word count: {df['ingredient_word_count'].max():.0f}")
    
    # Check for common cleaning artifacts
    print(f"\n🧹 Cleaning Verification:")
    has_punctuation = df['ingredients'].fillna('').str.contains(r'[,\.;:]').sum()
    has_numbers = df['ingredients'].fillna('').str.contains(r'\d').sum()
    has_uppercase = df['ingredients'].fillna('').str.contains(r'[A-Z]').sum()
    
    print(f"   Recipes with punctuation: {has_punctuation} ({has_punctuation/len(df)*100:.2f}%)")
    print(f"   Recipes with numbers: {has_numbers} ({has_numbers/len(df)*100:.2f}%)")
    print(f"   Recipes with uppercase: {has_uppercase} ({has_uppercase/len(df)*100:.2f}%)")
    
    # Display random samples
    print(f"\n📝 Random Sample of Cleaned Ingredients ({sample_size} recipes):")
    print("=" * 80)
    
    samples = df.sample(min(sample_size, len(df)))
    for idx, row in samples.iterrows():
        print(f"\n{row['title']}")
        print(f"   Cleaned: {row['ingredients']}")
        print(f"   Words: {row['ingredient_word_count']:.0f} | Chars: {row['ingredient_length']:.0f}")
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)

if __name__ == '__main__':
    analyze_cleaned_data()
