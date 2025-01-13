import pandas as pd

def preprocess_lang8(filepath, output_filepath):
    try:
        # Load the Lang-8 data
        df = pd.read_csv(filepath)

        # Select relevant columns ('Learner Sentence', 'Corrected Sentences')
        df = df[['Learner Sentence', 'Corrected Sentences']]


        # Rename columns for easier use
        df.columns = ['text', 'corrected_text']

        # Remove rows where 'Corrected Sentences' is empty
        df.dropna(subset=['corrected_text'], inplace=True) # Drop rows with missing corrections

        # Remove duplicate rows
        df.drop_duplicates(inplace=True) #Drop any duplicate sentences (keeping only unique corrections).

        # Save preprocessed data (Optional)
        df.to_csv(output_filepath, index=False)  # Save to a new CSV file
        print(f"Preprocessed Lang-8 data saved to {output_filepath}")
        return df

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


file_path = "data/raw_lang8.csv"  
output_file_path = "data/lang8.csv"
lang8_df = preprocess_lang8(file_path, output_file_path)

if lang8_df is not None:
    print(lang8_df.head())