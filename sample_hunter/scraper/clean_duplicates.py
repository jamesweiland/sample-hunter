import pandas as pd
import sys

def clean_duplicates(csv_path):
    df = pd.read_csv(csv_path)
    # Remove empty and duplicate URLs (ignore empty)
    df = df[df['yt_url'].notnull() & (df['yt_url'] != "")]
    df = df.drop_duplicates(subset=['yt_url'])
    df.to_csv(csv_path, index=False)
    print(f"Cleaned duplicates in {csv_path}")

if __name__ == "__main__":
    # Usage: python clean_duplicates.py query_list_with_urls.csv
    if len(sys.argv) > 1:
        clean_duplicates(sys.argv[1])
    else:
        print("Usage: python clean_duplicates.py <csv_path>")
