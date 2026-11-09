import kagglehub
import pandas as pd
import os

# 1. Get the folder path
path = kagglehub.dataset_download("get2jawa/toxic-comments-train")

# 2. Join the path with 'train.csv'
csv_path = os.path.join(path, "train.csv")

# 3. Read the dataset into a Pandas Dataframe
df = pd.read_csv(csv_path)

# 4. View the layout, columns, and a small preview
print("\n" + "="*50)
print(f"Dataset Size: {df.shape[0]} rows, {df.shape[1]} columns")
print("="*50)

print("\n--- Available Columns ---")
print(df.columns.tolist())

print("\n--- First 3 Comments Preview ---")
print(df[['comment_text', 'toxic']].head(3))
print("FILE LOCATION:", csv_path)
print("FILE LOCATION:", csv_path)
import pandas as pd

# 1. Load the dataset you just copied to your workspace
file_path = "train.csv"
print("Loading dataset... Please wait.")
df = pd.read_csv(file_path)

# 2. Count toxic vs non-toxic comments
# In this dataset, a '1' means toxic and a '0' means non-toxic
toxic_counts = df['toxic'].value_counts()

# 3. Calculate percentages
total_comments = len(df)
toxic_qty = toxic_counts.get(1, 0)
non_toxic_qty = toxic_counts.get(0, 0)

toxic_pct = (toxic_qty / total_comments) * 100
non_toxic_pct = (non_toxic_qty / total_comments) * 100

# 4. Display the results beautifully
print("\n" + "="*45)
print("       DATASET BALANCE ANALYSIS")
print("="*45)
print(f"Total Comments Analyzed: {total_comments:,}")
print("-"*45)
print(f"🟢 Non-Toxic Comments:  {non_toxic_qty:,} ({non_toxic_pct:.2f}%)")
print(f"🔴 Toxic Comments:      {toxic_qty:,} ({toxic_pct:.2f}%)")
print("="*45)

# 5. Quick insight on dataset balance
print("\n💡 Dataset Balance Insight:")
if abs(toxic_pct - non_toxic_pct) > 40:
    print("This dataset is heavily Imbalanced! There are significantly fewer toxic comments.")
    print("When training a machine learning model later, you will need to handle this skew")
    print("so your code doesn't just guess 'non-toxic' every time.")
else:
    print("The dataset is relatively balanced.")
    import pandas as pd
import re

# 1. Load the dataset
print("Loading data...")
df = pd.read_csv("train.csv")

# 2. Define a basic cleaning function using regex (Regular Expressions)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove newline characters
    text = re.sub(r'\n', ' ', text)
    # Remove punctuation and special characters, keep only letters and spaces
    text = re.sub(r'[^a-zA-r\s]', '', text)
    # Trim multiple spaces down to a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 3. Apply the cleaning function to a small slice of toxic data to test it
print("Cleaning a sample of toxic text...")
toxic_sample = df[df['toxic'] == 1].head(3).copy()
toxic_sample['cleaned_comment'] = toxic_sample['comment_text'].apply(clean_text)

# 4. Compare the results
for idx, row in toxic_sample.iterrows():
    print("\n--- ORIGINAL ---")
    print(row['comment_text'][:150] + "...")
    print("--- CLEANED ---")
    print(row['cleaned_comment'][:150] + "...")
    print("="*60)


