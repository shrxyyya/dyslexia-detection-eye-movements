# # data-update.py
# # This script processes all CSV files in a specified folder, removing duplicates and rows with empty or NaN values.
# import os
# import pandas as pd

# # Set your folder path here
# folder_path = 'D:/Shreya/engg/internship/cheal/eye-tracking-dataset/data'

# # Loop through all CSV files in the folder
# for filename in os.listdir(folder_path):
#     if filename.endswith('.csv'):
#         file_path = os.path.join(folder_path, filename)

#         # Load CSV
#         df = pd.read_csv(file_path)

#         df.drop_duplicates(inplace=True)

#         # Drop rows with any empty or NaN values
#         df.replace('', pd.NA, inplace=True)
#         df_cleaned = df.dropna()

#         # Option 1: Overwrite the original file
#         df_cleaned.to_csv(file_path, index=False)

#         print(f"Processed: {filename} ({len(df) - len(df_cleaned)} rows removed)")






# Synthetic subjects creation
# This processes the dyslexia_class_label.csv file and duplicates each entry twice.
import pandas as pd

# Read the original CSV
df = pd.read_csv('D:/Shreya/engg/internship/cheal/eye-tracking-dataset/dyslexia_class_label.csv')

# List to store all rows including duplicates
rows = []

for _, row in df.iterrows():
    # Original row
    rows.append(row)
    # First duplicate with subject_id + 1000
    dup1 = row.copy()
    dup1['subject_id'] = int('2' + str(row['subject_id'])[1:]) if len(str(row['subject_id'])) == 4 else row['subject_id'] + 1000
    rows.append(dup1)
    # Second duplicate with subject_id + 2000
    dup2 = row.copy()
    dup2['subject_id'] = int('3' + str(row['subject_id'])[1:]) if len(str(row['subject_id'])) == 4 else row['subject_id'] + 2000
    rows.append(dup2)

# Create new DataFrame
df_expanded = pd.DataFrame(rows)
df_expanded.drop_duplicates(inplace=True)

# Save to new CSV
df_expanded.to_csv('expanded_file.csv', index=False)
# Print the number of rows in the new DataFrame
print(f"Original rows: {len(df)}, Expanded rows: {len(df_expanded)}")