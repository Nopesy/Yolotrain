import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Specify your CSV files here ---
csv_file_paths = [
    "fixede3.csv",
    "fixede4.csv",
    # Add more paths as needed
]

# Derive labels from filenames
labels = [os.path.splitext(os.path.basename(p))[0] for p in csv_file_paths]

# Read all CSVs into DataFrames
dfs = [pd.read_csv(p) for p in csv_file_paths]

# Compute a continuous step index (epoch + batch fraction) for training-loss plot
for df in dfs:
    max_batch = df['batch'][df['batch'].apply(lambda x: str(x).isdigit())].astype(int).max()
    df['step'] = df.apply(
        lambda row: row['epoch'] + (int(row['batch']) / (max_batch + 1)
                                    if str(row['batch']).isdigit() else 1.0),
        axis=1
    )

plt.figure(figsize=(10, 6))
for df, label in zip(dfs, labels):
    # Plot training loss as solid line
    plt.plot(df['step'], df['loss'], label=f"{label} train")

    # Extract validation-loss rows (batch non-numeric)
    val_rows = df[~df['batch'].apply(lambda x: str(x).isdigit())]
    if val_rows.empty:
        val_rows = df.groupby('epoch').tail(1)

    # Plot validation loss as dots
    plt.scatter(val_rows['epoch'], val_rows['val_loss'], marker='o', label=f"{label} val")

plt.xlabel('Epoch + Batch Fraction')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()