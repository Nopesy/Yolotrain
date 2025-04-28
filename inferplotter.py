import pandas as pd
import matplotlib.pyplot as plt

# 1) Load metrics
df_train = pd.read_csv('train_inference.csv')
df_val   = pd.read_csv('validate_inference.csv')
df = pd.concat([df_train, df_val], ignore_index=True)

# 2) Print overall accuracies
classification_accuracy = df['correct_cls'].mean()
localization_accuracy   = df['iou_best'].mean()
print(f"Overall Classification Accuracy: {classification_accuracy:.2%}")
print(f"Overall Localization Accuracy (mean IoU): {localization_accuracy:.2f}")

# 3) Confidence vs. IoU scatter
plt.figure(figsize=(6,6))
scatter = plt.scatter(
    df['conf_best'], df['iou_best'],
    c=df['correct_cls'], cmap='coolwarm', alpha=0.6
)
plt.colorbar(scatter, label='Correct Classification (1=yes, 0=no)')
plt.axvline(0.5, linestyle='--', label='Confidence Threshold = 0.5')
plt.xlabel('Confidence Score')
plt.ylabel('IoU with GT Box')
plt.title('Confidence vs. IoU Scatter')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 4) Per-class classification accuracy
CLASS_NAMES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car",
    "cat","chair","cow","diningtable","dog","horse","motorbike",
    "person","pottedplant","sheep","sofa","train","tvmonitor"
]
per_class_acc = df.groupby('gt_class')['correct_cls'].mean().reindex(CLASS_NAMES)

plt.figure(figsize=(10,4))
per_class_acc.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Classification Accuracy')
plt.title('Accuracy by Class')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 5) Histogram of extra false positives per image
plt.figure(figsize=(5,3))
plt.hist(df['extra_fp_cnt'],
         bins=range(0, df['extra_fp_cnt'].max()+2),
         align='left', edgecolor='black')
plt.xlabel('Extra False Positives per Image')
plt.ylabel('Number of Samples')
plt.title('FP Count Distribution')
plt.tight_layout()
plt.show()

# 6) Histogram of IoU_best distribution
plt.figure(figsize=(5,3))
plt.hist(df['iou_best'], bins=20, range=(0,1), edgecolor='black')
plt.xlabel('IoU Value')
plt.ylabel('Frequency')
plt.title('Distribution of IoU_best')
plt.tight_layout()
plt.show()