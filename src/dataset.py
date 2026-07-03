import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# --- PATHS ---
BASE_PATH = r"D:\PDPU\Projects\Diabetic Retionpathy Detection\archive"

TRAIN_CSV  = os.path.join(BASE_PATH, "train_1.csv")
TRAIN_IMGS = os.path.join(BASE_PATH, "train_images", "train_images")

# Grade descriptions
GRADE_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# Load CSV
df = pd.read_csv(TRAIN_CSV)

print("=== Dataset Info ===")
print(f"Total training images: {len(df)}")
print(f"\nGrade distribution:")
print(df['diagnosis'].value_counts().sort_index())

# Show one sample image per grade (5 total)
fig, axes = plt.subplots(1, 5, figsize=(15, 4))
fig.suptitle("Sample Fundus Images - DR Detection", fontsize=14)

samples = df.groupby('diagnosis').first().reset_index()

for i, row in samples.iterrows():
    img_path = os.path.join(TRAIN_IMGS, row['id_code'] + ".png")
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].set_title(f"Grade {int(row['diagnosis'])}\n{GRADE_NAMES[int(row['diagnosis'])]}")
    axes[i].axis('off')

plt.tight_layout()

OUTPUT_PATH = r"D:\PDPU\Projects\Diabetic Retionpathy Detection\outputs\sample_images.png"
plt.savefig(OUTPUT_PATH)
plt.show()
print("\nDone. Image saved to outputs folder.")