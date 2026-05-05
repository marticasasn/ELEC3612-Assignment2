# This dataset will take up ~ 10GB of space on your workstation, make sure your laptop has the storage available
# Run this to save the dataset locally

# BEFORE RUNNING THIS EITHER DOWNLOAD PIP or if you have PIP
# then run the following in the terminal:
# pip install datasets pandas matplotlib seaborn librosa jupyter ipykernel

from datasets import load_dataset, concatenate_datasets

ds = load_dataset("lewtun/music_genres")

# Combine train and test into one
combined = concatenate_datasets([ds["train"], ds["test"]])

print(combined)  # Should show ~24,985 rows

# Save combined dataset to disk
combined.save_to_disk("music_genres_local")
print("Done!")

