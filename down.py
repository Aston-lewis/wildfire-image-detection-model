
import kagglehub

path = kagglehub.dataset_download(
    "abdelghaniaaba/wildfire-prediction-dataset",
    output_dir="./dataset"
)

print("Dataset downloaded to:", path)
