import pandas as pd

train_path = "data/centralized/trainData_cleaned.csv"
test_path = "data/test/testData_cleaned.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("===== TRAIN INFO =====")
print("shape:", train_df.shape)
print("columns:", train_df.columns.tolist())
print(train_df.head())

print("\n===== TEST INFO =====")
print("shape:", test_df.shape)
print("columns:", test_df.columns.tolist())
print(test_df.head())