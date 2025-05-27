import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

print("Loading dataset from data.pickle...")
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Handle potentially inconsistent feature dimensions
raw_data = data_dict['data']
labels = np.array(data_dict['labels'])

print(f"Dataset loaded with {len(raw_data)} samples")

# Find the maximum length of feature vectors
max_length = max(len(x) for x in raw_data)
print(f"Maximum feature length: {max_length}")

# Pad all feature vectors to the same length
padded_data = []
for sample in raw_data:
    # If the feature vector is shorter than the maximum length, pad with zeros
    if len(sample) < max_length:
        padded = sample + [0.0] * (max_length - len(sample))
    else:
        padded = sample
    padded_data.append(padded)

data = np.array(padded_data)

# Convert string labels to numeric
le = LabelEncoder()
numeric_labels = le.fit_transform(labels)

print(f"Dataset prepared with {len(data)} samples and {len(np.unique(numeric_labels))} classes")

# Show class distribution
unique_labels = np.unique(labels)
for i, label in enumerate(unique_labels):
    count = np.sum(labels == label)
    print(f"Class {label} ({i}): {count} samples")

print("\nSplitting dataset into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split(data, numeric_labels, test_size=0.2, shuffle=True, stratify=numeric_labels)

print(f"Training set size: {len(x_train)} samples")
print(f"Testing set size: {len(x_test)} samples")

print("\nTraining RandomForest model...")
# Use a simpler model with fewer hyperparameters to train faster
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(x_train, y_train)

# Evaluate the model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f"\nTest accuracy: {score * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_predict))

print("\nSaving model to model.p...")
f = open('model.p', 'wb')
# Save the model and the label encoder
pickle.dump({'model': model, 'label_encoder': le}, f)
f.close()

print("Model training complete!")
