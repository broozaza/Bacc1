import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import baccarat  # Assuming baccarat.py is in the same directory

# Load the dataset
baccarat_data = np.load("baccarat_data.npy", allow_pickle=True)
X = baccarat_data[:, :-1]  # Features (player_hand, banker_hand)
y = baccarat_data[:, -1]   # Labels (outcome)

# Split the dataset into training and testing sets (80% training, 20% testing)
print("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Dataset split complete.")

# Train the Random Forest classifier
print("Training the Random Forest classifier...")
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Training complete.")

# Test the model
print("Testing the model...")
accuracy = model.score(X_test, y_test)
print("Test accuracy:", accuracy)

# Save the trained model
print("Saving the trained model...")
with open("baccarat_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved successfully.")
