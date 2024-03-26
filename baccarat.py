import numpy as np

def simulate_baccarat_game():
    player_hand = np.random.randint(0, 10)
    banker_hand = np.random.randint(0, 10)
    
    if player_hand > banker_hand:
        outcome = 0  # Player win
    elif player_hand < banker_hand:
        outcome = 1  # Banker win
    else:
        outcome = 2  # Tie
    
    return player_hand, banker_hand, outcome

def simulate_baccarat_data(num_games):
    print("Simulating baccarat games...")
    data = []
    for i in range(num_games):
        if i % 1000000 == 0:
            print(f"Simulated {i} games out of {num_games}...")
        player_hand, banker_hand, outcome = simulate_baccarat_game()
        data.append([player_hand, banker_hand, outcome])
    print("Simulation completed.")
    return np.array(data)

def train_and_check(X_train, X_test, y_train, y_test, threshold=0.8):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

def save_models(best_models):
    for i, (model, accuracy) in enumerate(best_models):
        with open(f"baccarat_model_{i}_accuracy_{accuracy:.2f}.pkl", "wb") as f:
            pickle.dump(model, f)
