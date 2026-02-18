import torch
import torch.nn as nn
import numpy as np
import os

class PixelClassifier(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, fc_hidden=64):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.2) 
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.ReLU(),
            nn.Linear(fc_hidden // 2, 1)
        )

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)        
        last_hidden_state = h_n[-1]
        logits = self.fc(last_hidden_state)
        return logits


def predict_from_all_trajectories(model, all_trajectories_data, device):
    
    model.eval()

    input_tensor = torch.from_numpy(all_trajectories_data).float().to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits).squeeze() 
        predictions = (probabilities > 0.5).int()

    num_pixels = len(predictions)
    num_twitch_predictions = torch.sum(predictions).item()
    num_not_twitch_predictions = num_pixels - num_twitch_predictions

    if num_twitch_predictions > num_not_twitch_predictions:
        final_prediction_label = "Twitch"
        confidence = (num_twitch_predictions / num_pixels) * 100
    else: 
        final_prediction_label = "Not Twitch"
        confidence = (num_not_twitch_predictions / num_pixels) * 100
        
    return final_prediction_label, confidence, num_twitch_predictions, num_not_twitch_predictions


if __name__ == "__main__":
    CHECKPOINT_PATH = 'best_model_checkpoint.pth'
    INPUT_NPY_PATH = 'all_pixel_trajectories.npy'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(INPUT_NPY_PATH):
        print(f"Error: Input data file not found at '{INPUT_NPY_PATH}'")
        exit()

    all_trajectories = np.load(INPUT_NPY_PATH)
    
    num_features = all_trajectories.shape[2]

    model = PixelClassifier(input_size=num_features).to(device)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found at '{CHECKPOINT_PATH}'")
        exit()

    print(f"Loading model from checkpoint: '{CHECKPOINT_PATH}'")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully.")

    print(f"\nLoaded all pixel trajectories with shape: {all_trajectories.shape}")

    prediction, confidence, twitch_count, not_twitch_count = predict_from_all_trajectories(model, all_trajectories, device)

    print("\n--- Inference Result ---")
    print(f"Total Trajectories:    {twitch_count + not_twitch_count}")
    print(f"Twitch Predictions:    {twitch_count}")
    print(f"Not Twitch Predictions: {not_twitch_count}")
    print("---------------------------------")
    print(f"Final Prediction:      {prediction}")
    print(f"Agreement Confidence:  {confidence:.2f}%")
