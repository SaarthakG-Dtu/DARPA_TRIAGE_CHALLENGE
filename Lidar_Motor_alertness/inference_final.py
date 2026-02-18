import torch
import torch.nn as nn
import numpy as np
import os
import argparse

class PixelClassifier(nn.Module):
    def __init__(self, input_size=4, hidden_size=512, num_layers=2, fc_hidden=256):
        super(PixelClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden), nn.ReLU(),
            nn.Linear(fc_hidden, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out,(h_n,_) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.fc(last_hidden)
        return out

def predict_from_all_trajectories(model, all_trajectories_data, device):
    if all_trajectories_data.shape[0] == 0:
        print("[WARN] No trajectories to analyze. Predicting 'Not Twitch'.")
        # Return empty predictions array
        return "Not Twitch", 100.0, 0, 0, np.array([])

    model.eval()
    input_tensor = torch.from_numpy(all_trajectories_data).float().to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits).squeeze()
        predictions = (probabilities > 0.5).int()

    num_twitch_predictions = torch.sum(predictions).item()
    prob_sum_twitch = torch.sum(probabilities[predictions == 1])
    
    num_pixels = len(predictions)
    num_not_twitch_predictions = num_pixels - num_twitch_predictions

    # Decision Logic
    if num_twitch_predictions > 10: # Threshold for considering it a twitch event
        final_prediction_label = "Twitch"
        confidence = (prob_sum_twitch / num_twitch_predictions) * 100 if num_twitch_predictions > 0 else 0
    else:
        final_prediction_label = "Not Twitch"
        confidence = (num_not_twitch_predictions / num_pixels) * 100 if num_pixels > 0 else 100
    
    # Return the raw predictions as a numpy array
    return final_prediction_label, confidence, num_twitch_predictions, num_not_twitch_predictions, predictions.cpu().numpy()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.input_file):
        print(f"[ERROR] Input data file not found at '{args.input_file}'")
        return

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint file not found at '{args.checkpoint}'")
        return

    all_trajectories = np.load(args.input_file)
    print(f"\nLoaded filtered features with shape: {all_trajectories.shape}")

    model = PixelClassifier(input_size=all_trajectories.shape[2]).to(device)
    
    print(f"Loading model from checkpoint: '{args.checkpoint}'")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully.")

    prediction, confidence, twitch_count, not_twitch_count, raw_predictions = predict_from_all_trajectories(model, all_trajectories, device)

    # --- NEW: Save the raw predictions ---
    if args.output_file:
        np.save(args.output_file, raw_predictions)
        print(f"[INFO] Saved trajectory predictions to: {args.output_file}")
    # ------------------------------------

    print("\n--- Inference Result ---")
    print(f"Total Trajectories Analyzed: {twitch_count + not_twitch_count}")
    print(f"Twitch Predictions:          {twitch_count}")
    print(f"Not Twitch Predictions:      {not_twitch_count}")
    print("---------------------------------")
    print(f"Final Prediction:      {prediction}")
    print(f"Agreement Confidence:  {confidence:.2f}%")
    print("\n[COMPLETE] Inference finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run twitch detection inference on feature data.")
    parser.add_argument("--input_file", type=str, default="stacked_features.npy", help="Path to the input stacked features .npy file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint file.")
    # --- NEW: Argument to specify output path for predictions ---
    parser.add_argument("--output_file", type=str, help="Path to save the raw trajectory predictions .npy file.")
    # -----------------------------------------------------------
    args = parser.parse_args()
    main(args)