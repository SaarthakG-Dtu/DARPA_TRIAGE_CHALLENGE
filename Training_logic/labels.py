import numpy as np
import os

def create_stacked_one_hot(num_pixels=1296, num_columns=19, positions_dict=None, output_path="directory_path.npy"):
    """
    Args:
        num_pixels (int): Number of pixels per column.
        num_columns (int): Number of columns (frames or sets).
        positions_dict (dict): Dictionary where key=column index (0 to num_columns-1) 
                               and value=list of positions (1-based) to set as 1.
        output_path (str): Path to save the output .npy file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize a single flat vector
    stacked_vector = np.zeros((num_pixels * num_columns, 1), dtype=np.int32)

    for col, positions in positions_dict.items():
        base_idx = col * num_pixels  # offset for each column
        for pos in positions:
            if 1 <= pos <= num_pixels:
                stacked_vector[base_idx + (pos - 1), 0] = 1
            else:
                print(f"Warning: Position {pos} is out of range for column {col}")

    np.save(output_path, stacked_vector)
    print(f"Saved stacked one-hot vector to {output_path}")
    print(f"Shape: {stacked_vector.shape}")

    return stacked_vector


positions_dict = {
    0: [1, 50, 100],       # column 0
    1: [200, 400, 800],    
    2: [1296],             
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
    10: [],
    11: [],
    12: [],
    13: [],
    14: [],
    15: [],
    16: [],
    17: [],
    18: [10, 20, 30]    
}

stacked_vector = create_stacked_one_hot(
    num_pixels=1296,
    num_columns=19,
    positions_dict=positions_dict,
    output_path="/home/saarthak/Motor_alertness_final/labels.npy"
)
