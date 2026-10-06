import ctypes
import os
import sys
import torch
import numpy as np

# 1. Load compiled C++ library
lib_path = "./cube_engine.so" if sys.platform != "win32" else "./cube_engine.dll"
try:
    cpp_cube = ctypes.CDLL(os.path.abspath(lib_path))
except OSError as e:
    print(f"[!] Error loading C++ library: {e}")
    sys.exit(1)

# Function Signatures
cpp_cube.get_state.restype = ctypes.POINTER(ctypes.c_int * 54)
cpp_cube.execute_move_id.argtypes = [ctypes.c_int]
cpp_cube.is_solved.restype = ctypes.c_bool

# Grid mapping matching Pygame net positions (grid_x, grid_y)
# Face grid positions (col_offset, row_offset) in a 12x9 grid
FACE_POSITIONS = {
    'U': (3, 0),  # Red
    'L': (0, 3),  # Green
    'F': (3, 3),  # White
    'R': (6, 3),  # Blue
    'B': (9, 3),  # Yellow
    'D': (3, 6),  # Orange
}

FACE_RANGES = {
    'U': range(0, 9),  'L': range(9, 18), 'F': range(18, 27),
    'R': range(27, 36), 'D': range(36, 45), 'B': range(45, 54)
}

def state_to_tensor(raw_54_state):
    """
    Converts raw 54-element C++ array into a compact PyTorch Tensor.
    User's logic: 6 Colors x 6 Faces x 9 Stickers
    Reshaped for CNNs to: (6, 6, 3, 3) -> (Colors, Faces, Height, Width)
    """
    # Initialize an empty tensor with exact required dimensions
    tensor = torch.zeros((6, 6, 3, 3), dtype=torch.float32)
    
    for i in range(54):
        face = i // 9           # Integer division gives face index (0 to 5)
        sticker = i % 9         # Modulo gives sticker index on that face (0 to 8)
        row = sticker // 3      # Row on the 3x3 face
        col = sticker % 3       # Col on the 3x3 face
        
        color_val = raw_54_state[i]  # The actual color (0 to 5)
        
        # One-hot encode: Set the specific color channel to 1.0
        tensor[color_val, face, row, col] = 1.0
        
    return tensor

def generate_rl_dataset(num_samples=200, max_scramble_depth=6):
    """
    Trajectory Sampling: Records every single intersection along the highway.
    Generates vastly more data per batch with better connected logic.
    """
    dataset_tensors = []
    dataset_targets = []
    
    state_ptr = cpp_cube.get_state()

    for i in range(num_samples):
        # 1. Reset C++ cube to perfect solved state
        for j in range(54):
            state_ptr.contents[j] = j // 9 
            
        depth = np.random.randint(1, max_scramble_depth + 1)
        last_move = -1
        
        start_recording_at = max(1, depth - 4)
        # 2. Scramble backward and SAVE EVERY STEP
        for current_step in range(1, depth + 1):
            
            # Pick a valid random move (no immediate U followed by U')
            while True:
                random_move = np.random.randint(0, 18)
                if last_move != -1:
                    last_axis = last_move // 3
                    curr_axis = random_move // 3
                    if last_axis == curr_axis:
                        continue 
                break
                
            # Apply the move in C++
            cpp_cube.execute_move_id(random_move)
            last_move = random_move

            # 3. RECORD THIS INTERSECTION
            # Because we started at 0 and scrambled outward, the 'current_step'
            # is the exact shortest distance back to 0!
            if current_step >= start_recording_at:
                raw_state = state_ptr.contents
                cube_tensor = state_to_tensor(raw_state)

                dataset_tensors.append(cube_tensor)
                dataset_targets.append(float(current_step))

    # Stack into PyTorch batch tensors
    X = torch.stack(dataset_tensors)
    Y = torch.tensor(dataset_targets, dtype=torch.float32).unsqueeze(1)

    return X, Y

if __name__ == "__main__":
    # Test Step 1: Read solved state and convert to tensor
    solved_raw = cpp_cube.get_state().contents
    tensor_sample = state_to_tensor(solved_raw)
    print(f"Single State Tensor Shape: {tensor_sample.shape}")  # Should be torch.Size([6, 9, 12])
    
    # Test dataset generation
    X_train, Y_train = generate_rl_dataset(num_samples=50, max_scramble_depth=8)
    print(f"Batch X Shape: {X_train.shape}")  # Should be torch.Size([50, 6, 9, 12])
    print(f"Batch Y Shape: {Y_train.shape}")  # Should be torch.Size([50, 1])
    print("[✓] Step 1 Complete: Tensor converter and RL Data Generator working.")
