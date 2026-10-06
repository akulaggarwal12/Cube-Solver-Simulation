import torch
import torch.nn as nn
import torch.optim as optim
import time

# Import the functions and classes we built in Steps 1 & 2
# (Assuming you saved them as step1_data_generator.py and step2_cnn_model.py)
from step1_data_generator import generate_rl_dataset
from step2_cnn_model import CubeCNN

def train_model():
    # 1. Initialize the Neural Network
    model = CubeCNN()
    
    # Move model to GPU if available for faster training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[+] Training on device: {device}")

    # 2. Define the Optimizer and Loss Function
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # Mean Squared Error

    # 3. Training Hyperparameters
    epochs = 10000           # How many times we update the network
    batch_size = 50  # How many cubes we scramble per epoch
    epochs_spent_on_current_depth = 0
    # 3. Training Hyperparameters & Advanced Trackers
    base_lr = 0.001
    min_lr = 0.00001  # Hard floor to prevent vanishing LR!
    max_scramble_depth = 12
    current_max_depth = 2

    barrier_percentage = 0.15   # Loss must drop by 15% to trigger the brakes
    anchor_loss = float('inf')
    best_loss = float('inf')

    best_final_loss = 0.5486
    target_max_depth = 10 # Change this to whatever your final level is!

    print("\n--- Starting Deep RL Training ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epochs_spent_on_current_depth += 1
    
    # FIXED: Used * instead of +, and used current_max_depth
        required_epochs = int(300 * (1.2 ** (current_max_depth - 2)))
    
    # FIXED: Consistent variable names
        if epochs_spent_on_current_depth >= required_epochs and current_max_depth < 10:
            current_max_depth += 1
            epochs_spent_on_current_depth = 0 # Reset counter
            print(f"\n[+] LEVEL UP! Now training up to Depth {current_max_depth}")
            
            # CRITICAL FIX: Prevent Ghost Anchors!
            anchor_loss = float('inf')              
            best_loss = float('inf')

        # Warm Restart the Learning Rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr
            
        else:
        # Gentle LR Decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.998
        # A. Generate a fresh batch of random experiences (Autodidactic learning)
        # X: The (6, 6, 3, 3) tensors. Y: The actual scramble depths.
        X_batch, Y_batch = generate_rl_dataset(num_samples=batch_size, max_scramble_depth=max_scramble_depth)
        
        # Move data to the same device as the model (CPU/GPU)
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        # B. Forward Pass (Ask the CNN for its predictions)
        predictions = model(X_batch)

        # C. Calculate the Loss (How wrong were the predictions?)
        loss = criterion(predictions, Y_batch)

        # D. Backward Pass (Calculate the gradients)
        optimizer.zero_grad() # Clear old gradients
        loss.backward()       # Compute new gradients via backpropagation

        # E. Update the Weights (Adam takes a step down the gradient)
        optimizer.step()
        
        # --- THE TWO-SIDED TRUST REGION ---
        current_loss_val = loss.item()
        
        # Always track the absolute lowest point achieved
        if current_loss_val < best_loss:
            best_loss = current_loss_val
            
        # YOUR FIX: Only arm the anchors and brakes if loss is in the stable zone (<= 3.0)
        if current_loss_val <= 3.0:
            
            # Arm the anchor the very first time we cross into the Safe Zone
            if anchor_loss == float('inf'):
                anchor_loss = current_loss_val
                
            lower_bound_success = anchor_loss * (1.0 - barrier_percentage)
            upper_bound_spike = anchor_loss * (1.0 + barrier_percentage)
            
            # 1. Breakthrough (Lower Bound)
            if current_loss_val <= lower_bound_success:
                old_anchor = anchor_loss 
                anchor_loss = current_loss_val 
                
                #print(f"\n[*] 15% Breakthrough! Loss dropped from {old_anchor:.4f} to {current_loss_val:.4f}")
                for param_group in optimizer.param_groups:
                    # YOUR FIX: Gentle 0.99 brake, protected by min_lr
                    param_group['lr'] = max(min_lr, param_group['lr'] * 0.99) 
                current_lr = optimizer.param_groups[0]['lr']
                #print(f"[*] Locking in minimum. New LR: {current_lr:.6f}")
                    
            # 2. Runaway Spike (Upper Bound)
            elif current_loss_val >= upper_bound_spike:
                #print(f"\n[!] 15% Spike Detected! Loss jumped from {anchor_loss:.4f} to {current_loss_val:.4f}")
                #print("[!] Network is bouncing out of the valley. Hitting the brakes!")
                
                for param_group in optimizer.param_groups:
                    # YOUR FIX: Gentle 0.99 brake, protected by min_lr
                    param_group['lr'] = max(min_lr, param_group['lr'] * 0.99) 
                    
                # FIX: We MUST move the anchor to the current loss to stop the infinite loop trap!
                anchor_loss = current_loss_val 
                current_lr = optimizer.param_groups[0]['lr']
                #print(f"[*] Anchor updated to {anchor_loss:.4f} to monitor new baseline. New LR: {current_lr:.6f}")

        else:
            # If loss is above 3.0, we are in the chaotic zone!
            # Do nothing but standard 0.998 decay, and keep the anchor detached.
            anchor_loss = float('inf')

        # --- PRINT PROGRESS ---
        if epoch % 50 == 0:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch}/{epochs}] | Depth: {current_max_depth} | Loss: {current_loss_val:.4f} | LR: {current_lr:.6f} | Time: {elapsed:.1f}s")

        # --- IDEA 5: Final Depth Model Checkpointing ---
        # Only start saving the model if we have reached our final difficulty level
        if current_max_depth == target_max_depth:
            if current_loss_val < best_final_loss:
                best_final_loss = current_loss_val
                print(f"\n[🏆] New absolute best at Max Depth! Loss: {best_final_loss:.4f}")
                print("[🏆] Saving checkpoint to 'best_cube_cnn_weights.pth'...")
                
                # Save a separate file so it doesn't overwrite your standard training save
                torch.save(model.state_dict(), 'best_cube_cnn_weights.pth')
    
    print("--- Training Complete ---")
    
    # Save the trained weights to a file so we can use them in the solver later
    torch.save(model.state_dict(), "cube_cnn_weights.pth")
    print("[+] Model saved to 'cube_cnn_weights.pth'")

if __name__ == "__main__":
    train_model()

