import pygame
import ctypes
import os
import sys
import threading
import time
import heapq
import torch
import numpy as np

# Import your CNN architecture
from step2_cnn_model import CubeCNN

# --- 1. Load C++ Engine ---
lib_path = "./cube_engine.so" if sys.platform != "win32" else "./cube_engine.dll"
try:
    cpp_cube = ctypes.CDLL(os.path.abspath(lib_path))
except OSError as e:
    print(f"[!] Library Load Error: {e}")
    sys.exit(1)

cpp_cube.get_state.restype = ctypes.POINTER(ctypes.c_int * 54)
cpp_cube.execute_move_id.argtypes = [ctypes.c_int]

# Move definitions
MOVE_FUNCTIONS = {
    0: cpp_cube.move_U,  1: cpp_cube.move_Up, 2: cpp_cube.move_U2,
    3: cpp_cube.move_L,  4: cpp_cube.move_Lp, 5: cpp_cube.move_L2,
    6: cpp_cube.move_F,  7: cpp_cube.move_Fp, 8: cpp_cube.move_F2,
    9: cpp_cube.move_R, 10: cpp_cube.move_Rp, 11: cpp_cube.move_R2,
   12: cpp_cube.move_D, 13: cpp_cube.move_Dp, 14: cpp_cube.move_D2,
   15: cpp_cube.move_B, 16: cpp_cube.move_Bp, 17: cpp_cube.move_B2,
}
MOVE_NAMES = {
    0: 'U',  1: "U'", 2: 'U2', 3: 'L',  4: "L'", 5: 'L2',
    6: 'F',  7: "F'", 8: 'F2', 9: 'R', 10: "R'", 11: 'R2',
   12: 'D', 13: "D'", 14: 'D2', 15: 'B', 16: "B'", 17: 'B2'
}

def undo_move_id(m):
    """Returns the inverse move ID to backtrack."""
    axis = m // 3
    power = m % 3
    if power == 0: return axis * 3 + 1
    if power == 1: return axis * 3 + 0
    return axis * 3 + 2

# --- 2. PyTorch Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CubeCNN()

try:
    model.load_state_dict(torch.load("best_cube_cnn_weights.pth", map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    print("[+] CNN Weights loaded successfully!")
except FileNotFoundError:
    print("[!] Warning: 'best_cube_cnn_weights.pth' not found. Model will output random guesses.")
    model.to(device)
    model.eval()

SOLVED_STATE_TUPLE = tuple([0]*9 + [1]*9 + [2]*9 + [3]*9 + [4]*9 + [5]*9)

def batch_states_to_tensor(states_list):
    """
    Vectorized One-Hot Encoding: Processes 18 states in microseconds 
    without a single Python 'for' loop.
    """
    N = len(states_list)
    # Convert list of tuples directly to a NumPy array
    states_arr = np.array(states_list, dtype=np.int64) # Shape: (N, 54)
    
    # Create empty tensor for one-hot encoding
    tensors = torch.zeros((N, 6, 54), dtype=torch.float32)
    
    # Expand colors for scatter operation
    colors = torch.from_numpy(states_arr).unsqueeze(1) # Shape: (N, 1, 54)
    
    # Scatter 1.0 into the correct color channels natively in C++ backend
    tensors.scatter_(1, colors, 1.0)
    
    # Reshape to fit our CNN architecture (N, Colors, Faces, Height, Width)
    return tensors.view(N, 6, 6, 3, 3).to(device)

# --- 3. Neural A* Search ---
def set_cpp_state(state_tuple):
    ptr = cpp_cube.get_state()
    for i in range(54): ptr.contents[i] = state_tuple[i]

def solve_with_cnn(start_state, max_moves=20, weight=2.0):
    """Weighted A* Search guided by the CNN heuristic."""
    open_set = []
    # Heap format: (f_score, g_score, state_tuple, path, last_move)
    heapq.heappush(open_set, (0.0, 0, start_state, [], -1))
    closed_set = set()
    
    while open_set:
        f, g, current_state, path, last_move = heapq.heappop(open_set)
        
        if current_state == SOLVED_STATE_TUPLE:
            return path
            
        if current_state in closed_set:
            continue
        closed_set.add(current_state)
        
        if g >= max_moves:
            continue
            
        # 1. Generate neighbors using C++ logic
        set_cpp_state(current_state)
        valid_moves = []
        neighbor_states = []
        
        for m in range(18):
            # Prune redundant moves
            if last_move != -1:
                last_axis = last_move // 3
                curr_axis = m // 3
                if curr_axis == last_axis: continue
                if last_axis == 4 and curr_axis == 0: continue
                if last_axis == 3 and curr_axis == 1: continue
                if last_axis == 5 and curr_axis == 2: continue
            
            cpp_cube.execute_move_id(m)
            n_state = tuple(cpp_cube.get_state().contents)
            # --- THE FIX: INSTANT SHORT-CIRCUIT ---
            if n_state == SOLVED_STATE_TUPLE:
                cpp_cube.execute_move_id(undo_move_id(m)) # Clean up C++ state
                return path + [m] # WE FOUND IT, STOP SEARCHING!

            if n_state not in closed_set:
                valid_moves.append(m)
                neighbor_states.append(n_state)
            cpp_cube.execute_move_id(undo_move_id(m)) # Undo move
            
        if not valid_moves:
            continue
            
        # 2. Batch Evaluate all neighbors with the CNN instantly
        tensors = batch_states_to_tensor(neighbor_states)

        with torch.no_grad():
            # Get CNN's predicted distance for all neighbors
            h_vals = model(tensors).squeeze().cpu().numpy()
            
        # Handle case where only 1 neighbor exists (numpy squeezing behavior)
        if len(valid_moves) == 1:
            h_vals = [h_vals.item()]
            
        # 3. Add to search queue
        for i, m in enumerate(valid_moves):
            h = h_vals[i]
            new_f = (g + 1) + (weight * h) # Weighted A* Formula
            heapq.heappush(open_set, (new_f, g + 1, neighbor_states[i], path + [m], m))
            
    return None

# --- 4. Pygame UI & Rendering ---
pygame.init()
CELL_SIZE = 50
GAP = 3
FONT = pygame.font.Font(None, 32)
SCREEN_WIDTH, SCREEN_HEIGHT = 1300, 650
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Rubik's Cube Simulator (Deep RL CNN Solver)")

STICKER_COLORS = {
    0: (220, 20, 60), 1: (0, 150, 0), 2: (255, 255, 255),
    3: (0, 100, 220), 4: (255, 140, 0), 5: (255, 215, 0)
}
FACE_POSITIONS = {'U': (3,0), 'L': (0,3), 'F': (3,3), 'R': (6,3), 'B': (9,3), 'D': (3,6)}
FACE_RANGES = {
    'U': range(0,9), 'L': range(9,18), 'F': range(18,27),
    'R': range(27,36), 'D': range(36,45), 'B': range(45,54)
}

PANEL_X, PANEL_WIDTH = 700, 600
SOLVE_BUTTON_RECT = pygame.Rect(730, 250, 240, 60)
solution_output = "Status: Solved"
solving_in_progress = False
cached_render_state = None # Holds visual state so UI doesn't flicker during search

def render_cube_net(state_to_render):
    offset_x, offset_y = 30, 40
    for face, (grid_x, grid_y) in FACE_POSITIONS.items():
        indices = FACE_RANGES[face]
        for i in range(9):
            r, c = i // 3, i % 3
            px, py = offset_x + (grid_x + c) * (CELL_SIZE + GAP), offset_y + (grid_y + r) * (CELL_SIZE + GAP)
            pygame.draw.rect(screen, STICKER_COLORS[state_to_render[indices[i]]], (px, py, CELL_SIZE, CELL_SIZE), border_radius=4)
            pygame.draw.rect(screen, (0, 0, 0), (px, py, CELL_SIZE, CELL_SIZE), 2, border_radius=4)

def render_wrapped_text(text, font, color, start_x, start_y, max_width):
    words = text.split(" ")
    current_line = ""
    lines = []
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        if font.size(test_line)[0] <= max_width: current_line = test_line
        else: lines.append(current_line); current_line = word
    if current_line: lines.append(current_line)
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, color), (start_x, start_y + i * 35))

def draw_ui():
    pygame.draw.rect(screen, (50, 50, 55), (PANEL_X, 0, PANEL_WIDTH, SCREEN_HEIGHT))
    screen.blit(FONT.render("Cube Simulator (CNN Engine)", True, (255, 255, 255)), (730, 40))
    screen.blit(FONT.render("Scramble: Press [X]", True, (200, 200, 200)), (730, 100))

    btn_color = (120, 80, 80) if solving_in_progress else ((100, 100, 160) if SOLVE_BUTTON_RECT.collidepoint(pygame.mouse.get_pos()) else (70, 70, 110))
    pygame.draw.rect(screen, btn_color, SOLVE_BUTTON_RECT, border_radius=8)
    solve_text = FONT.render("SOLVING..." if solving_in_progress else "SOLVE (CNN)", True, (255, 255, 255))
    screen.blit(solve_text, solve_text.get_rect(center=SOLVE_BUTTON_RECT.center))
    render_wrapped_text(solution_output, FONT, (255, 215, 0), 730, 350, PANEL_WIDTH - 60)

def run_solver_thread(start_state_tuple):
    global solution_output, solving_in_progress, cached_render_state
    
    start_time = time.time()
    path_indices = solve_with_cnn(start_state_tuple, max_moves=20, weight=2.0)
    elapsed = time.time() - start_time
    
    # Restore C++ engine to start state so we can animate properly
    set_cpp_state(start_state_tuple)
    
    if path_indices is not None:
        path_str = " ".join(MOVE_NAMES[idx] for idx in path_indices)
        solution_output = f"Solved in {len(path_indices)} move(s) [{elapsed:.2f}s]: {path_str}"
        
        # Animate the solution playback
        time.sleep(0.5)
        for move_id in path_indices:
            cpp_cube.execute_move_id(move_id)
            cached_render_state = tuple(cpp_cube.get_state().contents)
            time.sleep(0.4)
    else:
        solution_output = "No solution found. (Network may need more training)"
        
    solving_in_progress = False

# --- Main Loop ---
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and not solving_in_progress:
            shift_pressed = pygame.key.get_mods() & pygame.KMOD_SHIFT
            if event.key == pygame.K_f: cpp_cube.move_Fp() if shift_pressed else cpp_cube.move_F()
            elif event.key == pygame.K_u: cpp_cube.move_Up() if shift_pressed else cpp_cube.move_U()
            elif event.key == pygame.K_r: cpp_cube.move_Rp() if shift_pressed else cpp_cube.move_R()
            elif event.key == pygame.K_l: cpp_cube.move_Lp() if shift_pressed else cpp_cube.move_L()
            elif event.key == pygame.K_d: cpp_cube.move_Dp() if shift_pressed else cpp_cube.move_D()
            elif event.key == pygame.K_b: cpp_cube.move_Bp() if shift_pressed else cpp_cube.move_B()
            elif event.key == pygame.K_x:
                solution_output = "Scrambled!"
                for m in [0, 7, 9, 12, 3]: cpp_cube.execute_move_id(m)
            cached_render_state = tuple(cpp_cube.get_state().contents)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if SOLVE_BUTTON_RECT.collidepoint(event.pos) and not solving_in_progress:
                solving_in_progress = True
                solution_output = "CNN is evaluating paths..."
                current_state_tuple = tuple(cpp_cube.get_state().contents)
                cached_render_state = current_state_tuple # Freeze UI
                threading.Thread(target=run_solver_thread, args=(current_state_tuple,), daemon=True).start()

    screen.fill((35, 35, 38))
    
    # If starting up, populate cache
    if cached_render_state is None:
        cached_render_state = tuple(cpp_cube.get_state().contents)
        
    render_cube_net(cached_render_state)
    draw_ui()
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
