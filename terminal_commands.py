import sys
import io

# Set stdout encoding to handle UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

guide = """
================================================================================
          STOCK PRICE PREDICTION - TERMINAL COMMANDS
          (All Graphs Display in Terminal)
================================================================================

WHERE TO PASTE COMMANDS:
────────────────────────────────────────────────────────────────────────────

1. Open PowerShell:
   - Press: Windows Key + R
   - Type: powershell
   - Press: Enter

2. Navigate to folder:
   
   cd "d:\\ML\\ML LAB\\Stock-Price-Prediction-master"
   
   Then press Enter


================================================================================
MAIN COMMANDS (Copy & Paste in PowerShell):
────────────────────────────────────────────────────────────────────────────

1. LIVE GRAPHS + ACCURACY (Everything in Terminal):
   
   python live_terminal_graphs.py
   
   Shows 6 different graphs:
   - Accuracy comparison bars
   - Error metrics (RMSE & MAE)
   - Actual vs Predicted chart
   - Residuals distribution histogram
   - Detailed metrics table
   - Error magnitude distribution
   Fast: ~15-20 seconds
   No image files needed


2. ACCURACY ONLY:
   
   python show_accuracy.py
   
   Shows: R² Score, RMSE, MAE, percentage accuracy
   Fast: ~10 seconds


3. TRAIN MODEL:
   
   python stock_pred_simple.py
   
   Trains the model and shows training progress
   Time: ~30-45 seconds


================================================================================
QUICK USAGE:
────────────────────────────────────────────────────────────────────────────

See all graphs:      python live_terminal_graphs.py
See accuracy only:   python show_accuracy.py
Train & run model:   python stock_pred_simple.py


================================================================================
WHAT YOU'LL SEE:
────────────────────────────────────────────────────────────────────────────

Running: python live_terminal_graphs.py

Will display:

GRAPH 1: Accuracy Comparison (bar chart)
  Training:  [████████████████████████] 97.64%
  Testing:   [███████████████████████ ]  90.07%

GRAPH 2: Error Metrics (RMSE & MAE)
  Bars showing error values

GRAPH 3: Actual vs Predicted
  ASCII chart visualization

GRAPH 4: Residuals Distribution
  Histogram showing error distribution

GRAPH 5: Detailed Metrics Table
  R², RMSE, MAE, statistics

GRAPH 6: Error Distribution
  Percentile breakdown with bars

SUMMARY: Final status and metrics


================================================================================
EXECUTION TIMES:
────────────────────────────────────────────────────────────────────────────

live_terminal_graphs.py   10-20 seconds
show_accuracy.py          5-10 seconds
stock_pred_simple.py      30-45 seconds


================================================================================
SYSTEM INFO:
────────────────────────────────────────────────────────────────────────────

All graphs display in terminal only
No image viewer needed
No extra files created
Works on any terminal/console
Model files stored in: outputs/lstm_best_7-3-19_12AM/dropout_layers_0.4_0.4/


================================================================================
"""

print(guide)

# Save as text file
output_file = r"d:\\ML\\ML LAB\\Stock-Price-Prediction-master\\TERMINAL_COMMANDS.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(guide)

print(f"\\n✓ Guide saved to: {output_file}")
print("\\n✨ Ready to use! Just paste the commands above in your PowerShell terminal.")
