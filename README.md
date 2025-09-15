# Python Adaptive Sparse Grid Toolkit

This toolkit provides a complete, three-phase workflow for building accurate surrogate models using adaptive sparse grids. It is designed to minimize the number of expensive function evaluations (e.g., complex simulations or physical experiments) by intelligently selecting the most informative points to add to an initial model.

The process compares a `Baseline Model`, built on an initial coarse grid, with a final `Informed Model`, which has been refined with new, strategically chosen data points.

## Features

- **Structured 3-Phase Workflow**: Guides the user from initial grid generation to final analysis.
- **Adaptive Refinement**: Intelligently identifies and selects new points to improve model accuracy where it's needed most, based on a user-defined criterion.
- **Automated Budgeting**: Can automatically determine how many new points to add using "elbow" or "threshold" methods.
- **Comprehensive Analysis**: Generates comparison plots for model accuracy, response surfaces, and error distributions.
- **Configurable**: The entire workflow is controlled by a single, easy-to-understand `config.json` file.
- **Command-Line Interface**: Simple and clear execution of each workflow phase.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Then, install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## The Three-Phase Workflow

The core of this toolkit is a three-step process designed to build a high-fidelity model efficiently.

### **Phase 1: Generate Baseline Grid**
The script generates a coarse, low-level sparse grid. The coordinates of these points are saved to a file.
- **Your Action:** You must evaluate your expensive function (run your simulation, experiment, etc.) at these specific points and fill in the results.

### **Phase 2: Select Refinement Points**
Using the data you provided from Phase 1, the script analyzes the model's behavior. It calculates a "discrepancy indicator" for a set of new candidate points and selects the most promising ones for refinement. These new points are saved to a *new* file.
- **Your Action:** You only need to evaluate your function for this smaller, targeted set of new points and fill in the results.

### **Phase 3: Final Analysis & Plotting**
The script combines the baseline data and the new refinement data. It then builds and compares two models:
1.  **Baseline Model**: The original, coarse model from Phase 1.
2.  **Informed Model**: The final, refined model using all available data.

It performs a validation check (if validation data is provided) and generates a suite of plots to visualize the improvements in accuracy and detail.

## Usage

The entire workflow is managed through `main.py` using command-line arguments. You must execute each phase in order.

### Step 1: Execute Phase 1
This will create the initial file of points for you to evaluate. For a project named `analisi_v1`, it will generate `hifi-data/analisi_v1_phase1_to_fill.dat`.

```bash
python main.py --phase 1
```

Action Required: Open the generated .dat file and fill in the output columns with your results.

### **Phase 2: Fill the refinements points**
After filling the Phase 1 file, run this command. It will analyze your data, select the best new points, and generate `hifi-data/analisi_v1_phase2_to_fill.dat`. It also saves a plot (`output/analisi_v1_phase2_discrepancy_plot.png`) showing which points were selected.

```bash
python main.py --phase 2
```

Action Required: Open the generated .dat file and fill it with the results for these new points.

### **Phase 3: Analyze the data**
Once both data files are complete, run the final phase. This will perform the full analysis, compare the models, and save all plots to the output directory.

```bash
python main.py --phase 3
```

## Configuration (`config.json`)

The `config.json` file controls all aspects of the workflow. Below is an explanation of each section using the provided example.

```json
{
    "project_name": "analisi_v1",
    "output_dir": "output",
    "data_dir": "hifi-data",

    "problem_definition": {
        "dim_names": ["W", "L", "D", "H"],
        "domains": [
            [0.4, 0.8],
            [0.0, 4.0],
            [0.3, 1.7],
            [0.3, 0.9]
        ],
        "output_quantities": ["V_FB", "T_B"]
    },

    "sparse_grid_settings": {
        "base_level": 2,
        "target_level": 3
    },

    "refinement_settings": {
        "refinement_criterion": "V_FB",
        "use_auto_budget": true,
        "auto_budget_method": "threshold",
        "auto_budget_threshold": 0.10,
        "max_budget": 33
    },
    
    "validation": {
        "validation_data_file": "validation-data.dat" 
    },

    "plotting": {
        "plot_resolution": 50,
        "slices_2d": [
            {"fixed": {"L": 0.0, "H": 0.6}, "varying": ["W", "D"]},
            {"fixed": {"W": 0.6, "D": 1.0}, "varying": ["L", "H"]}
        ]
    }
}

project_name: A unique name for your analysis. Used for naming output files.
output_dir: Directory where all plots and results will be saved.
data_dir: Directory where the data files to be filled (..._to_fill.dat) are stored.

#### problem_definition
Defines the physical problem.
- **`dim_names`**: A list of strings with the names of your input parameters. The order must be consistent.
- **`domains`**: A list of `[min, max]` pairs for each dimension, in the same order as `dim_names`.
- **`output_quantities`**: A list of strings with the names of your output quantities of interest.

#### sparse_grid_settings
Controls the sparse grid complexity.
- **`base_level`**: The level (w) of the initial coarse grid (Phase 1). A higher number means more initial points.
- **`target_level`**: The level of the denser grid from which refinement candidates are drawn (Phase 2).

#### refinement_settings
Controls the adaptive selection logic.
- **`refinement_criterion`**: The name of the output quantity (from output_quantities) that will be used to guide the refinement process. The algorithm will try to add points where this quantity is most "interesting".
- **`use_auto_budget`**: If true, the script will automatically determine how many new points to select. If false, it will use the max_budget value directly.
- **`auto_budget_method`**: Method to use for auto-budgeting. Can be "elbow" (finds the "knee" in the sorted indicator values) or "threshold".
- **`auto_budget_threshold`**: If using the threshold method, this value (e.g., 0.10 = 10%) selects all points with an indicator value above this percentage of the maximum indicator.
- **`max_budget`**: The maximum number of new points to select. Acts as a cap for auto-budgeting methods.

#### validation
Specifies how to validate the final models.
- **`validation_data_file`**: Path to a file containing test data. The script will use this data to calculate the prediction error of the baseline and informed models. If the file is not found, this step is skipped.

#### plotting
Controls the generation of 2D visualizations.
- **`plot_resolution`**: The number of points per axis for generating response surface plots (e.g., 50 means a 50x50 grid).
- **`slices_2d`**: A list of 2D slices to plot. Each item in the list must define:
    - **`fixed`**: A dictionary specifying the dimensions to hold constant and their values.
    - **`varying`**: A list of the two dimensions that will form the x and y axes of the plot.

## Input Data File Format
The script generates and reads tab-separated value (.dat) files. You can open these with any spreadsheet program or text editor.

### Phase 1 & 2 Files (..._to_fill.dat)
These files contain the coordinates where you need to provide results. The input dimension columns are pre-filled, and the output quantity columns are empty (NaN).

**Example: analisi_v1_phase1_to_fill.dat**
```tsv
W	L	D	H	V_FB	T_B
0.6	2.0	1.0	0.6	  #	     #
0.4	0.0	0.3	0.3	  #	     #
0.8	4.0	1.7	0.9	  #	     #
...	...	...	...	...	...

You need to replace # with your calculated values.

### Validation File (validation-data.dat)
This file, which you must create, provides a "ground truth" to test the models' accuracy. It should have the same structure, but all columns (inputs and outputs) must be filled with known data. These points should ideally be different from the ones used to train the models.

**Example: validation-data.dat**
```tsv
W	L	D	H	V_FB	T_B
0.55	1.5	1.1	0.75	1.23	745.6
0.71	3.2	0.8	0.40	3.98	880.1
...	...	...	...	...	...

## Outputs

All generated files are placed in the directory specified by `output_dir` in the config.

- **`{project_name}_phase2_discrepancy_plot.png`**: (From Phase 2) A bar chart showing the importance of each candidate point and which ones were selected.
- **`error_barchart_{qty}.png`**: (From Phase 3) Compares the prediction error of the baseline vs. informed model for each point in the validation set.
- **`error_distribution_{qty}.png`**: (From Phase 3) A histogram showing the distribution of prediction errors for both models.
- **`response_surface_{qty}_{slice}.png`**: (From Phase 3) A side-by-side comparison of the response surfaces generated by the baseline and informed models for a specific 2D slice.
- **`quadrature_points_{slice}.png`**: (From Phase 3) A scatter plot showing the distribution of baseline points, candidate points, and newly selected refinement points on a 2D slice.