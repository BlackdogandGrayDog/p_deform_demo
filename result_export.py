import os
import re
import pandas as pd

def extract_chamfer_distances(root_dir, step_size="1"):
    """Extracts Chamfer Distance from evaluation results and ranks models based on performance."""
    
    # ANSI escape codes for terminal output
    RED_BOLD = "\033[1;31m"  # Best model
    BLUE_BOLD = "\033[1;34m"  # Second best
    YELLOW_BOLD = "\033[1;33m"  # Third best
    RESET = "\033[0m"

    # Markdown formatting (GitHub & VS Code compatible)
    MD_RED_BOLD = "**🔴** "
    MD_BLUE_BOLD = "**🔵** "
    MD_YELLOW_BOLD = "**🟡** "
    MD_BOLD_RED = "<span style='color:red; font-weight:bold'>"

    # Dictionary to store extracted Chamfer Distance values
    data = {}

    # Iterate through each experiment folder (sorted for consistency)
    for experiment in sorted(os.listdir(root_dir)):  
        # if not (experiment.startswith("orig") or experiment.startswith("patched_0.005_0.01")):
        #     continue
        experiment_path = os.path.join(root_dir, experiment)

        if os.path.isdir(experiment_path):  # Ensure it's a folder
            eval_path = os.path.join(experiment_path, "eval")

            if os.path.exists(eval_path):
                for eval_case in sorted(os.listdir(eval_path)):  
                    eval_case_path = os.path.join(eval_path, eval_case)

                    if os.path.isdir(eval_case_path):
                        match = re.match(r"(st|gtd)_input_data_(\d+)(?:_noise_[\d\.]+)?_ts_(\d+)_gs_(\d+)_.*", eval_case)
                        if match:
                            _, traj_id, ts_id, gs_id = match.groups()  # Generalized to accept both st & gtd
                            col_name = f"trj_{traj_id}_gs_{gs_id}"
                            
                            # if traj_id != "6":
                            #     continue
                            
                            for file in os.listdir(eval_case_path):
                                if file == f"step_{step_size}_prediction_comparison.txt":  # Dynamic file selection
                                    txt_file_path = os.path.join(eval_case_path, file)
                                    with open(txt_file_path, "r") as f:
                                        content = f.read()

                                    chamfer_match = re.search(r"Chamfer Distance:\s*([\d\.eE-]+)", content)
                                    if chamfer_match:
                                        chamfer_value = float(chamfer_match.group(1))

                                        if experiment not in data:
                                            data[experiment] = {}
                                        data[experiment][col_name] = chamfer_value

    # Convert dictionary to pandas DataFrame
    df = pd.DataFrame.from_dict(data, orient="index")

    # Convert to float for numeric operations
    df_float = df.apply(pd.to_numeric, errors='coerce')

    # Identify the minimum Chamfer Distance per column
    min_values = df_float.min()

    # Compute **average Chamfer Distance** for ranking models
    df_float["Average Distance"] = df_float.mean(axis=1)
    min_values["Average Distance"] = df_float["Average Distance"].min()

    # Sort experiments by best performance
    avg_distances = df_float["Average Distance"].sort_values()
    ranked_experiments = avg_distances.index.tolist()

    # Get the top 3 best-performing models
    best_exp = ranked_experiments[0] if len(ranked_experiments) > 0 else None
    second_best_exp = ranked_experiments[1] if len(ranked_experiments) > 1 else None
    third_best_exp = ranked_experiments[2] if len(ranked_experiments) > 2 else None

    # Formatting function (strict `X.XXXe±XX`)
    def format_value(val):
        return f"{val:.3e}" if pd.notna(val) else ""

    df_formatted = df_float.applymap(format_value)

    # Apply red bold formatting for lowest values in Terminal and Markdown
    def highlight_minimum(val, column):
        """Highlight minimum values in Terminal"""
        if val == format_value(min_values[column]):
            return f"{RED_BOLD}{val}{RESET}"  # Terminal red bold
        return val

    def highlight_minimum_md(val, column):
        """Highlight minimum values in Markdown as bold red"""
        if val == format_value(min_values[column]):
            return f"{MD_BOLD_RED}{val}</span>"  # Markdown red bold for lowest values
        return val

    df_highlighted = df_formatted.apply(lambda col: col.map(lambda val: highlight_minimum(val, col.name)))
    df_md = df_formatted.apply(lambda col: col.map(lambda val: highlight_minimum_md(val, col.name)))

    # Apply ranking markers in Markdown
    def highlight_exp_name_md(exp_name):
        """Mark the best models with emojis in Markdown."""
        if exp_name == best_exp:
            return f"{MD_RED_BOLD}{exp_name}"  # 🔴 Best
        elif exp_name == second_best_exp:
            return f"{MD_BLUE_BOLD}{exp_name}"  # 🔵 Second Best
        elif exp_name == third_best_exp:
            return f"{MD_YELLOW_BOLD}{exp_name}"  # 🟡 Third Best
        return exp_name

    df_md.index = [highlight_exp_name_md(exp) for exp in df_md.index]

    # Save as Markdown format
    md_output_file = os.path.join(root_dir, f"chamfer_distance_results_step_{step_size}.md")

    with open(md_output_file, "w") as md_file:
        # Header row
        md_file.write("| Experiment Name           | " + " | ".join(df_md.columns) + " |\n")
        md_file.write("|---------------------------|" + "|".join(["------------"] * len(df_md.columns)) + "|\n")

        # Data rows
        for index, row in df_md.iterrows():
            md_file.write(f"| {index:<30} | " + " | ".join(row) + " |\n")

    # Print Markdown file location
    print("\nChamfer Distance Extraction Complete. Results saved to:")
    print("➡ Markdown Table: ", md_output_file)

    # Compute improvement of best experiment over orig_loss_130k_steps and orig_loss_56k_steps
    orig_models = [exp for exp in df_float.index if exp.startswith("orig_")]
    best_avg = df_float.loc[best_exp, "Average Distance"]

    print("\n🔬 **Best Experiment Improvement Over Original Loss Models** 🔬")

    for orig_model in orig_models:
        orig_avg = df_float.loc[orig_model, "Average Distance"]

        abs_improvement = orig_avg - best_avg
        percent_improvement = (abs_improvement / orig_avg) * 100

        print(f"📉 Against `{orig_model}`: {abs_improvement:.3e} ({percent_improvement:.2f}%) reduction")

    # Print Ranking Summary
    print("🔴 Best Model:", best_exp)
    if second_best_exp:
        print("🔵 Second Best Model:", second_best_exp)
    if third_best_exp:
        print("🟡 Third Best Model:", third_best_exp)

    return df_highlighted  # Return DataFrame for further use if needed

def extract_distances(root_dir, max_step_size=9):
    for step_size in range(1, max_step_size + 1):
        print("\n")
        print(f"✨ Extracting Chamfer Distances for step size {step_size}... 📏🚀")
        extract_chamfer_distances(root_dir, step_size=str(step_size))

