import json
import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from upsetplot import plot as upset_plot
from upsetplot import from_contents

# Define the file sets to compare
# Each inner list is a set of files for a specific application
FILE_SETS = [
    [
        "JellyfinExample/APIClassification_Original.txt",
        "JellyfinExample/APIClassification_Securebert.txt",
        "JellyfinExample/APIClassification_LLama.txt"
    ],
    [
        "CasdoorExample/APIClassification_Original.txt",
        "CasdoorExample/APIClassification_Securebert.txt",
        "CasdoorExample/APIClassification_Llama.txt"
    ],
    [
        "AppwriteExample/APIClassification_Original.txt",
        "AppwriteExample/APIClassification_Securebert.txt",
        "AppwriteExample/APIClassification_LLama.txt"
    ]
]

OUTPUT_DIR = "comparison_results"

def load_data(file_path):
    """Loads API classification data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Remove 'upload_api' category if it exists
        if 'upload_api' in data:
            del data['upload_api']
        # Convert API lists to sets of tuples for easier comparison
        for category, apis in data.items():
            data[category] = set(tuple(api) for api in apis)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return None

def generate_summary_table(all_data):
    """Generates a summary table of API counts per category."""
    summary = defaultdict(dict)
    for file_path, data in all_data.items():
        model_name = os.path.basename(file_path).replace("APIClassification_", "").replace(".txt", "")
        for category, apis in data.items():
            summary[category][model_name] = len(apis)
    
    df = pd.DataFrame(summary).fillna(0).astype(int)
    return df

def plot_summary_bar_chart(summary_df, output_path):
    """Plots a bar chart summarizing API counts."""
    ax = summary_df.T.plot(kind='bar', figsize=(15, 8))
    plt.title("API Count per Category and Model")
    plt.ylabel("Number of APIs")
    plt.xticks(rotation=45, ha='right')

    # Add labels to the bars
    for container in ax.containers:
        ax.bar_label(container)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_venn_diagrams(data_dict, output_prefix):
    """Generates Venn diagrams for 2 or 3 sets of data."""
    num_files = len(data_dict)
    if num_files not in [2, 3]:
        print("Venn diagrams can only be generated for 2 or 3 files.")
        return

    model_names = [os.path.basename(f).replace("APIClassification_", "").replace(".txt", "") for f in data_dict.keys()]
    
    # Get all categories
    all_categories = set()
    for data in data_dict.values():
        all_categories.update(data.keys())

    for category in all_categories:
        plt.figure(figsize=(10, 7))
        sets = [data.get(category, set()) for data in data_dict.values()]
        
        if num_files == 2:
            venn2(sets, set_labels=model_names[:2])
        elif num_files == 3:
            venn3(sets, set_labels=model_names)
        
        plt.title(f"API Overlap in '{category}' Category")
        plt.savefig(f"{output_prefix}_{category}_venn.png")
        plt.close()

def plot_upset_plots(data_dict, output_prefix):
    """Generates UpSet plots for visualizing intersections of multiple sets."""
    model_names = [os.path.basename(f).replace("APIClassification_", "").replace(".txt", "") for f in data_dict.keys()]
    
    all_categories = set()
    for data in data_dict.values():
        all_categories.update(data.keys())

    for category in all_categories:
        contents = {name: data.get(category, set()) for name, data in zip(model_names, data_dict.values())}
        
        try:
            upset_data = from_contents(contents)
            if upset_data.empty:
                print(f"Skipping empty UpSet plot for category '{category}'")
                continue

            plt.figure(figsize=(12, 6))
            upset_plot(upset_data, show_counts=True, sort_by='cardinality')
            plt.title(f"API Intersections in '{category}' Category")
            
            # Adjust layout to prevent title overlap
            plt.suptitle(f"UpSet Plot for {category}", y=1.02)
            plt.savefig(f"{output_prefix}_{category}_upset.png", bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Could not generate UpSet plot for '{category}': {e}")


def main():
    """Main function to run the comparison."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_summaries = []
    for i, file_set in enumerate(FILE_SETS):
        print(f"--- Processing File Set {i+1} ---")
        
        all_data = {file: load_data(file) for file in file_set}
        all_data = {k: v for k, v in all_data.items() if v is not None}

        if not all_data:
            print("No valid data to process for this set.")
            continue

        # Generate and print summary table
        summary_df = generate_summary_table(all_data)
        all_summaries.append(summary_df)
        print("Summary of API Counts:")
        print(summary_df)
        print("-" * 30)

        set_name = os.path.basename(os.path.dirname(file_set[0]))
        
        # Generate summary bar chart
        bar_chart_path = os.path.join(OUTPUT_DIR, f"{set_name}_summary_barchart.png")
        plot_summary_bar_chart(summary_df, bar_chart_path)
        print(f"Summary bar chart saved to {bar_chart_path}")

        # Generate Venn diagrams if applicable
        if len(file_set) in [2, 3]:
            venn_prefix = os.path.join(OUTPUT_DIR, f"{set_name}")
            plot_venn_diagrams(all_data, venn_prefix)
            print(f"Venn diagrams saved with prefix {venn_prefix}")

        # Generate UpSet plots
        upset_prefix = os.path.join(OUTPUT_DIR, f"{set_name}")
        plot_upset_plots(all_data, upset_prefix)
        print(f"UpSet plots saved with prefix {upset_prefix}")
        print("\n")

    if all_summaries:
        print("--- Combined Summary Across All Examples ---")
        # The `sort=False` argument is added to preserve column order
        combined_summary_df = pd.concat(all_summaries).groupby(level=0, sort=False).sum().fillna(0).astype(int)
        
        # Ensure all original categories are present, even if they sum to 0
        all_categories = set()
        for df in all_summaries:
            all_categories.update(df.columns)
        for cat in all_categories:
            if cat not in combined_summary_df.columns:
                combined_summary_df[cat] = 0
        
        # Reorder columns to be consistent
        combined_summary_df = combined_summary_df[sorted(all_categories)]

        print(combined_summary_df)
        
        combined_chart_path = os.path.join(OUTPUT_DIR, "all_examples_summary_barchart.png")
        plot_summary_bar_chart(combined_summary_df, combined_chart_path)
        print(f"\nCombined summary bar chart saved to {combined_chart_path}")


if __name__ == "__main__":
    main()
