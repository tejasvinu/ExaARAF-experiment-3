import matplotlib
matplotlib.use('Agg') # Use non-interactive backend, suitable for scripts
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import re

def clean_column_name(col_name):
    """Cleans nvidia-smi CSV column names for easier use in pandas."""
    name = re.sub(r'\s*\[.*?\]', '', col_name) # Remove units like [MiB], [%], [W]
    name = name.strip().replace('.', '_').replace(' ', '_') # Replace . and space with _
    return name

def plot_gpu_stats(csv_filepath):
    """
    Reads GPU statistics from a CSV file and generates plots.
    """
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: GPU stats file not found at {csv_filepath}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: No data in GPU stats file {csv_filepath}")
        return
    except Exception as e:
        print(f"Error reading CSV file {csv_filepath}: {e}")
        return

    if df.empty:
        print(f"Warning: GPU stats file {csv_filepath} is empty. No plots will be generated.")
        return

    # Clean column names (e.g., "utilization.gpu [%]" -> "utilization_gpu")
    df.columns = [clean_column_name(col) for col in df.columns]

    # Ensure 'timestamp' column exists before trying to parse it
    if 'timestamp' not in df.columns:
        print(f"Error: 'timestamp' column not found in {csv_filepath}. Available columns: {df.columns.tolist()}")
        return

    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['elapsed_time_s'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    except Exception as e:
        print(f"Error processing timestamp column: {e}. Skipping plots that require time.")
        # Allow plots that don't rely on 'elapsed_time_s' if possible, or just return
        return


    output_dir = os.path.dirname(csv_filepath)
    if not output_dir: # If csv_filepath is just a filename (e.g. run in same dir)
        output_dir = '.'

    plots_to_generate = {
        'temperature_gpu': ('GPU Temperature (°C)', 'Temperature (°C)', 'gpu_temperature.png'),
        'utilization_gpu': ('GPU Utilization (%)', 'Utilization (%)', 'gpu_utilization.png'),
        'utilization_memory': ('GPU Memory Utilization (%)', 'Memory Utilization (%)', 'gpu_memory_utilization.png'),
        'memory_used': ('GPU Memory Used (MiB)', 'Memory Used (MiB)', 'gpu_memory_used.png'),
        'power_draw': ('GPU Power Draw (W)', 'Power Draw (W)', 'gpu_power_draw.png')
    }

    plots_generated_count = 0
    for col_name, (title, ylabel, filename) in plots_to_generate.items():
        if col_name in df.columns:
            try:
                plt.figure(figsize=(12, 6))
                plt.plot(df['elapsed_time_s'], df[col_name], label=title)
                if col_name == 'memory_used' and 'memory_total' in df.columns:
                    if df['memory_total'].nunique() == 1: # Plot if total memory is constant
                        total_mem = df['memory_total'].iloc[0]
                        plt.axhline(y=total_mem, color='r', linestyle='--', label=f'Total Memory ({total_mem} MiB)')
                
                plt.xlabel('Time (seconds)')
                plt.ylabel(ylabel)
                plt.title(title + ' Over Time')
                plt.legend()
                plt.grid(True)
                plot_path = os.path.join(output_dir, filename)
                plt.savefig(plot_path)
                plt.close()
                print(f"Saved plot: {plot_path}")
                plots_generated_count += 1
            except Exception as e:
                print(f"Could not generate plot for {col_name}: {e}")
        else:
            print(f"Column '{col_name}' not found in CSV, skipping corresponding plot.")
            
    if plots_generated_count == 0:
        print(f"No plots were generated. Check CSV content and column names. Available columns: {df.columns.tolist()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot GPU statistics from nvidia-smi CSV log.')
    parser.add_argument('csv_file', type=str, help='Path to the nvidia-smi CSV log file.')
    args = parser.parse_args()
    
    # Basic check for file existence before calling the main function
    if not os.path.exists(args.csv_file):
        print(f"Error: The file '{args.csv_file}' does not exist.")
    elif os.path.getsize(args.csv_file) == 0:
        print(f"Error: The file '{args.csv_file}' is empty.")
    else:
        plot_gpu_stats(args.csv_file)
