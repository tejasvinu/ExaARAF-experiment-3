import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_system_metrics(csv_filepath):
    """
    Reads system metrics from a CSV file and generates plots.
    """
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: System metrics file not found at {csv_filepath}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: No data in system metrics file {csv_filepath}")
        return
    except Exception as e:
        print(f"Error reading CSV file {csv_filepath}: {e}")
        return

    if df.empty:
        print(f"Warning: System metrics file {csv_filepath} is empty. No plots will be generated.")
        return

    if 'timestamp' not in df.columns:
        print(f"Error: 'timestamp' column not found in {csv_filepath}. Available columns: {df.columns.tolist()}")
        return

    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['elapsed_time_s'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    except Exception as e:
        print(f"Error processing timestamp column: {e}. Skipping plots that require time.")
        return

    output_dir = os.path.dirname(csv_filepath)
    if not output_dir:
        output_dir = '.'

    plots_generated_count = 0

    # CPU Utilization
    if 'cpu_percent' in df.columns:
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(df['elapsed_time_s'], df['cpu_percent'], label='CPU Utilization')
            plt.xlabel('Time (seconds)')
            plt.ylabel('CPU Utilization (%)')
            plt.title('CPU Utilization Over Time')
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(output_dir, 'system_cpu_utilization.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot: {plot_path}")
            plots_generated_count += 1
        except Exception as e:
            print(f"Could not generate plot for cpu_percent: {e}")
    else:
        print("Column 'cpu_percent' not found, skipping CPU plot.")

    # Memory Usage
    if 'memory_percent' in df.columns and 'memory_available_gb' in df.columns and 'memory_total_gb' in df.columns:
        try:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            color = 'tab:red'
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Memory Utilization (%)', color=color)
            ax1.plot(df['elapsed_time_s'], df['memory_percent'], color=color, label='Memory Utilization (%)')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.legend(loc='upper left')

            ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            ax2.set_ylabel('Memory Available (GB)', color=color) # we already handled the x-label with ax1
            ax2.plot(df['elapsed_time_s'], df['memory_available_gb'], color=color, linestyle='--', label='Memory Available (GB)')
            # If total memory is constant, plot it as a reference
            if df['memory_total_gb'].nunique() == 1:
                 total_mem_gb = df['memory_total_gb'].iloc[0]
                 ax2.axhline(y=total_mem_gb, color='tab:green', linestyle=':', label=f'Total Memory ({total_mem_gb:.2f} GB)')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.legend(loc='upper right')

            fig.tight_layout() # otherwise the right y-label is slightly clipped
            plt.title('System Memory Usage Over Time')
            plt.grid(True)
            plot_path = os.path.join(output_dir, 'system_memory_usage.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot: {plot_path}")
            plots_generated_count += 1
        except Exception as e:
            print(f"Could not generate plot for memory usage: {e}")
    else:
        print("Memory columns ('memory_percent', 'memory_available_gb', 'memory_total_gb') not found, skipping memory plot.")

    # Disk I/O Throughput (MB/s based on interval)
    if 'disk_read_mb_interval' in df.columns and 'disk_write_mb_interval' in df.columns:
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(df['elapsed_time_s'], df['disk_read_mb_interval'], label='Disk Read (MB/interval)')
            plt.plot(df['elapsed_time_s'], df['disk_write_mb_interval'], label='Disk Write (MB/interval)')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Data Transferred (MB per interval)')
            plt.title('Disk I/O Throughput Over Time')
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(output_dir, 'system_disk_io_throughput.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot: {plot_path}")
            plots_generated_count += 1
        except Exception as e:
            print(f"Could not generate plot for disk I/O throughput: {e}")
    else:
        print("Disk I/O columns ('disk_read_mb_interval', 'disk_write_mb_interval') not found, skipping disk I/O throughput plot.")

    # Network I/O Throughput (MB/s based on interval)
    if 'net_sent_mb_interval' in df.columns and 'net_recv_mb_interval' in df.columns:
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(df['elapsed_time_s'], df['net_sent_mb_interval'], label='Network Sent (MB/interval)')
            plt.plot(df['elapsed_time_s'], df['net_recv_mb_interval'], label='Network Received (MB/interval)')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Data Transferred (MB per interval)')
            plt.title('Network I/O Throughput Over Time')
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(output_dir, 'system_network_io_throughput.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot: {plot_path}")
            plots_generated_count += 1
        except Exception as e:
            print(f"Could not generate plot for network I/O throughput: {e}")
    else:
        print("Network I/O columns ('net_sent_mb_interval', 'net_recv_mb_interval') not found, skipping network I/O throughput plot.")

    if plots_generated_count == 0:
        print(f"No system metrics plots were generated. Check CSV content and column names. Available columns: {df.columns.tolist()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot system metrics from a CSV log file.')
    parser.add_argument('csv_file', type=str, help='Path to the system metrics CSV log file.')
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: The file '{args.csv_file}' does not exist.")
    elif os.path.getsize(args.csv_file) == 0:
        print(f"Error: The file '{args.csv_file}' is empty.")
    else:
        plot_system_metrics(args.csv_file)
