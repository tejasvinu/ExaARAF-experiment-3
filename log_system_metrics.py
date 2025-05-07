import psutil
import csv
import time
import argparse
import datetime
import os

def log_system_metrics(output_file, interval):
    """Logs system metrics (CPU, memory, disk I/O, network I/O) to a CSV file."""
    fieldnames = [
        'timestamp',
        'cpu_percent',
        'memory_total_gb',
        'memory_available_gb',
        'memory_percent',
        'disk_read_mb_interval',
        'disk_write_mb_interval',
        'disk_read_count_interval',
        'disk_write_count_interval',
        'net_sent_mb_interval',
        'net_recv_mb_interval'
    ]

    print(f"Logging system metrics to {output_file} every {interval} seconds. This script will be terminated by the main job script.")
    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Initialize I/O counters before the loop
            disk_io_before = psutil.disk_io_counters()
            net_io_before = psutil.net_io_counters()

            while True:
                # psutil.cpu_percent(interval=interval) blocks for 'interval' seconds
                # and measures CPU utilization over this period.
                current_cpu_percent = psutil.cpu_percent(interval=interval)

                # Timestamp for the end of the measurement interval
                timestamp = datetime.datetime.now().isoformat()

                # Memory (current state at end of interval)
                mem = psutil.virtual_memory()
                memory_total_gb = mem.total / (1024**3)
                memory_available_gb = mem.available / (1024**3)
                memory_percent = mem.percent

                # Disk I/O (delta over the interval)
                disk_io_after = psutil.disk_io_counters()
                disk_read_mb_interval = (disk_io_after.read_bytes - disk_io_before.read_bytes) / (1024**2)
                disk_write_mb_interval = (disk_io_after.write_bytes - disk_io_before.write_bytes) / (1024**2)
                disk_read_count_interval = disk_io_after.read_count - disk_io_before.read_count
                disk_write_count_interval = disk_io_after.write_count - disk_io_before.write_count
                disk_io_before = disk_io_after # Update for next interval

                # Network I/O (delta over the interval)
                net_io_after = psutil.net_io_counters()
                net_sent_mb_interval = (net_io_after.bytes_sent - net_io_before.bytes_sent) / (1024**2)
                net_recv_mb_interval = (net_io_after.bytes_recv - net_io_before.bytes_recv) / (1024**2)
                net_io_before = net_io_after # Update for next interval

                writer.writerow({
                    'timestamp': timestamp,
                    'cpu_percent': current_cpu_percent,
                    'memory_total_gb': round(memory_total_gb, 2),
                    'memory_available_gb': round(memory_available_gb, 2),
                    'memory_percent': memory_percent,
                    'disk_read_mb_interval': round(disk_read_mb_interval, 2),
                    'disk_write_mb_interval': round(disk_write_mb_interval, 2),
                    'disk_read_count_interval': disk_read_count_interval,
                    'disk_write_count_interval': disk_write_count_interval,
                    'net_sent_mb_interval': round(net_sent_mb_interval, 2),
                    'net_recv_mb_interval': round(net_recv_mb_interval, 2)
                })
                csvfile.flush() # Ensure data is written to disk

    except KeyboardInterrupt:
        print(f"Stopping system metrics logging for {output_file} due to KeyboardInterrupt.")
    except Exception as e:
        print(f"Error during system metrics logging: {e}")
    finally:
        print(f"System metrics logging to {output_file} has finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log system metrics (CPU, memory, disk I/O, network I/O).")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path.")
    parser.add_argument("--interval", type=int, default=5, help="Logging interval in seconds.")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            # Exit if directory cannot be made, as file cannot be written
            exit(1)
            
    log_system_metrics(args.output, args.interval)
