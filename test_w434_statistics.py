from datetime import datetime
import re
import os
import numpy as np
import argparse
import csv
from collections import defaultdict
import signal
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import linregress

def handle_interrupt(signal, frame):
    print("\nKeyboard interrupt received. Cleaning up...")
    # Perform any global cleanup here
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, handle_interrupt)

def count_files(directory, unit_substring, extension_pattern=None):
    """
    Count the total number of files in a directory tree, optionally filtered by extensions.
    
    :param directory: The directory to count files in.
    :param extensions: List of allowed file extensions (e.g., ['.txt', '.log']). Count all if None or ["*"].
    :return: Total file count.
    """

    # Combine the unit substring and extension into a regex pattern
    # Match filenames containing the unit substring and ending with the desired extension
    regex_string = rf".*{re.escape(unit_substring)}.*{extension_pattern}$"
    # print(f"Regex string: {regex_string}")  # Print the regex string for debugging

    # Compile the regex pattern
    file_regex = re.compile(regex_string, re.IGNORECASE)

    # Initialize a list to collect matching files
    target_files = []

    for root, _, files in os.walk(directory):
        # Filter files matching the regex pattern
        target_files.extend([f for f in files if file_regex.match(f)])

    total_files = len(target_files)

    return total_files

def normalize_extensions(extensions):
    """
    Normalize extensions to ensure they start with a dot.
    """
    if not extensions:
        return []
    return [ext if ext.startswith(".") else f".{ext}" for ext in extensions]

def extensions_to_regex(extensions):
    """
    Convert a list of file extensions (with leading dots) into a regex-compatible pattern.
    
    Args:
        extensions (list): List of file extensions (e.g., ['.txt', '.log', '.csv']).
    
    Returns:
        str: A regex pattern that matches any of the specified extensions.
    """
    # Remove the leading dots and escape the extensions for regex compatibility
    escaped_extensions = [re.escape(ext) for ext in extensions]
    # Combine extensions into a single regex pattern using alternation (|)
    return r"(?:" + "|".join(escaped_extensions) + r")"

def parse_resistance(resistance_str):
    # Parses resistance value from a string with units and returns it as a float in mOhm.
    resistance_str = resistance_str.strip()

    # Handles out-of-range values prefixed with '>' or '<'.
    resistance_str = resistance_str.strip()
    is_greater = resistance_str.startswith(">")
    is_less = resistance_str.startswith("<")
    
    # Remove '>' or '<' if they are present
    if is_greater or is_less:
        resistance_str = resistance_str[1:].strip()

    if resistance_str.endswith("mOhm"):  # milliohm
        return float(resistance_str.replace("mOhm", "").strip())
    elif resistance_str.endswith("kOhm"):  # kilo-ohm
        return float(resistance_str.replace("kOhm", "").strip()) * 1e6
    elif resistance_str.endswith("MOhm"):  # mega-ohm
        return float(resistance_str.replace("MOhm", "").strip()) * 1e9
    elif resistance_str.endswith("GOhm"):  # giga-ohm
        return float(resistance_str.replace("GOhm", "").strip()) * 1e12
    elif resistance_str.endswith("Ohm"):  # Ohm
        return float(resistance_str.replace("Ohm", "").strip()) * 1000
    else:
        return None  # Unknown unit

def convert_threshold_to_mohm(threshold_str):
    """Converts a threshold specified in various units to mOhm."""
    threshold_str = threshold_str.strip()
    if threshold_str.endswith("mOhm"):
        return float(threshold_str.replace("mOhm", "").strip())
    elif threshold_str.endswith("Ohm"):
        return float(threshold_str.replace("Ohm", "").strip()) * 1000
    elif threshold_str.endswith("kOhm"):
        return float(threshold_str.replace("kOhm", "").strip()) * 1e6
    elif threshold_str.endswith("MOhm"):
        return float(threshold_str.replace("MOhm", "").strip()) * 1e9
    elif threshold_str.endswith("GOhm"):
        return float(threshold_str.replace("GOhm", "").strip()) * 1e12
    else:
        raise ValueError("Unsupported threshold unit. Use mOhm, Ohm, kOhm, MOhm, or GOhm.")
    
def convert_to_microamps(current_str):
    """Convert current (given as a string with units) to microamps."""
    match = re.match(r"([\d.]+)([uAmkA]*)", current_str.strip())
    if not match:
        raise ValueError("Invalid current format")
    
    value, unit = match.groups()
    value = float(value)
    
    if unit == "uA":       # microamps
        return value
    elif unit == "mA":     # milliamps
        return value * 1e3
    elif unit == "A":      # amps
        return value * 1e6
    elif unit == "kA":     # kiloamps
        return value * 1e9
    else:
        return value       # default to microamps if no unit

def extract_datetime(line):
    """
    Extracts date and time from a line with the format 'Date/Time: 7/24/2024 12:41:26 PM'.

    Args:
        line (str): The line containing the date and time.

    Returns:
        datetime: A datetime object representing the extracted date and time, or None if not found.
    """
    # Regex pattern to match the date and time format
    pattern = r"Date/Time: (\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} (?:AM|PM))"
    match = re.search(pattern, line)
    
    if match:
        datetime_str = match.group(1)
        # Parse the date and time using the matched string
        return datetime.strptime(datetime_str, "%m/%d/%Y %I:%M:%S %p")
    else:
        return None

def should_exclude(parts):
    # Checks if the line should be excluded based on specific substrings in parts[1].
    excluded_substrings = {"NCA", "NCG HV", "NC HV", "NCL LV", "NCL HV", "NCA HV"}
    return any(substring in parts[1] for substring in excluded_substrings)

def extract_resistance(line):
    from_pin = ""
    to_pin = ""
    resistance = ""

    # Split on two or more spaces
    parts = re.split(r"\s{2,}", line)

    if len(parts) < 4:
        return None
    
    if should_exclude(parts):
        return None
    
    if "NCL" in parts[0].strip().upper():
        return None
    
    if len(parts) == 4:
        from_pin = parts[2]
        to_pin = "N/A"
        resistance = parse_resistance(parts[3])
    elif len(parts) == 5:
        from_pin = parts[2]
        to_pin = parts[3]
        resistance = parse_resistance(parts[4])

    return {"from_pin": from_pin, "to_pin": to_pin, "resistance": resistance}

    
def extract_db(line):
    # Parse a line to extract 'from pin', 'to pin', 'Ir', and 'Ii' with units converted to microamps.

    # Split on two or more spaces
    parts = re.split(r"\s{2,}", line)

    if len(parts) < 3:
        return None
        
    # Ensure the second column contains "DB" as a substring
    if "DB" not in parts[1]:
        return None

    if len(parts) == 4:
        from_pin = parts[2]
        to_pin = "N/A"
    
    if len(parts) == 5:
        from_pin = parts[2]
        to_pin = parts[3]
    
    # Use regex to find Ir and Ii values
    ir_match = re.search(r"Ir=([\d.]+[uAmkA]*)", line)
    ii_match = re.search(r"Ii=([\d.]+[uAmkA]*)", line)
    
    # Extract and convert real current (Ir)
    if ir_match:
        ir_value = convert_to_microamps(ir_match.group(1))
    else:
        ir_value = None

    # Extract and convert imaginary current (Ii)
    if ii_match:
        ii_value = convert_to_microamps(ii_match.group(1))
    else:
        ii_value = None

    return {
        "from_pin": from_pin,
        "to_pin": to_pin,
        "Ir_uA": ir_value,
        "Ii_uA": ii_value
    }

def process_file_content(file_content):
    """Process a list of lines to extract pin and current information for lines containing 'DB' in column 2."""
    results_db = []
    results_continuity = []

    for line in file_content:
        line_upper = line.upper()

        if "DATE/TIME" in line_upper:
            timestamp = extract_datetime(line)
        
        if "DB" in line:
            line_data = extract_db(line)
            if line_data:
                results_db.append(line_data)

        if "OHM" in line_upper:
            line_data = extract_resistance(line)

            if line_data:
                results_continuity.append(line_data)

        
    return timestamp, results_continuity, results_db

def calculate_statistics(data_array):
    """Calculate statistics including min, max, mean, median, standard deviation, and count of outliers."""
    if data_array.size == 0:
        return None, None, None, None, None, 0  # Return None for std dev and 0 for outliers if the array is empty
    
    min_val = np.min(data_array)
    mean_val = np.mean(data_array)
    max_val = np.max(data_array)
    median_val = np.median(data_array)
    std_dev = np.std(data_array)  # Calculate standard deviation
    
    # Calculate outliers using IQR
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers_count = np.sum((data_array < lower_bound) | (data_array > upper_bound))
    
    return min_val, mean_val, max_val, median_val, std_dev, outliers_count


def process_directory(directory_path, unit_substring, extensions):
    """Process all files in a directory and its subdirectories that contain the specified unit substring and have the specified extension."""
    aggregated_db_data = defaultdict(lambda: {"Ir_uA": [], "Ii_uA": []})
    aggregated_cont_data = defaultdict(lambda: {"Continuity": []})
    extracted_data = []  # Collect timestamp, continuity, and dielectric data for plotting

    print(f"Unit substring:{unit_substring}")
    # Normalize extensions
    extensions = normalize_extensions(extensions)
    print(f"Extensions: {extensions}")

    extension_pattern = extensions_to_regex(extensions)
    # print(f"Regex extensions: {extension_pattern}")
   
    # Combine the unit substring and extension into a regex pattern
    # Match filenames containing the unit substring and ending with the desired extension
    file_regex = re.compile(rf".*{re.escape(unit_substring)}.*{extension_pattern}$", re.IGNORECASE)

    # Count files and initialize progress bar
    total_files = count_files(directory_path, unit_substring, extension_pattern)
    print(f"Total files: {total_files}")

    with tqdm(total=total_files, desc="Processing Files", dynamic_ncols=True, disable=not sys.stdout.isatty()) as progress_bar:
        for root, _, files in os.walk(directory_path):

            # Filter files matching the regex pattern
            target_files = [f for f in files if file_regex.match(f)]

            for filename in target_files:

                file_path = os.path.join(root, filename)
                if os.path.isfile(file_path):
                    progress_bar.update(1)

                    with open(file_path, 'r') as file:
                        file_content = file.readlines()
                        
                        if any("Unit under test".upper() in line.upper() and unit_substring.upper() in line.upper() for line in file_content):
                            timestamp, continuity, dielectric = process_file_content(file_content)

                            if timestamp:
                                extracted_data.append((timestamp, continuity, dielectric))  # Collect raw data

                            for entry in dielectric:
                                key = (entry["from_pin"], entry["to_pin"])
                                if entry["Ir_uA"] is not None:
                                    aggregated_db_data[key]["Ir_uA"].append(entry["Ir_uA"])
                                if entry["Ii_uA"] is not None:
                                    aggregated_db_data[key]["Ii_uA"].append(entry["Ii_uA"])

                            for entry in continuity:
                                key = (entry["from_pin"], entry["to_pin"])

                                if entry["resistance"] is not None:
                                    aggregated_cont_data[key]["Continuity"].append(entry["resistance"])

    stats_db = []
    for key, values in aggregated_db_data.items():
        from_pin, to_pin = key
        ir_array = np.array(values["Ir_uA"]) if values["Ir_uA"] else np.array([])
        ii_array = np.array(values["Ii_uA"]) if values["Ii_uA"] else np.array([])
        
        # Calculate statistics for Ir and Ii arrays
        ir_stats = calculate_statistics(ir_array)
        ii_stats = calculate_statistics(ii_array)

        stats_db.append([
            from_pin,
            to_pin,
            ir_array.size,
            *ir_stats[:4],  # min, mean, max, median for Ir
            ir_stats[4],    # standard deviation for Ir
            ir_stats[5],    # number of outliers for Ir
            ii_array.size,
            *ii_stats[:4],  # min, mean, max, median for Ii
            ii_stats[4],    # standard deviation for Ii
            ii_stats[5]     # number of outliers for Ii
        ])


    stats_cont = []
    for key, values in aggregated_cont_data.items():
        from_pin, to_pin = key
        cont_array = np.array(values["Continuity"]) if values["Continuity"] else np.array([])

        # Calculate statistics for Continuity array
        cont_stats = calculate_statistics(cont_array)

        stats_cont.append([
            from_pin,
            to_pin,
            cont_array.size,
            *cont_stats[:4],  # min, mean, max, median for Continuity
            cont_stats[4],    # standard deviation for Continuity
            cont_stats[5]     # number of outliers for Continuity
        ])


    return extracted_data, stats_cont, stats_db

def remove_outliers(timestamps, ii_values):
    """
    Remove outliers from the data using the IQR method.

    Args:
        timestamps (np.ndarray): Array of timestamps (in seconds).
        ii_values (np.ndarray): Array of Ii values.

    Returns:
        np.ndarray, np.ndarray: Cleaned timestamps and Ii values.
    """

    # Filter out None or non-numeric values
    valid_indices = [i for i, val in enumerate(ii_values) if isinstance(val, (int, float))]
    timestamps = np.array(timestamps)[valid_indices]
    ii_values = np.array(ii_values)[valid_indices]

    q1, q3 = np.percentile(ii_values, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 2.5 * iqr
    upper_bound = q3 + 2.5 * iqr

    mask = (ii_values >= lower_bound) & (ii_values <= upper_bound)
    return timestamps[mask], ii_values[mask]
    
def calculate_y_limits(ii_values):
        """
        Calculate suitable Y-axis limits using IQR to exclude extreme outliers.

        Args:
            ii_values (list): List of all Ii values.

        Returns:
            tuple: (y_min, y_max) suitable for plotting.
        """

        # Remove NaN values
        ii_values = np.array(ii_values, dtype=float)
        ii_values = ii_values[~np.isnan(ii_values)]

        if len(ii_values) == 0:
            return None, None  # Return None if all values are NaN

        q1, q3 = np.percentile(ii_values, [25, 75])
        iqr = q3 - q1
        # print(f"Incoming ii_values: {ii_values}")
        # print(f"q1: {q1}, q3: {q3}, iqr: {iqr}")

        lower_bound = max(0, q1 - 1.5 * iqr)  # Lower bound should not be negative
        upper_bound = q3 + 1.5 * iqr
        return lower_bound, upper_bound

def plot_largest_ii_pins(extracted_data, stats_db, x, output_file):
    """
    Plot the X largest dielectric breakdown pins based on mean imaginary leakage (Ii) over time.

    Args:
        extracted_data (list): List of tuples containing timestamp, continuity, and dielectric data.
        stats_db (list): Processed dielectric breakdown data containing statistics (list of lists).
        x (int): Number of largest pins to plot based on mean Ii.
        output_file (str): Base name for the output plot files (e.g., 'output').
    """

    y_margin=0.1
    
    # Filter and sort the data by mean Ii (index 9 in stats_db)
    ii_data = [
        {
            "from_pin": entry[0],
            "to_pin": entry[1],
            "mean_ii": entry[11],  # Mean Ii
            "pin_key": (entry[0], entry[1]),  # Key to find data in extracted_data
        }
        for entry in stats_db if entry[11] is not None
    ]

    # Sort by mean Ii in descending order and select the top X entries
    ii_data_sorted = sorted(ii_data, key=lambda d: d["mean_ii"], reverse=True)[:x]

    # Group data for plotting
    pin_grouped_data = defaultdict(lambda: {"timestamps": [], "ii_values": []})
    for pin_entry in ii_data_sorted:
        for timestamp, continuity, dielectric in extracted_data:
            for die in dielectric:
                if (die["from_pin"], die["to_pin"]) == pin_entry["pin_key"]:
                    pin_grouped_data[pin_entry["pin_key"]]["timestamps"].append(timestamp)
                    pin_grouped_data[pin_entry["pin_key"]]["ii_values"].append(die["Ii_uA"])

    # Sort timestamps within each pin group
    for key in pin_grouped_data:
        sorted_data = sorted(zip(pin_grouped_data[key]["timestamps"], pin_grouped_data[key]["ii_values"]))
        pin_grouped_data[key]["timestamps"], pin_grouped_data[key]["ii_values"] = zip(*sorted_data)

    # Determine per-plot and global Y-axis limits
    local_y_limits = []
    for data in pin_grouped_data.values():
        ii_values = np.array(data["ii_values"], dtype=float)
        local_y_min, local_y_max = calculate_y_limits(ii_values)
        # print(f"Local min: {local_y_min}, max: {local_y_max}")

        if local_y_min is not None and local_y_max is not None:
            local_y_limits.append((local_y_min, local_y_max))

    if not local_y_limits:
        print("No valid data for plotting. Exiting.")
        return  # Exit if there are no valid data points

    global_y_min = min(limit[0] for limit in local_y_limits)
    global_y_max = max(limit[1] for limit in local_y_limits)

    # Add margin to global limits
    global_y_min -= y_margin * (global_y_max - global_y_min)
    global_y_max += y_margin * (global_y_max - global_y_min)

    # print(f"Global min: {global_y_min}, max: {global_y_max}")

    if not np.isfinite(global_y_min) or not np.isfinite(global_y_max):
        print("Global Y-axis limits are invalid. Exiting.")
        return  # Exit if the limits are not finite

    # Plotting
    plt.figure(figsize=(12, 8), constrained_layout=True)

    # Linear plot
    for pin_key, data in pin_grouped_data.items():
        label = f"{pin_key[0]} -> {pin_key[1]} ({len(data['timestamps'])} points)"
        timestamps = np.array([(ts - data["timestamps"][0]).total_seconds() for ts in data["timestamps"]], dtype=float)
        ii_values = np.array(data["ii_values"], dtype=float)

        line, = plt.plot(
            data["timestamps"],
            data["ii_values"],
            label=label,
            marker='o',  # Add markers for each data point
        )

        # Remove outliers
        clean_timestamps, clean_ii_values = remove_outliers(timestamps, ii_values)

        # Fit a quadratic regression (degree 2 polynomial) on clean data
        if len(clean_timestamps) > 2:  # Ensure there are enough points for regression
            poly_coeffs = np.polyfit(clean_timestamps, clean_ii_values, 2)  # Fit quadratic
            trend_line = np.polyval(poly_coeffs, clean_timestamps)
            plt.plot(
                [data["timestamps"][i] for i in range(len(timestamps)) if timestamps[i] in clean_timestamps],
                trend_line,
                linestyle='--',
                color=line.get_color(),  # Match the color of the dataset
                alpha=0.8,
                # label=f"{label} Quadratic Trend",
            )

    plt.title(f"Top {x} Pins by Mean Imaginary Leakage (Linear Scale)")
    plt.ylim(global_y_min, global_y_max)
    plt.xlabel("Timestamp")
    plt.ylabel("Imaginary Leakage Ii (uA)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    # plt.tight_layout()
    plt.savefig(f"{output_file}_linear.png")
    plt.close()

    # Log plot
    plt.figure(figsize=(12, 8))
    for pin_key, data in pin_grouped_data.items():
        label = f"{pin_key[0]} -> {pin_key[1]} ({len(data['timestamps'])} points)"
        timestamps = np.array([(ts - data["timestamps"][0]).total_seconds() for ts in data["timestamps"]], dtype=float)
        ii_values = np.array(data["ii_values"], dtype=float)
        line, = plt.plot(
            data["timestamps"],
            data["ii_values"],
            label=label,
            marker='o',  # Add markers for each data point
        )
        
        # Remove outliers
        clean_timestamps, clean_ii_values = remove_outliers(timestamps, ii_values)

        # Fit a quadratic regression (degree 2 polynomial) on clean data
        if len(clean_timestamps) > 2:  # Ensure there are enough points for regression
            poly_coeffs = np.polyfit(clean_timestamps, clean_ii_values, 2)  # Fit quadratic
            trend_line = np.polyval(poly_coeffs, clean_timestamps)
            plt.plot(
                [data["timestamps"][i] for i in range(len(timestamps)) if timestamps[i] in clean_timestamps],
                trend_line,
                linestyle='--',
                color=line.get_color(),  # Match the color of the dataset
                alpha=0.8,
                # label=f"{label} Quadratic Trend",
            )

    plt.yscale("log")
    plt.ylim(global_y_min, global_y_max)
    plt.title(f"Top {x} Pins by Mean Imaginary Leakage (Log Scale)")
    plt.xlabel("Timestamp")
    plt.ylabel("Imaginary Leakage Ii (uA)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=45)
    # plt.tight_layout()
    plt.savefig(f"{output_file}_log.png")
    plt.close()

def print_table(headers, rows, directory_path, unit_substring, output_filename=None, decimal_places=1):
    """
    Prints a table with the given headers and rows to the console and optionally to a file.
    
    Args:
        headers (list): List of column headers.
        rows (list of lists): List of rows, where each row is a list of values.
        output_filename (str, optional): If provided, writes output to this file.
        decimal_places (int, optional): Limits floats to this many decimal places.
    """
    # Replace None or other invalid values with empty strings and format floats
    formatted_rows = []
    for row in rows:
        formatted_row = []
        for item in row:
            if isinstance(item, float) and decimal_places is not None:
                formatted_item = f"{item:.{decimal_places}f}"
            else:
                formatted_item = str(item) if item is not None else ''
            formatted_row.append(formatted_item)
        formatted_rows.append(formatted_row)

    # Ensure all rows have the same number of columns as headers
    for row in formatted_rows:
        while len(row) < len(headers):
            row.append('')

    # Calculate the maximum width for each column
    col_widths = [max(len(str(item)) for item in col) for col in zip(*([headers] + formatted_rows))]
    
    # Create format string for headers and rows
    header_format = ' | '.join(f'{{:<{width}}}' for width in col_widths)
    row_format = ' | '.join(f'{{:<{width}}}' for width in col_widths)

    # Determine the default output filename based on the script's filename, if not provided
    if output_filename is None:
        base_filename = os.path.basename(__file__)
    else:
        base_filename = output_filename

    output_filename_txt = os.path.splitext(base_filename)[0] + '.txt'
    output_filename_csv = os.path.splitext(base_filename)[0] + '.csv'

    # Print header to console (and optionally to file if output_filename is provided)
    header_line = header_format.format(*headers)
    separator_line = '-' * (sum(col_widths) + 3 * (len(headers) - 1))
    
    # Output to console
    print(f"Statistics for directory [{directory_path}] and Unit Under Test [{unit_substring}]")
    print(header_line)
    print(separator_line)

    for row in formatted_rows:
        print(row_format.format(*row))

    # If an output file is specified, write to it as well
    if output_filename_txt:
        with open(output_filename_txt, 'w') as file:
            file.write(f"Statistics for directory [{directory_path}] and Unit Under Test [{unit_substring}]\n\n")
            file.write(header_line + '\n')
            file.write(separator_line + '\n')
            for row in formatted_rows:
                file.write(row_format.format(*row) + '\n')

    if output_filename_csv:
        # Write to CSV file
        with open(output_filename_csv, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(headers)
            csvwriter.writerows(formatted_rows)

def plot_largest_cont_pins(extracted_data, stats_cont, x, output_file):
    """
    Plot the X largest dielectric breakdown pins based on mean imaginary leakage (Ii) over time.

    Args:
        extracted_data (list): List of tuples containing timestamp, continuity, and dielectric data.
        stats_db (list): Processed dielectric breakdown data containing statistics (list of lists).
        x (int): Number of largest pins to plot based on mean Ii.
        output_file (str): Base name for the output plot files (e.g., 'output').
    """

    y_margin=0.1
    
    # Filter and sort the data by mean Ii (index 9 in stats_db)
    cont_data  = [
        {
            "from_pin": entry[0],
            "to_pin": entry[1],
            "mean_cont": entry[4],  # Mean continuity
            "pin_key": (entry[0], entry[1]),  # Key to find data in extracted_data
        }
        for entry in stats_cont if entry[4] is not None
    ]

    # Sort by mean in descending order and select the top X entries
    cont_data_sorted  = sorted(cont_data, key=lambda d: d["mean_cont"], reverse=True)[:x]

    # Group data for plotting
    pin_grouped_data = defaultdict(lambda: {"timestamps": [], "cont": []})
    for pin_entry in cont_data_sorted:
        for timestamp, continuity, dielectric in extracted_data:
            for c in continuity:
                if (c["from_pin"], c["to_pin"]) == pin_entry["pin_key"]:
                    pin_grouped_data[pin_entry["pin_key"]]["timestamps"].append(timestamp)
                    pin_grouped_data[pin_entry["pin_key"]]["cont"].append(c["resistance"])

    # Sort timestamps within each pin group
    for key in pin_grouped_data:
        sorted_data = sorted(zip(pin_grouped_data[key]["timestamps"], pin_grouped_data[key]["cont"]))
        pin_grouped_data[key]["timestamps"], pin_grouped_data[key]["cont"] = zip(*sorted_data)

    # Determine per-plot and global Y-axis limits
    local_y_limits = []
    for data in pin_grouped_data.values():
        cont_values = np.array(data["cont"], dtype=float)
        local_y_min, local_y_max = calculate_y_limits(cont_values)
        # print(f"Local min: {local_y_min}, max: {local_y_max}")

        if local_y_min is not None and local_y_max is not None:
            local_y_limits.append((local_y_min, local_y_max))

    if not local_y_limits:
        print("No valid data for plotting. Exiting.")
        return  # Exit if there are no valid data points

    global_y_min = min(limit[0] for limit in local_y_limits)
    global_y_max = max(limit[1] for limit in local_y_limits)

    # Add margin to global limits
    global_y_min -= y_margin * (global_y_max - global_y_min)
    global_y_max += y_margin * (global_y_max - global_y_min)

    # print(f"Global min: {global_y_min}, max: {global_y_max}")

    if not np.isfinite(global_y_min) or not np.isfinite(global_y_max):
        print("Global Y-axis limits are invalid. Exiting.")
        return  # Exit if the limits are not finite

    # Plotting
    plt.figure(figsize=(12, 8), constrained_layout=True)

    # Linear plot
    for pin_key, data in pin_grouped_data.items():
        label = f"{pin_key[0]} -> {pin_key[1]} ({len(data['timestamps'])} points)"
        timestamps = np.array([(ts - data["timestamps"][0]).total_seconds() for ts in data["timestamps"]], dtype=float)
        cont_values = np.array(data["cont"], dtype=float)

        line, = plt.plot(
            data["timestamps"],
            data["cont"],
            label=label,
            marker='o',  # Add markers for each data point
        )

        # Remove outliers
        clean_timestamps, clean_cont_values = remove_outliers(timestamps, cont_values)

        # Fit a quadratic regression (degree 2 polynomial) on clean data
        if len(clean_timestamps) > 2:  # Ensure there are enough points for regression
            poly_coeffs = np.polyfit(clean_timestamps, clean_cont_values, 2)  # Fit quadratic
            trend_line = np.polyval(poly_coeffs, clean_timestamps)
            plt.plot(
                [data["timestamps"][i] for i in range(len(timestamps)) if timestamps[i] in clean_timestamps],
                trend_line,
                linestyle='--',
                color=line.get_color(),  # Match the color of the dataset
                alpha=0.8,
                # label=f"{label} Quadratic Trend",
            )

    plt.title(f"Top {x} Pins by Mean Continuity (Linear Scale)")
    plt.ylim(global_y_min, global_y_max)
    plt.xlabel("Timestamp")
    plt.ylabel("Continuity (uOhms)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    # plt.tight_layout()
    plt.savefig(f"{output_file}_linear.png")
    plt.close()

    # Log plot
    plt.figure(figsize=(12, 8))
    for pin_key, data in pin_grouped_data.items():
        label = f"{pin_key[0]} -> {pin_key[1]} ({len(data['timestamps'])} points)"
        timestamps = np.array([(ts - data["timestamps"][0]).total_seconds() for ts in data["timestamps"]], dtype=float)
        cont_values = np.array(data["cont"], dtype=float)
        line, = plt.plot(
            data["timestamps"],
            data["cont"],
            label=label,
            marker='o',  # Add markers for each data point
        )
        
        # Remove outliers
        clean_timestamps, clean_cont_values = remove_outliers(timestamps, cont_values)

        # Fit a quadratic regression (degree 2 polynomial) on clean data
        if len(clean_timestamps) > 2:  # Ensure there are enough points for regression
            poly_coeffs = np.polyfit(clean_timestamps, clean_cont_values, 2)  # Fit quadratic
            trend_line = np.polyval(poly_coeffs, clean_timestamps)
            plt.plot(
                [data["timestamps"][i] for i in range(len(timestamps)) if timestamps[i] in clean_timestamps],
                trend_line,
                linestyle='--',
                color=line.get_color(),  # Match the color of the dataset
                alpha=0.8,
                # label=f"{label} Quadratic Trend",
            )

    plt.yscale("log")
    plt.ylim(global_y_min, global_y_max)
    plt.title(f"Top {x} Pins by Continuity (Log Scale)")
    plt.xlabel("Timestamp")
    plt.ylabel("Continuity (uOhms)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=45)
    # plt.tight_layout()
    plt.savefig(f"{output_file}_log.png")
    plt.close()

def plot_daily_continuity_boxplot(extracted_data, output_file):
    """
    Create a box‐and‐whisker plot of all raw continuity values, grouped by day.
    Saves the plot to `{output_file}_daily_boxplot.png`.
    """
    # group resistances by date
    daily_vals = defaultdict(list)
    for timestamp, continuity, _ in extracted_data:
        day = timestamp.date()
        for c in continuity:
            if c["resistance"] is not None:
                daily_vals[day].append(c["resistance"])

    # sort the days and collect data
    days = sorted(daily_vals)
    data = [daily_vals[day] for day in days]

    # nothing to plot?
    if not data or all(len(d)==0 for d in data):
        print("No continuity data to plot by day.")
        return

    # build the boxplot
    plt.figure(figsize=(12, 8), constrained_layout=True)
    plt.boxplot(data, tick_labels=[d.strftime("%Y-%m-%d") for d in days], showfliers=True)
    plt.title("Daily Continuity Distribution (mΩ)")
    plt.xlabel("Date")
    plt.ylabel("Continuity (mΩ)")
    plt.xticks(rotation=45)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(f"{output_file}_daily_boxplot.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Process current data files in a directory, filter by 'Unit under test' substring, and output aggregated results to a CSV file."
    )
    parser.add_argument("-d", "--directory", required=True, help="Directory containing files to process")
    parser.add_argument("-u", "--unit", required=True, help="Substring to match in 'Unit under test' lines")
    parser.add_argument("-o", "--output", required=True, help="Output file name base (no extension)")
    parser.add_argument("-e", "--extension", required=True, help="File extension to filter (e.g., .txt)")
    parser.add_argument("-p", "--decimal_places", type=int, default=2, help="Number of decimal places for float values")
    parser.add_argument("-n", "--num", type=int, default=3, help="Number of highest mean imaginary leakage pins to plot")
    
    args = parser.parse_args()

    directory_path = args.directory
    unit_substring = args.unit
    output_file = args.output

    if not os.path.isdir(directory_path):
        print("Provided path is not a directory")
        sys.exit(1)

    # Determine the default output filename based on the script's filename, if not provided
    if output_file is None:
        base_filename = os.path.basename(__file__)
    else:
        base_filename = output_file

    output_filename_db = os.path.splitext(base_filename)[0] + '_db'

    output_filename_cont = os.path.splitext(base_filename)[0] + '_continuity'

    extracted_data, stats_cont, stats_db = process_directory(directory_path, unit_substring, [args.extension])
    
    print("\n")

    headers_db = [
        "From Pin", "To Pin",
        "Ir Count", "Ir Min (uA)", "Ir Mean (uA)", "Ir Max (uA)", "Ir Median (uA)", "Ir Std Dev (uA)", "Ir Outliers (n)",
        "Ii Count", "Ii Min (uA)", "Ii Mean (uA)", "Ii Max (uA)", "Ii Median (uA)", "Ii Std Dev (uA)", "Ii Outliers (n)"
    ]

    if stats_db:
        print_table(headers_db, stats_db, directory_path, unit_substring, output_filename_db, args.decimal_places)
        print("\n")
        print(f"Data written to {output_filename_db}.txt and {output_filename_db}.csv")
        print("\n")

    headers_cont = [
        "From Pin", "To Pin",
        "Continuity Count", "Continuity Min (mOhms)", "Continuity Mean (mOhms)",
        "Continuity Max (mOhms)", "Continuity Median (mOhms)", "Continuity Std Dev (mOhms)", "Continuity Outliers (n)"
    ]  

    if stats_cont:
        print_table(headers_cont, stats_cont, directory_path, unit_substring, output_filename_cont, args.decimal_places)
        print("\n")
        print(f"Data written to {output_filename_cont}.txt and {output_filename_cont}.csv")
        print("\n")

    
    # Plot the top 5 dielectric breakdown pins by mean Ii
    plot_largest_ii_pins(extracted_data, stats_db, args.num, output_file=output_filename_db)

    plot_largest_cont_pins(extracted_data, stats_cont, args.num, output_file=output_filename_cont)
    print(f"Continuity plots saved as {output_filename_cont}_linear.png and {output_filename_cont}_log.png")

    plot_daily_continuity_boxplot(extracted_data, output_filename_cont)
    print(f"Daily continuity box‐and‐whisker plot saved as {output_filename_cont}_daily_boxplot.png")

if __name__ == "__main__":
    main()
