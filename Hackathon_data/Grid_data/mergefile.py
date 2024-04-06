import os
import pandas as pd

# Set the working directory
os.chdir("D:\Ap Shah\Hackwave\Hackathon_data")

# Define folder path and file patterns
folder_path = "D:\Ap Shah\Hackwave\Hackathon_data\wind_power_data"
file_pattern_at = "air_temperature_"
file_pattern_pressure = "pressure_"
file_pattern_ws = "wind_speed_"
file_pattern_power = "power_gen_"

# Function to read and concatenate files
def read_and_concat_files(folder_path, file_pattern):
    files_to_append = [file for file in os.listdir(folder_path) if file.startswith(file_pattern) and file.endswith(".xlsx")]
    data_frames = [pd.read_excel(os.path.join(folder_path, file)) for file in files_to_append]
    return pd.concat(data_frames, ignore_index=True)

# Read and concatenate files for each parameter
air_temp = read_and_concat_files(folder_path, file_pattern_at)
pressure = read_and_concat_files(folder_path, file_pattern_pressure)
wind_speed = read_and_concat_files(folder_path, file_pattern_ws)
power_gen = read_and_concat_files(folder_path, file_pattern_power)[1:6]  # Exclude the first file which contains only header

# Merge data frames
master_file = pd.concat([power_gen, air_temp.iloc[:, 1], pressure.iloc[:, 1], wind_speed.iloc[:, 1]], axis=1)

# Write to CSV
master_file.to_csv("merged_file2.csv", index=False)
