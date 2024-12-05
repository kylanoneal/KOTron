import os

# Define the directory containing your files
directory = "./game_data/20240725_870k_mediocre"

# Get a list of all files in the directory
files = sorted(list(os.listdir(directory)))

# Determine the number of digits needed for zero padding


# Rename each file with zero padding
for filename in files:
    # Extract the base number from the filename

    num_string = filename.replace(".json", "")
    num_string = num_string.replace("game_data_", "")

    base_number = int(num_string)

    # Create the new filename with zero padding
    new_filename = f'game_data_{base_number:04}.json'

    # Construct full file paths
    old_file = os.path.join(directory, filename)
    new_file = os.path.join(directory, new_filename)

    # Rename the file
    os.rename(old_file, new_file)
    print(f'Renamed {old_file} to {new_file}')

print('Renaming complete.')
