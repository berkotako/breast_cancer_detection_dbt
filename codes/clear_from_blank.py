import os

def filter_and_delete_files_by_size(directory, max_size_kb):
    # Convert max_size_kb to bytes
    max_size_bytes = max_size_kb * 1024

    # List to hold files less than max_size_kb
    small_files = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            # Get file size
            file_size = os.path.getsize(filepath)
            # Check if file size is less than max_size_bytes
            if file_size < max_size_bytes:
                small_files.append((filename, file_size))

    # Print all files and their sizes
    print("All files and their sizes:")
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_size = os.path.getsize(filepath)
            print(f"{filename}: {file_size / 1024:.2f} KB")

    # Delete files smaller than max_size_kb
    for file, size in small_files:
        os.remove(os.path.join(directory, file))
        print(f"Deleted {file}: {size / 1024:.2f} KB")

# Example usage
directory = '/content/drive/MyDrive/Praktikum/3d_dataset_slices_fcsv2_20P/'  # Replace with your directory path
max_size_kb = 76
filter_and_delete_files_by_size(directory, max_size_kb)
#Urgent! code will be delete files, before usage please check the code and make sure that you want to delete files
