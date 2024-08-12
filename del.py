import os

def reduce_file_size(file_path, max_size_mb):
    max_size_bytes = max_size_mb * 1024 * 1024
    with open(file_path, 'r') as file:
        lines = file.readlines()

    while os.path.getsize(file_path) > max_size_bytes and lines:
        lines.pop(0)  # 删除第一行
        with open(file_path, 'w') as file:
            file.writelines(lines)

reduce_file_size('household_power_consumption.txt', 100)
