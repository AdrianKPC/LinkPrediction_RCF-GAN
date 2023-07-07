import sys

def parse_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        last_line = lines[-1]
        values = last_line.strip().split()
        second_last_value = float(values[-2])
        last_value = float(values[-1])
        return second_last_value, last_value

def calculate_average(file_prefix, num_files):
    total_second_last = 0
    total_last = 0
    for i in range(1, num_files + 1):
        filename = f"{file_prefix}_{i}.txt"
        second_last, last = parse_file(filename)
        total_second_last += second_last
        total_last += last

    average_second_last = total_second_last / num_files
    average_last = total_last / num_files
    return average_second_last, average_last

if len(sys.argv) < 3:
    print("Usage: python script.py <file_prefix> <num_files>")
    sys.exit(1)

file_prefix = sys.argv[1]
num_files = int(sys.argv[2])

average_second_last, average_last = calculate_average(file_prefix, num_files)

print("Average Test Score (AUC) in the bottom row:", average_second_last)
print("Average Test Score (AP) in the bottom row:", average_last)

