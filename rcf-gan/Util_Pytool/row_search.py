import argparse


def read_txt_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        num_rows = len(lines)
        num_columns = len(lines[0].split())

        return num_rows, num_columns
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count rows and columns in a txt file')
    parser.add_argument('file', type=str, help='Path to the txt file')

    args = parser.parse_args()
    file_path = args.file

    result = read_txt_file(file_path)
    if result:
        num_rows, num_columns = result
        print(f"Number of rows: {num_rows}")
        print(f"Number of columns: {num_columns}")

#python3 row_search.py file_to_count.txt
