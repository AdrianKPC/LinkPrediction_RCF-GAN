import argparse


def find_value_in_txt_file(file_path, search_value):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for row_number, line in enumerate(lines, 1):
            values = line.split()
            if str(search_value) in values:
                print(f"Found '{search_value}' in row {row_number}")
                return row_number

        print(f"'{search_value}' not found in the file.")
        return None
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Search for a value in a txt file')
    parser.add_argument('file', type=str, help='Path to the txt file')
    parser.add_argument('value', type=float, help='Numerical value to search for')

    args = parser.parse_args()
    file_path = args.file
    search_value = args.value

    find_value_in_txt_file(file_path, search_value)


#python value_search.py file_to_search.txt value_to_search
