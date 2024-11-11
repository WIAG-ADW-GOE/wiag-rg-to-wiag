import csv
import os

import re
import os

def split_sql_dump(dump_file_path, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Regular expressions to match table creation and insertion
    create_table_re = re.compile(r"(?i)^CREATE TABLE `?(\w+)`?", re.MULTILINE)
    insert_into_re = re.compile(r"(?i)^INSERT INTO `?(\w+)`?", re.MULTILINE)

    with open(dump_file_path, 'r') as dump_file:
        content = dump_file.read()

    # Split the content by semi-colon followed by a new line (end of SQL command)
    commands = content.split(";\n")

    table_files = {}

    # Iterate over each command to detect tables and write files
    for command in commands:
        create_match = create_table_re.search(command)
        insert_match = insert_into_re.search(command)

        if create_match:
            table_name = create_match.group(1)
            filename = os.path.join(output_dir, f"{table_name}_create.sql")
            table_files[table_name] = filename

            # Write the CREATE TABLE statement to its own file
            with open(filename, 'w') as table_file:
                table_file.write(command.strip() + ";\n")

        elif insert_match:
            table_name = insert_match.group(1)
            filename = table_files.get(table_name, os.path.join(output_dir, f"{table_name}_data.sql"))

            # Append the INSERT INTO statement to the corresponding file
            with open(filename, 'a') as table_file:
                table_file.write(command.strip() + ";\n")

    print(f"SQL dump has been split into individual files in '{output_dir}'.")

# Usage
split_sql_dump('../web1064.sql', '.')


def extract_value_tuples(values_part):
    tuples = []
    start = None
    depth = 0
    in_quote = False
    escape = False

    for i, char in enumerate(values_part):
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == "'":
            in_quote = not in_quote
        elif char == '(' and not in_quote:
            if depth == 0:
                start = i
            depth += 1
        elif char == ')' and not in_quote:
            depth -= 1
            if depth == 0 and start is not None:
                tuples.append(values_part[start:i+1])
                start = None
    return tuples

def parse_insert_statements(sql_content):
    # Regular expression to match INSERT statements
    insert_regex = re.compile(
        r"INSERT\s+INTO\s+`?(\w+)`?\s*(?:\((.*?)\))?\s+VALUES\s*(.*?);\n",
        re.IGNORECASE | re.DOTALL
    )

    # Find all INSERT statements
    insert_statements = insert_regex.findall(sql_content)

    data = {}

    for table_name, columns_part, values_part in insert_statements:
        # Process columns
        if columns_part:
            columns = [col.strip(' `') for col in columns_part.split(',')]
        else:
            columns = []

        # Initialize data storage for the table
        if table_name not in data:
            data[table_name] = {'columns': columns, 'rows': []}

        # Find all value tuples
        value_tuples = extract_value_tuples(values_part)

        for value_tuple in value_tuples:
            # Remove the surrounding parentheses
            value_tuple = value_tuple.strip('()')

            # Use csv module to parse the values, handling commas inside quotes
            reader = csv.reader([value_tuple], delimiter=',', quotechar="'", escapechar='\\')
            values = next(reader)

            # Clean up values
            cleaned_values = []
            for v in values:
                v = v.strip().strip("'")
                v = v.replace('"', '""')
                if v.upper() == 'NULL':
                    v = ''
                else:
                    v = v.replace("''", "'")
                cleaned_values.append(v)
            data[table_name]['rows'].append(cleaned_values)

    return data

def sql_files_to_csv(sql_dir, csv_dir):
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    for filename in os.listdir(sql_dir):
        if filename.endswith('.sql'):
            sql_file_path = os.path.join(sql_dir, filename)

            with open(sql_file_path, 'r', encoding='utf-8') as sql_file:
                sql_content = sql_file.read()

            data = parse_insert_statements(sql_content)

            for table_name, table_data in data.items():
                # Use the table name to create a unique CSV file
                csv_filename = f"{table_name}.csv"
                csv_file_path = os.path.join(csv_dir, csv_filename)

                with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)

                    # Write header if columns are available
                    if table_data['columns']:
                        writer.writerow(table_data['columns'])

                    # Write rows
                    writer.writerows(table_data['rows'])

# Replace 'data_dump' and 'csv_dump' with your actual directories
sql_files_to_csv('data_dump', 'csv_dump')
