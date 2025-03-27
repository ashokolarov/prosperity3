import os


def combine_python_files(directory_path, output_file="combined_output.py"):
    """
    Combine all Python files in the specified directory into a single file.
    Handles multi-line imports with parentheses.

    Args:
        directory_path (str): Path to the directory containing Python files
        output_file (str): Name of the output file (default: combined_output.py)
    """
    # Get all Python files in the directory
    python_files = [f for f in os.listdir(directory_path) if f.endswith(".py")]

    if not python_files:
        print(f"No Python files found in {directory_path}")
        return

    print(f"Found {len(python_files)} Python files: {', '.join(python_files)}")

    # Store all import statements to avoid duplicates
    import_statements = set()

    # Store the content from each file (excluding imports)
    file_contents = []

    # Process each file
    for file_name in sorted(python_files):
        file_path = os.path.join(directory_path, file_name)
        print(f"Processing {file_path}...")

        with open(file_path, "r") as file:
            lines = file.readlines()

        # Process lines to handle imports (including multi-line imports)
        i = 0
        file_content = []
        while i < len(lines):
            line = lines[i].strip()

            # Check if this is the start of an import statement
            if line.startswith("import ") or line.startswith("from "):
                import_statement = [line]

                # Check if this is a multi-line import (ends with open parenthesis)
                if "(" in line and ")" not in line:
                    j = i + 1
                    # Continue collecting lines until we find the closing parenthesis
                    while j < len(lines) and ")" not in lines[j]:
                        import_statement.append(lines[j].strip())
                        j += 1

                    # Add the line with the closing parenthesis
                    if j < len(lines):
                        import_statement.append(lines[j].strip())
                        i = j  # Update i to the last line of the import

                # Add the complete import statement to our set
                import_statements.add("\n".join(import_statement))
            else:
                # If not an import, add to file content
                file_content.append(lines[i])

            i += 1

        # Join the non-import content and add to file contents
        content_without_imports = "".join(file_content).strip()

        # Add a header comment indicating the source file
        formatted_content = f"\n\n# Code from {file_name}\n{content_without_imports}"
        file_contents.append(formatted_content)

    # Write everything to the output file
    with open(os.path.join(directory_path, output_file), "w") as out_file:
        # Write a header comment
        out_file.write("# Combined Python Files\n")
        out_file.write(f"# Files combined: {', '.join(sorted(python_files))}\n\n")

        # Write all import statements sorted alphabetically
        out_file.write("# Import statements\n")
        for imp in sorted(import_statements):
            out_file.write(f"{imp}\n\n")

        # Write all file contents
        for content in file_contents:
            out_file.write(content)

    print(f"Successfully combined {len(python_files)} files into {output_file}")


if __name__ == "__main__":
    # Get directory path from user input
    directory = "src/"
    output = "submission.py"

    combine_python_files(directory, output)
