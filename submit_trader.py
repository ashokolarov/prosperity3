import argparse
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
    exclude_file = "datamodel.py"
    python_files = [
        f
        for f in os.listdir(directory_path)
        if f.endswith(".py") and exclude_file not in f
    ]

    if not python_files:
        print(f"No Python files found in {directory_path}")
        return

    print(f"Found {len(python_files)} Python files: {', '.join(python_files)}")

    # Store all import statements to avoid duplicates
    import_statements = set()

    # Store the content from each file (excluding imports)
    file_contents = []

    allowed_local_imports = ["datamodel"]
    remove_local_imports = []
    for file_name in python_files:
        file_name = os.path.basename(file_name)
        import_name = file_name.split(".")[0]

        if import_name not in allowed_local_imports:
            remove_local_imports.append(import_name)

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

    allowed_import_statements = []
    for import_statement in import_statements:
        allowed = True
        for remove_local_import in remove_local_imports:
            if remove_local_import in import_statement:
                allowed = False
        if allowed:
            allowed_import_statements.append(import_statement)

    # Write everything to the output file
    with open(os.path.join(directory_path, output_file), "w") as out_file:
        # Write a header comment
        out_file.write("# Combined Python Files\n")
        out_file.write(f"# Files combined: {', '.join(sorted(python_files))}\n\n")

        # Write all import statements sorted alphabetically
        out_file.write("# Import statements\n")
        for imp in sorted(allowed_import_statements):
            out_file.write(f"{imp}\n\n")

        # Write all file contents
        for content in file_contents:
            out_file.write(content)

    print(f"Successfully combined {len(python_files)} files into {output_file}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Combine Python files from a directory into a single file."
    )
    parser.add_argument(
        "-d",
        "--directory",
        default="src/",
        help="Directory containing Python files to combine (default: src/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="../submissions/submission.py",
        help="Output file path (default: ../submissions/submission.py)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Interactive prompt for the output file name
    output_path = (
        input(f"Enter output file name (default: {args.output}): ").strip()
        or args.output
    )

    output_path = "../submissions/" + output_path
    if output_path[-3:] != ".py":
        output_path += ".py"

    # Run the combination function
    combine_python_files(args.directory, output_path)
