import glob
import os
import shutil

if __name__ == "__main__":
    outfilename = "submission.py"
    with open(outfilename, "wb") as outfile:
        for filename in glob.glob("*.py"):
            if filename == outfilename or filename == "submit_trader.py":
                # don't want to copy the output into the output
                continue
            with open(filename, "rb") as readfile:
                shutil.copyfileobj(readfile, outfile)

    size_kb = os.path.getsize(outfilename) / 1024
    print(f"File size: {size_kb:.2f} KB")
