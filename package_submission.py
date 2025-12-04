import zipfile
import os
import glob

def package_project():
    # The user requested to package download and graphing scripts.
    # We also include other analysis scripts (ngram, stylo, tfidf) as they are source code 
    # required to reproduce the results, and are not derived files.
    files_to_package = [
        'download.py',
        'download.slurm',
        'create_network.py',
        'draw_network.py',
        'ngram.py',
        'stylo.py',
        'tfidf.py'
    ]
    
    zip_filename = 'submission.zip'
    
    print(f"Creating {zip_filename}...")
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for filename in files_to_package:
            if os.path.exists(filename):
                print(f"  Adding {filename}")
                zipf.write(filename)
            else:
                print(f"  Warning: {filename} not found")
                
    print("Done.")

if __name__ == "__main__":
    package_project()
