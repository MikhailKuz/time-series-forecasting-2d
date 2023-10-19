import subprocess
import os

files = [
    "./data/get_config.py",
    
    "./models/convlstm/make_config.py",
    "./models/convlstm/run0.py",
    
    "./models/earthformer/make_config.py",
    "./models/earthformer/run0.py",
    
    "./models/linear/make_config.py",
    "./models/linear/run0.py",
    
    "./evaluation/get_metrics.py",
    "./evaluation/get_summary.py",
    
]

for file in files:
    file_name = os.path.basename(file)
    base_dir = file[:-len(file_name)]
    subprocess.call(['python3', file_name],
                    cwd=os.path.abspath(base_dir))
