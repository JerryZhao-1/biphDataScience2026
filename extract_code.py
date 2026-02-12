import json
import sys

def extract(nb_path, out_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(out_path, 'w', encoding='utf-8') as out:
        for i, cell in enumerate(nb['cells']):
            ct = cell['cell_type']
            src = ''.join(cell['source'])
            out.write(f"{'='*60}\n")
            out.write(f"Cell {i} [{ct}]\n")
            out.write(f"{'='*60}\n")
            out.write(src)
            out.write("\n\n")

extract(r'd:\A\Warton\JupyterNotebook\Processing.ipynb', r'd:\A\Warton\processing_code.txt')
extract(r'd:\A\Warton\JupyterNotebook\GridSearch.ipynb', r'd:\A\Warton\gridsearch_code.txt')
extract(r'd:\A\Warton\JupyterNotebook\Phase1a_Consolidated.ipynb', r'd:\A\Warton\gridseaPhase1a_Consolidated.txt')
print("Done")
