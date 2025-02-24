# structure_utils.py
import os
from rdkit import Chem
from rdkit.Chem import SDWriter

def get_safe_filename(molecule_number, similarity):
    """Crée un nom de fichier sécurisé."""
    return f"molecule_{molecule_number}_sim{similarity:.2f}"

def save_2d_structures(molecules, output_dir, molecule_number, similarity):
    """Sauvegarde les molécules en 2D au format SDF."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = get_safe_filename(molecule_number, similarity)
    output_file = os.path.join(output_dir, f"{safe_name}_2D.sdf")
    
    writer = SDWriter(output_file)
    for mol in molecules:
        writer.write(mol)
    writer.close()
    
    return output_file

def generate_3d_structures(input_sdf, output_dir, molecule_number, similarity, add_hydrogens=True, ph=7.4):
    """Génère des structures 3D avec OpenBabel."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = get_safe_filename(molecule_number, similarity)
    output_file = os.path.join(output_dir, f"{safe_name}_3D.sdf")
    
    cmd = f"obabel {input_sdf} -O {output_file} --gen3d"
    if add_hydrogens:
        cmd += f" -p {ph}"
    
    status = os.system(cmd)
    if status != 0:
        raise Exception("Erreur lors de la génération des structures 3D avec OpenBabel")
    
    return output_file
