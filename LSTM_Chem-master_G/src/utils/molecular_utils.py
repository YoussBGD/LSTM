# molecular_utils.py
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem, Descriptors, MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
import pandas as pd

def calculate_descriptors(mol):
    """Calcule les descripteurs moléculaires."""
    if mol is None:
        return None
        
    return {
        'Poids Moléculaire': Descriptors.ExactMolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'Nb Cycles': Descriptors.RingCount(mol),
        'Nb Donneurs H': Descriptors.NumHDonors(mol),
        'Nb Accepteurs H': Descriptors.NumHAcceptors(mol),
        'Nb Liens Rotatifs': Descriptors.NumRotatableBonds(mol),
        'Nb Atomes Lourds': Descriptors.HeavyAtomCount(mol),
        'Nb Carbones': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum()==6),
        'Fraction sp3': Descriptors.FractionCSP3(mol)
    }

def get_available_similarity_methods():
    """Retourne la liste des méthodes de similarité disponibles avec leurs paramètres."""
    return {
        'Morgan': {
            'description': 'Morgan fingerprints (ECFP)',
            'parameters': {'radius': 'entre 2 et 5', 'nBits': '2048 par défaut'}
        },
        'AtomPairs': {
            'description': 'Fingerprints basés sur les paires d\'atomes',
            'parameters': {}
        },
        'MACCS': {
            'description': 'MACCS keys (166 bits)',
            'parameters': {}
        },
        'RDKit': {
            'description': 'Fingerprints RDKit par défaut',
            'parameters': {}
        }
    }

def check_duplicates(molecule_list):
    """
    Vérifie les doublons dans une liste de molécules.
    Retourne un dictionnaire avec les SMILES comme clés et les occurrences comme valeurs.
    """
    smiles_dict = {}
    for mol in molecule_list:
        smiles = Chem.MolToSmiles(mol, canonical=True)
        if smiles in smiles_dict:
            smiles_dict[smiles] += 1
        else:
            smiles_dict[smiles] = 1
    return smiles_dict

def find_duplicate_groups(similarities_list, similarity_threshold=0.99):
    """
    Trouve les groupes de molécules identiques ou presque identiques.
    """
    duplicate_groups = []
    used_indices = set()
    
    for i, (sim1, _, smiles1, mol1, idx1) in enumerate(similarities_list):
        if i in used_indices:
            continue
            
        group = [(sim1, i, smiles1, mol1, idx1)]
        used_indices.add(i)
        
        for j, (sim2, _, smiles2, mol2, idx2) in enumerate(similarities_list[i+1:], i+1):
            if j in used_indices:
                continue
                
            # Vérifier si les molécules sont identiques (en utilisant les SMILES canoniques)
            smiles_1 = Chem.MolToSmiles(mol1, canonical=True)
            smiles_2 = Chem.MolToSmiles(mol2, canonical=True)
            if smiles_1 == smiles_2:
                group.append((sim2, j, smiles2, mol2, idx2))
                used_indices.add(j)
                
        if len(group) > 1:
            duplicate_groups.append(group)
            
    return duplicate_groups

def calculate_similarity(mol1, mol2, method='Morgan', **kwargs):
    """
    Calcule la similarité entre deux molécules selon différentes méthodes.
    """
    try:
        if mol1 is None or mol2 is None:
            return 0.0
            
        nBits = kwargs.get('nBits', 2048)
        
        if method == 'Morgan':
            radius = kwargs.get('radius', 2)
            if not 2 <= radius <= 5:
                raise ValueError("Le rayon doit être compris entre 2 et 5")
                
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits)
            
        elif method == 'AtomPairs':
            fp1 = Pairs.GetAtomPairFingerprint(mol1)
            fp2 = Pairs.GetAtomPairFingerprint(mol2)
            
        elif method == 'MACCS':
            fp1 = MACCSkeys.GenMACCSKeys(mol1)
            fp2 = MACCSkeys.GenMACCSKeys(mol2)
            
        elif method == 'RDKit':
            fp1 = Chem.RDKFingerprint(mol1)
            fp2 = Chem.RDKFingerprint(mol2)
            
        else:
            raise ValueError(f"Méthode {method} non reconnue")
            
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
        
    except Exception as e:
        print(f"Erreur lors du calcul de similarité: {str(e)}")
        return 0.0

def calculate_fingerprints(mol, fp_type="MACCS"):
    """Calcule les fingerprints moléculaires."""
    if mol is None:
        return None
        
    arr = None
    if fp_type == "MACCS":
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((167,))
    elif fp_type == "Morgan":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
        arr = np.zeros((1024,))
    elif fp_type == "AtomPairs":
        fp = Pairs.GetAtomPairFingerprint(mol)
        arr = np.zeros((1024,))
        
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def perform_statistical_analysis(reference_desc, generated_desc_list):
    """Réalise des analyses statistiques entre la molécule de référence et les générées."""
    results = {}
    ref_values = pd.Series(reference_desc)
    gen_values = pd.DataFrame(generated_desc_list)
    
    for descriptor in ref_values.index:
        t_stat, p_value = stats.ttest_1samp(gen_values[descriptor], ref_values[descriptor])
        results[descriptor] = {
            'reference_mean': ref_values[descriptor],
            'generated_mean': gen_values[descriptor].mean(),
            'generated_std': gen_values[descriptor].std(),
            't_statistic': t_stat,
            'p_value': p_value
        }
    
    return results

def perform_pca_analysis(reference_fp, generated_fps, dimensions=2):
    """Réalise une analyse PCA sur les fingerprints."""
    all_fps = np.vstack([reference_fp.reshape(1, -1)] + [fp.reshape(1, -1) for fp in generated_fps])
    
    pca = PCA(n_components=dimensions)
    pca_result = pca.fit_transform(all_fps)
    
    ref_pca = pca_result[0]
    gen_pca = pca_result[1:]
    
    return {
        'reference': ref_pca,
        'generated': gen_pca,
        'explained_variance': pca.explained_variance_ratio_
    }

def get_molecule_counts(all_filtered_mols, duplicate_groups):
    """
    Calcule précisément le nombre de molécules avec et sans doublons.
    """
    total_avec_doublons = len(all_filtered_mols)
    nb_duplicatas = sum(len(group) - 1 for group in duplicate_groups)
    total_sans_doublons = total_avec_doublons - nb_duplicatas
    return total_avec_doublons, total_sans_doublons
