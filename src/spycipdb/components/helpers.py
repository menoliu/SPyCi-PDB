"""Help SPyCi-PDB interact with third-party programs."""
import os
import subprocess
import sys

import pandas as pd

from spycipdb.components.hullrad import Sved, model_from_pdb
from DEERPREdict.PRE import PREpredict
from MDAnalysis import Universe

# Interesting way to import from repository that cannot be
# installed as a module ;-)
# https://www.geeksforgeeks.org/python-import-module-from-different-directory/
current_file_path = os.path.realpath(__file__)
curr_fp_split = current_file_path.split('/')
cspred_fp = ""
for item in curr_fp_split:
    if item == "SPyCi-PDB":
        cspred_fp += "CSpred"
        break
    else:
        cspred_fp += item + "/"
# Assumes CSpred is correctly installed in the directory housing SPyCi-PDB
sys.path.insert(0, cspred_fp)
try:
    from CSpred import calc_sing_pdb  # noqa: E402, F401
except ModuleNotFoundError:
    # Error message handled in `cli_cs.py`
    pass


# obtaining absolute path of pales executable from current file path
current_file_path = os.path.realpath(__file__)
curr_fp_split = current_file_path.split('/')
PALES_FP = ""
for item in curr_fp_split:
    if item == "SPyCi-PDB":
        PALES_FP += item + "/thirdparty/pales/linux/pales"
        break
    else:
        PALES_FP += item + "/"
        

def pales_helper(exp, pdb_path):
    """
    Handle external PALES shell command.

    Parameters
    ----------
    exp : str
        Absolute path of experimental file formatted per PALES
        standard.
    
    pdb_path : str
        Absolute path of PDB file.

    Returns
    -------
    format : dict
        Format of what atoms from which residues are
        back-calculated
    
    pdb_name_ext : str
        PDB file name with extension.
    
    rdc_bc : list
        List of RDC values. Formatting is given already.
    """
    rdc_bc = []
    format = {
        "resnum1": [],
        "resname1": [],
        "atomname1": [],
        "resnum2": [],
        "resname2": [],
        "atomname2": [],
        }
    pdb_name_ext = pdb_path.rsplit('/', 1)[-1]
    outpath = pdb_name_ext + ".txt"
    
    subprocess.run(
        f"{PALES_FP} -inD {exp} -pdb {pdb_path} -outD {outpath}",
        shell=True,
        capture_output=True,
        )
    
    with open(outpath, 'r') as pales_out:
        for line in pales_out:
            linesplit = line.split()
            try:
                if linesplit[0].isdigit():
                    format['resnum1'].append(int(linesplit[0]))
                    format['resname1'].append(linesplit[1])
                    format['atomname1'].append(linesplit[2])
                    format['resnum2'].append(int(linesplit[3]))
                    format['resname2'].append(linesplit[4])
                    format['atomname2'].append(linesplit[5])
                    rdc_bc.append(float(linesplit[8]))
            except IndexError:
                continue
    
    os.remove(outpath)
    
    return format, pdb_name_ext, rdc_bc


def hullrad_helper(pdb_path):
    """Return translational hydrodynamic radius given PDB."""
    pdb_name_ext = pdb_path.rsplit('/', 1)[-1]
    
    all_atm_rec, num_MG, num_MN, model_array = model_from_pdb(pdb_path)
    
    s, Dt, Dr, vbar_prot, Rht, ffo_hyd_P, M, Ro, Rhr, int_vis, a_b_ratio, \
        Ft, Rg, Dmax, tauC, asphr, AA, NA, GL, DT, useNumpy \
        = Sved(all_atm_rec, num_MG, num_MN, model_array)

    return pdb_name_ext, Rht


def crysol_helper(pdb_path, lm):
    """
    Handle external crysol shell command.

    Parameters
    ----------
    pdb_path : str
        Absolute path of PDB file.
    
    lm : int
        Maximum order of harmonics used for CRYSOL

    Returns
    -------
    pdb_name_ext : str
        PDB file name with extension.
    
    saxs_bc : dict
        Dictionary of index and values for each back-calculation.
    """
    saxs_bc = {}
    index = []
    value = []
    
    wrkdir = os.getcwd()
    pdb_name_ext = pdb_path.rsplit('/', 1)[-1]
    pdb_name = pdb_name_ext[0: pdb_name_ext.index('.')]
    paths = wrkdir + "/" + pdb_name
    
    p = subprocess.Popen(
        f"crysol {pdb_path} --lm={lm} --shell=water",
        stdout=subprocess.PIPE,
        shell=True,
        )
    p.communicate()  # waits for subprocess to stop running
    
    with open(paths + ".abs", mode='r') as crysol_out:
        data = crysol_out.readlines()
        data.pop(0)
        for line in data:
            splitted = line.split()
            index.append(float(splitted[0]))
            value.append(float(splitted[1]))
        
    saxs_bc['index'] = index
    saxs_bc['value'] = value
    
    # removing crysol generated files
    os.remove(paths + ".abs")
    os.remove(paths + ".alm")
    os.remove(paths + ".log")
    os.remove(paths + ".int")
    
    return pdb_name_ext, saxs_bc


def deerpredict_helper(
        fexp,
        residue,
        temperature,
        atom_selection,
        tau_c,
        wh,
        delay,
        r_2,
        pdb_path,
        ):
    """
    Handle PREpredict back-calculation from DEERPREdict

    Parameters
    ----------
    fexp : str or Path
        Path to the PRE experimental data file formatted with
        res1,atom1,res2,atom2.
    
    tau_c : float
        Rotational tumbling time.
        Defaults to 1.0e-9.

    wh : float
        Proton Larmor frequency / (2 pi 1e6).
        Defaults to 700.0.
    
    delay : float
        Inept delay depending on pulse sequence.
        Defaults to 10.0e-3.
        
    r_2 : float
        R2 spin value. Equal to R2,ox - R2,red.
    
    pdb_paths : list
        List of paths to PDBs of interest.
    
    Returns
    -------
    residues : list
        List of the residues as float.
    
    intensity_ratios : list
        List of intensity ratios as float.
    
    pre_rates : list
        List of PRE rates in Hz as float.
    """
    # TODO: see how to get PRE information of just
    # one conformer. right now you need an ensemble for this
    # and reweighting is done by BME...
    cwd = os.getcwd()
    pdb_name = os.path.basename(pdb_path[0])
    log_file = '.log'
    u = Universe(pdb_path[0], pdb_path)
    pre_predict = PREpredict(
        u,
        residue=residue,
        log_file=log_file,
        temperature=temperature,
        atom_selection=atom_selection
        )
    pre_predict.run(
        output_prefix=pdb_name,
        tau_c=tau_c,
        delay=delay,
        r_2=r_2,
        wh=wh,
        )
    
    os.remove(cwd + f"/{pdb_name}-{residue}.pkl")
    os.remove(cwd + f"/{pdb_name}-Z-{residue}.dat")
    os.remove(cwd + f"/.log")
    output = cwd + f"/{pdb_name}-{residue}.dat"
    output_df = pd.read_csv(output, delimiter=' ')
    
    residues = output_df.iloc[:, 0].tolist()
    intensity_ratios = output_df.iloc[:, 1].tolist()
    pre_rates = output_df.iloc[:, 2].tolist()
    
    return residues, intensity_ratios, pre_rates
    