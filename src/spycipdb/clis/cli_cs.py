"""
Back-calculates Chemical Shift  distances from PDB structure file.

Uses third-party software to do so. Default is UCBShift:
https://github.com/THGLab/CSpred

Publication:
Li, J., Bennett, K. C., Liu, Y., Martin, M. V., & Head-Gordon, T. (2020).
Accurate prediction of chemical shifts for aqueous protein structure on
“Real World” data. Chemical Science, 11(12), 3180-3191.
DOI: 10.1039/C9SC06561J

Back-calculator errors as reported for H, HA, C, CA, CB, N:
0.38, 0.22, 1.31, 0.97, 1.29 and 2.16 ppm respectively.

USAGE:
    $ spycipdb cs <PDB-FILES> [--ph]
    $ spycipdb cs <PDB-FILES> [--ph] [--output] [--ncores]

REQUIREMENTS:
    Installation of UCBShift, please refer to documentation.

OUTPUT:
    Output is in standard .JSON format as follows:
    {
        'format': {'res': [], 'resname': []}
        'pdb1': {
                    'H': [],
                    'HA': [],
                    'C': [],
                    'CA': [],
                    'CB': [],
                    'N': [],
                },
        'pdb2': {
                    'H': [],
                    'HA': [],
                    'C': [],
                    'CA': [],
                    'CB': [],
                    'N': [],
                },
        ...
    }
"""
import sys
import os
import json
import argparse
import shutil
from pathlib import Path
from functools import partial

from spycipdb import log
from spycipdb.libs import libcli
from spycipdb.libs.libfuncs import get_pdb_paths
from spycipdb.logger import S, T, init_files, report_on_crash

from idpconfgen.libs.libmulticore import pool_function

# Interesting way to import from repository that cannot be
# installed as a module ;-)
# https://www.geeksforgeeks.org/python-import-module-from-different-directory/
current_file_path = os.path.realpath(__file__)
curr_fp_split = current_file_path.split('/')
cspred_fp = "/"
for item in curr_fp_split:
    if item == "SPyCi-PDB":
        cspred_fp += item + "/" + "CSpred"
        break
    else:
        cspred_fp += item + "/"
sys.path.insert(0, cspred_fp)

from CSpred import calc_sing_pdb

LOGFILESNAME = '.spycipdb_cs'
_name = 'cs'
_help = 'CS back-calculator using UCBShift.'

_prog, _des, _usage = libcli.parse_doc_params(__doc__)

ap = libcli.CustomParser(
    prog=_prog,
    description=libcli.detailed.format(_des),
    usage=_usage,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    )

libcli.add_argument_pdb_files(ap)

ap.add_argument(
    '--ph',
    help='pH value to be considered. Defaults to 5',
    type=int,
    default=5,
    )

libcli.add_argument_output(ap)
#libcli.add_argument_ncores(ap)

TMPDIR = '__tmpcs__'
ap.add_argument(
    '--tmpdir',
    help=(
        'Temporary directory to store data during calculation '
        'if needed.'
        ),
    type=Path,
    default=TMPDIR,
    )


def main(
        pdb_files,
        output,
        ph=5,
        ncores=1,
        tmpdir=TMPDIR,
        **kwargs,
        ):
    """
    Main logic for using UCBShift to predict chemical shift
    values for PDB structures and output.

    Parameters
    ----------
    pdb_files : str or Path, required
        Path to a .TAR or folder of PDB files.
        
    pH : int, optional
        pH to consider while performing back-calculation.
        Defaults to 5.
    
    output : str or Path, optional
        Where to store the back-calculated data.
        Defaults to working directory.
        
    ncores : int, optional
        The number of cores to use.
        Defaults to 1.
    
    tmpdir : str or Path, optional
        Path to the temporary directory if working with .TAR files.
        Defaults to TMPDIR.
    """
    init_files(log, LOGFILESNAME)
    
    if ph < 2 or ph > 12:
        log.info(S('WARNING: Predictions for proteins in extreme pH '
                   'conditions are likely to be erroneous. '
                   'Take prediction results at your own risk!'
                   ))
    
    log.info(T('reading input paths'))
    pdbs2operate, _istarfile = get_pdb_paths(pdb_files, tmpdir)
    str_pdbpaths = [str(path) for path in pdbs2operate]
    log.info(S('done'))
    
    log.info(T(f'back calculaing using {ncores} workers'))
    
    # TODO: multiprocessing not working because UCBSHift is iterative
    # e.g. worker 1 needs blast/ dir but worker 3 has already removed it...
    '''
    execute = partial(
        report_on_crash,
        calc_sing_pdb,
        pH=ph,
        )
    execute_pool = pool_function(execute, str_pdbpaths, ncores=ncores)
    '''
    
    _output = {}
    for path in str_pdbpaths:
        name = path.rsplit('/',1)[-1]
        log.info(S(f'Processing {name}...'))
        per_struct = {}
        format = {}
        preds = calc_sing_pdb(path, pH=ph)
        
        format['res'] = preds.RESNUM.values.astype(int).tolist()
        format['resname'] = preds.RESNAME.values.tolist()
        
        per_struct['H'] = preds.H_UCBShift.values.astype(float).tolist()
        per_struct['HA'] = preds.HA_UCBShift.values.astype(float).tolist()
        per_struct['C'] = preds.C_UCBShift.values.astype(float).tolist()
        per_struct['CA'] = preds.CA_UCBShift.values.astype(float).tolist()
        per_struct['CB'] = preds.CB_UCBShift.values.astype(float).tolist()
        per_struct['N'] = preds.CB_UCBShift.values.astype(float).tolist()
        
        _output['format'] = format
        _output[name] = per_struct
        log.info(S('done'))
    log.info(S('done'))
    
    log.info(T('Writing output onto disk'))
    with open(output, mode="w") as fout:
        fout.write(json.dumps(_output, indent=4))
    log.info(S('done'))
    
    if _istarfile:
        shutil.rmtree(tmpdir)

    return


if __name__ == '__main__':
    libcli.maincli(ap, main)