"""
Back-calculates the PRE distances from PDB structure file.

Uses idpconfgen libraries for coordinate parsing as it's proven
to be faster than BioPython.

USAGE:
    $ spycipdb jcbc <PDB-FILES>
    $ spycipdb jcbc <PDB-FILES> [--output] [--ncores]
    
REQUIREMENTS:
    Experimental data must be comma-delimited with at least the following columns:
    
    resnum
    
    Where `resnum` indicates the JC for a specific residue.

OUTPUT:
    Output is in standard .JSON format as follows, with `jc_values`
    being for each residue aligned with `resnum` format.
    
    {
        'format': [resnum],
        'pdb1': [jc_values],
        'pdb2': [jc_values],
        ...
    }
"""
import numpy as np
import pandas as pd
import json
import argparse
import shutil
from pathlib import Path
from functools import partial

from spycipdb import log
from spycipdb.libs import libcli
from spycipdb.logger import S, T, init_files, report_on_crash

from idpconfgen.libs.libio import extract_from_tar, read_path_bundle
from idpconfgen.libs.libmulticore import pool_function
from idpconfgen.libs.libcalc import calc_torsion_angles

LOGFILESNAME = '.spycipdb_jc'
_name = 'jc'
_help = 'J-Coupling back-calculator given optional experimental data template.'

_prog, _des, _usage = libcli.parse_doc_params(__doc__)

ap = libcli.CustomParser(
    prog=_prog,
    description=libcli.detailed.format(_des),
    usage=_usage,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    )

libcli.add_argument_pdb_files(ap)
libcli.add_argument_exp_file(ap)
libcli.add_argument_output(ap)
libcli.add_argument_ncores(ap)

TMPDIR = '__tmpjc__'
ap.add_argument(
    '--tmpdir',
    help=(
        'Temporary directory to store data during calculation '
        'if needed.'
        ),
    type=Path,
    default=TMPDIR,
    )


def calc_jc(fexp, pdb):
    """
    Main logic for back-calculating JC data
    with residues of interest derived from experimental template.
    """
    exp = pd.read_csv(fexp)
    # align torsion index as the first residue doesn't have phi torsion
    resn = exp.resnum.values - 2
    
    # TODO: use logic from `get_torsions` or `calc_torsion_angles` from idpcg


def main(
        pdb_files,
        exp_file,
        output,
        ncores=1,
        tmpdir=TMPDIR,
        **kwargs,
        ):
    """
    Main logic for processing PDB structures and
    outputting back-calculatedJC values.
    
    Parameters
    ----------
    pdb_files : str or Path, required
        Path to a .TAR or folder of PDB files.
        
    exp_file : str or Path, required
        Path to experimental file template.
        Required to know for which residues to calculate.
    
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
    