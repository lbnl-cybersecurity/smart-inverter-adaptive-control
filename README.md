<img src="pycigar/docs/img/square_logo.svg" align="right" width="25%"/>

<!---
[![Test Status](https://github.com/lbnl-cybersecurity/ceds-cigar-external/workflows/Install%20and%20Test/badge.svg)](https://github.com/lbnl-cybersecurity/ceds-cigar-external/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/toanngosy/pycigar/blob/master/LICENSE.md)
-->

## DERAC

Adaptive Control Algorithm to Adjust Settings in Photovoltaic Inverters for Electric Grid Cybersecurity (DERAC)

## Installation (Using Regular Python)

```
git clone git+ssh://USERNAME@github.com/lbnl-cybersecurity/smart-inverter-adaptive-control.git
cd smart-inverter-adaptive-control
python3 -n venv pycigar
source pycigar/bin/activate
pip3 install -r requirements.txt
python3 setup.py develop
```

## Installation (Using Anaconda of Miniconda)

```
git clone git+ssh://USERNAME@github.com/lbnl-cybersecurity/smart-inverter-adaptive-control.git
cd smart-inverter-adaptive-control
conda env create -f environment.yml
conda activate pycigar
python setup.py develop
```


USERNAME must be set to a github username that has access to this private repo.

You can run the install and test via the [Dockerfile](Dockerfile) if you prefer.


## Testing your installation
------------------------

Once the above modules have been successfully installed, we can test the installation by running a few examples. Before trying to run any examples, be
sure to activate the `pycigar` environment by typing:

    - If you are using conda: `conda activate pycigar`
    - If you are using pip virtual env: `source pycigar/bin/activate`

## Experiment results
Please see in experiments folder.


Adaptive Control Algorithm to Adjust Settings in Photovoltaic Inverters for
Electric Grid Cybersecurity (DERAC) Copyright (c) 2021, The Regents of
the University of California, through Lawrence Berkeley National Laboratory
(subject to receipt of any required approvals from the U.S. Dept. of Energy). 
and Arizona State University. All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.
