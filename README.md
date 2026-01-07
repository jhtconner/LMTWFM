# LMTWFM - a Local Medium-Term Wind Forecasting Model
❗ This project is stil a work in progress ❗ <br>
<br>
*Final year project - Brunel University of London 2025/26* <br>
<br>
A machine learning model for regional medium-term wind speed prediction (3-10 days) with physics-guided, multivariate meterological data fusion. This model extends the MFWPN architecture by integrating ClimODE inspired modules to improve forecasting of extreme winds and sudden gusts at a local scale. 

## Abstract

Accurately predicting medium-term wind speeds, especially extreme wind events, is crucial for safe and efficient wind energy operations. LMTWFM (Local Medium-Term Wind Forecasting Model) combines multivariate meteorological fusion, spatiotemporal attention, and latent dynamics to forecast wind speeds in a regional domain, providing interpretable outputs to inform energy management decisions.



## Installation

```
conda create -n wpn python=3.10
conda activate wpn
git clone https://github.com/jhtconner/LMTWFM
cd LMTWFM
pip install -r requirements.txt
```

## Overview

- `data/:` contains a test set for the northeast region of the manuscript, which can be downloaded via the link .
- `openstl/models/mfwpn.py:` contains the network architecture of this LMTWFM.
- `openstl/modules/:` contains partial modules of the LMTWFM.
- `utils/:` contains data processing files and loss calculations..
- `chkfile/:` contains weights for predicting 24-hour wind speeds in local region of your choice, all results from model are from England.
- `result/:` contains predicted wind speed results and evaluation methods.
- `config.py：`  training configs for the LMTWFM.
- `main.py:` Train the LMTWFM.
- `test.py` Test the LMTWFM.

## Data preparation
Data can be sourced from ERA5 which is an open dataset. 

## Train
After the data is ready, use the following commands to start training the model:
```
python main.py
```

## Test
We provide the test model weights and test dataset, which can be tested using the following commands after downloading:
```
python test.py
```

Note that the predictions are obtained as npy files containing u, v variables, which need to be converted to wind speed results using: 
```
cd result
python uv_to_wind.py
```
After that, we can obtain the wind speed prediction evaluation result by:
```
python evaluate.py
```
## Acknowledgments

This project combines [MFWPN](https://github.com/Zhang-zongwei/MFWPN) and [ClimODE](https://github.com/Aalto-QuML/ClimODE), I'd like to take the chance to thank the wonderful people behind both for providing a baseline for this project. 

If you have any questions or suggestions, please do not hesitate to contact me at [2329263@brunel.ac.uk](2329263@brunel.ac.uk).
