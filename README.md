# DeCoRTAD
Repository for the paper "[DeCoRTAD: Diffusion Based Conditional Representation Learning for Online Trajectory Anomaly Detection](https://doi.org/10.3233/FAIA240810)" (ECAI 2024).

## Get Started
Requires Python 3.10
```
pip install -r requirements.txt
```
To train+evaluate the model on AIS dataset:
```
python train_evaluation_ais.py
```

To train+evaluate on Chengdu dataset:
```
python train_evaluation_chengdu.py
```

## Dataset
### Chengdu Dataset
This dataset contains taxi trajectories within the Second Ring Road of Chengdu City in China.
### [AIS Dataset](https://www.operations.amsa.gov.au/spatial/DataServices/DigitalData)
This dataset contains sea vessel traffic data from sub-areas in Australia. 
