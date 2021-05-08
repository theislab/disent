## Out-of-distribution prediction with disentangled representations for single-cell RNA sequencing data

The pdf for the paper can be found [here](https://github.com/theislab/disent/raw/master/paper.pdf).


### Usage

Reproducing results from Jupyter Notebooks:
* JN1_train.ipynb - loads, normalizes and trains the data
* JN2_kang_analysis.ipynb - further analyses the trained model for Kang dataset. It calculates both types of disentanglement scores and also visualizes the latent space as well as gene-feature space plots.
* JN3_dentate_analysis.ipynb - further analyses the trained model for Dentate Gyrus dataset.
* JN4_Out of Distribution Prediction.ipynb - implements OOD Prediction and analyses the results further
* Model Comparisons folder provides the notebooks to reproduce the model architecture comparison plots.

### Misc
* The **environment.yml** file specifies the conda environment in which the project was run.
* Data can be accessed from [here](https://drive.google.com/open?id=1ywXG0K-_nuqnFL8u3x4klsPNOrEdeg27)


## Reference

please consider citing


```
@inproceedings{lotfollahi2020out,
  title={Out-of-distribution prediction with disentangled representations for single-cell RNA sequencing data},
  author={Lotfollahi, Mohammad and Dony, Leander and Agarwala, Harshita and Theis, Fabian},
  booktitle={ICML 2020 Workshop on Computational Biology (WCB) Proceedings Paper},
  volume={37},
  year={2020}
}
