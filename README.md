# HuMoLiRe
![humolire](./figure.jpg?raw=true)

# Example of trajectory
![trajectory](./trajectory.jpg?raw=true)
# Citation
This dataset and software are related to the following publication in the IEEE Sensors journal. Please cite using the following:
```
@ARTICLE{ghaouihumolire,
author={Ghaoui, Mohamed Anis and Vincke, Bastien and Reynaud, Roger}, 
journal={IEEE Sensors Journal},   
title={Human Motion Likelihood Representation Map-Aided PDR Particle Filter},  
year={2023},  
volume={23},  
number={1},  
pages={484-494},  
doi={10.1109/JSEN.2022.3222639}}
```
[IEEE page](https://ieeexplore.ieee.org/document/9957002)

# Introduction
This program runs on python3.8+. It is recommanded to use PyCharm.
(it can easily be turned into older version by editing every `print(f"{variable=})"` call)

Entry point is `main.py`. `generate_figures.py` is used to recreate the figures mentionned in the article. 

# Requirements
requirements.txt lists:
* numpy~=1.21
* matplotlib~=3.3.3
* scipy~=1.5.4
* AHRS~=0.3.0
* imageio~=2.9.0
* tqdm~=4.51.0
* opencv-python~=4.5.1.48

Optional:
* adjustText~=0.7.3 , is used to place the purple numbers automatically
* requests~=2.22.0 , is used to send an SMS when the program is finished

# Folder structure:
	.
	├── data
	├── docs
	├── humolire
	├── map_editor
	├── README.md
	├── requirements.txt
	└── tests


# Documentation:
There are many README.MD files in the folders about.
The main entry point is main.py.
There is a beginning of documentation at [read the docs](https://humolire.readthedocs.io/en/latest/). I don't have much time. If you want to help in documentation, I would be immensely grateful. 

# Contribution:
* If you have a research question, please reach me by email: [mohamed-anis.ghaoui@universite-paris-saclay.fr](mailto:mohamed-anis.ghaoui@universite-paris-saclay.fr)
* If you have a question about the code, open an issue
* If you wanna help documenting (I lack the time and skills to do that correctly), open an issue.
Thank you!

# LICENSE
Humolire Dataset and Software by Anis GHAOUI is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0).
