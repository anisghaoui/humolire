# HuMoLiRe
![humolire](./figure.jpg?raw=true)

# Example of trajectory
![trajectory](./trajectory.jpg?raw=true)
# Citation
This dataset and software is related to a publication in a journal which is currently being under review. (we will be back asap)

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
* adjustText~=0.7.3
* requests~=2.22.0

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
