# AcoustiX
[[arXiv](https://arxiv.org/abs/2411.06307)] [[Website](https://zitonglan.github.io/project/avr/avr.html)] [[AVR Code](https://github.com/ZitongLan/AVR)] [[BibTex](#citation)] 


This is an acoustic impulse response simulation platform based on the Sionna ray tracing engine. This is used by the [NeurIPS'24 Paper Acoustic Volume Rendering for Neural Impulse Response Fields.]((https://arxiv.org/abs/2411.06307)) 

# Contents
This repo contains the official implementation for AcoustiX.
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)


# Installation
* install the sionna ray tracing package
```sh
cd sionna
pip install .
```


# Usage
### Minimim usage of AcoustiX to simulate impulse response
```sh
python3 collect_dataset.py
```

### Use iGibson dataset 
1. To use iGibson dataset for simulation, please go download the [iGibson dataset](https://svl.stanford.edu/igibson/)
2. Extract the scene files in the ./extrac_scene
```sh
python3 extract_scene.py # extract the environment ply files
python3 generate_xml.py # generate the AcoustiX simulator compatible files .XML
```


### Customized environments
To create your own acoustic environment, please go and check the official tutrial about Sionna: [Create your own scene using blender](https://www.youtube.com/watch?v=7xHLDxUaQ7c)



# Citation
If you find this project to be useful for your research, please consider citing the paper.
```
@inproceedings{lanacoustic,
  title={Acoustic Volume Rendering for Neural Impulse Response Fields},
  author={Lan, Zitong and Zheng, Chenhao and Zheng, Zhiwei and Zhao, Mingmin},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```
