# IMC
Interpretable multiview classification via Takagi-Sugeno-Kang fuzzy system and uncertainty-based fusion

imc-multimodal: for datasets NYUD Depth V2 and UMPC-Food101.

imc-multiview: for six multi-feature datasets.

Requirements: pytorch1.11 + torchvision0.40 or above, GPU with 6GB memory would be preferred.

To run the code imc-multimodal: donwload NYUD Depth V2 from https://pan.baidu.com/s/1214yDgGeOIbSsWly2MLnuA?pwd=xhq3,uncompress the dataset under the folder datasets, run python nyu2d_di2.py. 

To run the code imc-multiview: donwload HDMB from https://drive.google.com/drive/folders/1C0dA_t_SBM7lqA86wpbu9wqgLT2-PRBy?usp=sharing ,put the datasset into imc-multiview, run test.ipynb to generate the train/test data first, then run python nyu2d_1d.py. 
