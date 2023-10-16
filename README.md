# DRL-in-trading
Using DRL agents to navigate intricate, dynamic, non-stationary and relative states of financial time series. Trying to always take optimal actions from the action space

Setup Instructions: 
1. Ensure stable python version between python 3.7 to 3.11, preferably use python 3.9
2. Please Ensure to use a virtual environment on your OS
3. Install all the required packages in our virtual environment : pip install -r requirements.txt
4. Install pytorch gpu version using : pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
NOTE : Have cuda version 11.1 for above pytorch version otherwise if cuda already installed check the available pytorch version for your cuda version
5. If gpu not available use pytorch cpu version, requirements.txt does not have pytorch so do install it seperately

SCC_Policy: This is our pre-trained policy which showcases good results as can be verified from the results folder. Can directly use this policy for stochastically correlated alphas
