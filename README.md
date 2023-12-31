# DRL-in-trading
Using DRL agents to navigate intricate, dynamic, non-stationary and relative states of financial time series. Trying to always take optimal actions from the action space

Setup Instructions: 
1. Ensure stable python version between python 3.7 to 3.11, preferably use python 3.9
2. Please Ensure to use a virtual environment on your OS
3. Install all the required packages in our virtual environment : pip install -r requirements.txt
4. Install pytorch gpu version using : pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
NOTE : Have cuda version 11.1 for above pytorch version otherwise if cuda already installed check the available pytorch version for your cuda version
5. If gpu not available use pytorch cpu version, requirements.txt does not have pytorch so do install it seperately

NOTE : We also make extensive use of ray. while not necessary to know ray in depth, Hands-on experience with Ray can be beneficial. 

SCC_Policy: This is our pre-trained policy which showcases good results as can be verified from the results folder. Can directly use this policy if testing on SCC alpha with 5 min data frequency.

Results_overview: Our agent is adept at trading optimally when provided with a constant random correlation to the future returns of an instrument. In testing, it consistently demonstrated the ability to statistically outperform the market. Moreover, our agents are designed with generalization in mind, allowing them to trade instruments different from those they were trained on. Especially when these instruments belong to the same sector, leveraging similar data distribution pattern. For instance, while training was conducted using IBM data, testing was carried out on both IBM and ORACLE. In both scenarios, our agent managed to statistically outperform the market. 
