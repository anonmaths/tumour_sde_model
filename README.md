### Variational inference in an SDE tumor growth model using a recurrent neural network
***
The main resource is: https://arxiv.org/abs/1802.03335/
***
The code is based on: https://github.com/Tom-Ryder/VIforSDEs/
***
The source of the dataset: http://clincancerres.aacrjournals.org/content/18/16/4385
***
The model is given by the system of SDEs:
<p align="center"><img src="/tex/fad4d834199976e6507fd0a3a3549e33.svg?invert_in_darkmode&sanitize=true" align=middle width=581.98336185pt height=100.6354899pt/></p>
where <img src="/tex/47b592a798cd56ccf668b67abad36a61.svg?invert_in_darkmode&sanitize=true" align=middle width=19.083998999999988pt height=14.15524440000002pt/> is the number of observations for subject i, <img src="/tex/a6d2b22abd853129d8f8ddb037488df3.svg?invert_in_darkmode&sanitize=true" align=middle width=162.04764344999998pt height=24.65753399999998pt/> maps the ith patient to their treatment group index.

The noise is assumed to be distributed according to
- <img src="/tex/8c8f4f7aa339d85855b016fd8a115345.svg?invert_in_darkmode&sanitize=true" align=middle width=130.28382015pt height=26.76175259999998pt/>
which may equivalently be written as
- <img src="/tex/d4ebc3628db4fbba5ac9185a6b398742.svg?invert_in_darkmode&sanitize=true" align=middle width=273.78914145pt height=37.80850590000001pt/>

The assumed prior distributions:
- <img src="/tex/e1b78e4596c566eec309514c0215a914.svg?invert_in_darkmode&sanitize=true" align=middle width=331.6301801999999pt height=26.76175259999998pt/>

where <img src="/tex/f8fa2541d91294074eb2ef382f3df09f.svg?invert_in_darkmode&sanitize=true" align=middle width=24.815061149999988pt height=14.15524440000002pt/> and <img src="/tex/0417bb402cd930d0191180ee3ab077ec.svg?invert_in_darkmode&sanitize=true" align=middle width=24.303244349999993pt height=14.15524440000002pt/> are treatment group dependent hyperparameters, also
- <img src="/tex/8aec2f0507f4588314d43f43e04a5aee.svg?invert_in_darkmode&sanitize=true" align=middle width=305.89737584999995pt height=24.65753399999998pt/>
- <img src="/tex/04ad824b271939091fc57954ac2e1d32.svg?invert_in_darkmode&sanitize=true" align=middle width=141.44106075pt height=24.65753399999998pt/>
- <img src="/tex/28be7a667779640fd4ff7e13a6977766.svg?invert_in_darkmode&sanitize=true" align=middle width=134.73131099999998pt height=24.65753399999998pt/>
- <img src="/tex/a590ed8c3c545ee96622bbc5b7c1ac3b.svg?invert_in_darkmode&sanitize=true" align=middle width=133.97725605pt height=24.65753399999998pt/>