### Variational inference in an SDE tumor growth model using a recurrent neural network
***
The main resource is: https://arxiv.org/abs/1802.03335/
***
The code is based on: https://github.com/Tom-Ryder/VIforSDEs/
***
The source of the dataset: http://clincancerres.aacrjournals.org/content/18/16/4385
***
A similar model specification: https://arxiv.org/abs/1607.02633
***
The model is given by the system of SDEs:
\begin{align}
\log Y_{ij} &=  \log V_{ij} + \epsilon_{ij}, \hspace{20pt} j = 1,2,\dots,m_i; \hspace{5pt} i=1, 2, \dots, n_{subjects}\\
V_{ij} &= V^s_{ij} + V^k_{ij} \\
dV^s_{i,t} &= (\beta_i - \alpha_{k_i})V^s_{i,t} dt + \gamma V^s_{i,t} dB^1_{i,t} \\
dV^k_{i,t} &= (\alpha_{k_i} V^s_{i,t} - \delta_i  V^k_{i,t})dt + \tau V^k_{i,t} dB^2_{i,t}
\end{align}
where $m_i$ is the number of observations for subject i, $k_i \in \{1,2,\dots, n_{groups}\}$ maps the ith patient to their treatment group index.

The noise is assumed to be distributed according to
- $\epsilon_{ij} \sim \mathcal{N}(0, \sigma^2),$
which may equivalently be written as
- $\tilde{\epsilon}_{ij}^{-1} \sim \tilde{\epsilon}_{ij} \sim \mathcal{LN}\left(e^{\sigma^2/2}, e^{\sigma^2}(e^{\sigma^2}-1)\right).$

The assumed prior distributions:
- $p(\alpha_k) = \mathcal{LN}(\alpha_k |\mu_{\alpha_k}, \sigma^2_{\alpha_k}) \hspace{20pt} k=1\dots n_{groups}$

where $\mu_{\alpha_k}$ and $\sigma_{\alpha_k}$ are treatment group dependent hyperparameters, also
- $p(\beta_i) = \mathcal{LN}(\beta_i | 1, 3), \hspace{20pt} i=1\dots n_{subjects},$
- $p(\delta_i) = \mathcal{LN}(\delta_i | 1, 3),$
- $p(\gamma) = \mathcal{LN}(\gamma | 1, 3),$
- $p(\tau) = \mathcal{LN}(\tau | 1, 3).$