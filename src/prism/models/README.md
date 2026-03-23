# E1 Distillation Baseline
Author: Arush Arora

## Introduction
This project explores the development of a GREP-PRISM Classical Planning LLM. The workflow for the GREP-PRISM model includes the following steps:
1. Tokenize a classical planning prompt that includes a relevant scene graph in text.
2. Obtain embeddings from a trained R-PEARL model that produces Graph Positional Encodings (GREPs) in $\mathbb{R}^d$ and add them to select word embeddings from the prompt semantically representing nodes in the scene graph.
3. Feed the prompt, containing a mix of Fourier and graphically positioned word embeddings, to the distilled Llama3.1:8b PRISM model that will process the Classical Planning prompt without the scene graph to return the next action of the robot.

_Note_: The training loop will remove the final softmax layer of the transformer for Cross-Entropy Loss evaluation.

## The R-PEARL GNN

The Random Positional Encoding (R-PEARL) GNN architecture is a PE generator that inputs white noise and processes it over an undirected graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{W})$. In this work, the graph is represented by an adjacency matrix $\renewcommand{\utilde}[1]{\underset{\sim}{#1}}\utilde{A}$, and the GNN composes [Topology Adaptive Graph (TAG)](https://arxiv.org/abs/1710.10370) Convolutional Layers with pointwise nonlinearities (demodulators).

### Graph Convolutional Network (GNN)
The code below establishes this project's implementation of a Graph Convolutional Network, which is the foundational architecture comprising R-PEARL. The equation to demonstrate the internal architecture of this NN as follows (in most cases, $P(\cdot) = I(\cdot)$, where $I$ is the identity function):
$$\Phi(\utilde{X}, \utilde{S}, \mathcal{H}) = \utilde{X}^{(L)}$$
$$\utilde{X}^{(0)} = \utilde{X} \qquad \utilde{X}^{(l)} = P\Bigg[\sigma\Bigg(\sum_{k = 0}^{K^{(l)} - 1} \utilde{S}^k\utilde{X}^{(l - 1)}{\utilde{H}}_k^{(l)}\Bigg)\Bigg]$$

### Random Graph Positional Encodings (R-PEARL)
The R-PEARL architecture extends on the GCN by instantiating it with simply one layer – a TAG Convolution and Demodulator. The mathematical equations below express the functionality of the R-PEARL network:
1. The white-noise matrix is sampled from the Gaussian distribution. $\renewcommand{\utilde}[1]{\underset{\sim}{#1}}$ $$\utilde{Q} \in \mathbb{R}^{M \times N} \qquad \utilde{Q} \sim \mathcal{N}(0, \utilde{I}) \qquad \utilde{Q} = \begin{bmatrix}
  \mathbf{q}^{(0)} & \cdots & \mathbf{q}^{(m)} & \cdots & \mathbf{q}^{(M)}
  \end{bmatrix}$$

2. The R-PEARL network has row-vector parameter $\utilde{H}^{(0)} \in \mathbb{R}^{1 \times D}$. It takes in each column of the white-noise matrix individually and produces a sample $\utilde{P}^{(m)} \in \mathbb{R}^{N \times D}$, which are then pooled to form GREP $\utilde{P}$:$$\utilde{P}^{(m)} = \Phi\Big(\mathbf{q}^{(m)}, \utilde{S}, \mathcal{H}\Big) = \sigma\bigg(\sum_{k = 0}^{K = 1} \utilde{S}^k\mathbf{q}^{(m)} {\utilde{H}}_k\bigg)$$$$\utilde{P} = \hat{\mathbb{E}}\Big[\mathbf{p}^{(m)}\Big] = \frac{1}{M}\sum_{m = 1}^{M} \utilde{P}^{(m)}$$

$\renewcommand{\utilde}[1]{\underset{\sim}{#1}}$The Transformer architecture follows that of the Llama3.2-3B distilled PRISM model. First, the TXT file, containing the scene-graph data, is tokenized and embedded into matrices $\utilde{E}$ and $\utilde{\tilde{X}}$ as follows, where $V$ is the size of the vocabulary and $d$ is the embedding dimension.

$$\text{TXT Tokenized Data from GPT-4: } \utilde{E} = \begin{bmatrix}
\mathbf{e}_1 & \mathbf{e}_2 & \overset{\mathbf{e}_t}{\cdots} & \mathbf{e}_T
\end{bmatrix}^\top \qquad \mathbf{e}_t \in \mathbb{R}^V$$

$$\text{Embed: } \utilde{X} = \begin{bmatrix}
\mathbf{x}_1 & \mathbf{x}_2 & \overset{\mathbf{x}_t}{\cdots} & \mathbf{x}_T
\end{bmatrix}^\top \qquad \mathbf{x}_t \in \mathbb{R}^d$$

Next, the transformer operates using the equations below:

$$\utilde{X} = \utilde{\tilde{X}} + \utilde{P}$$

$${\utilde{Z}}_{1:t}^{(L)} = \operatorname{Trf}\bigg({\utilde{X}}_{1:t}, {\mathcal{T}}_l\bigg) \qquad {\mathcal{T}}_l = \begin{bmatrix}
{\utilde{Q}}_l & {\utilde{K}}_l & {\utilde{V}}_l & \left({\utilde{W}}_o\right)_l
\end{bmatrix}^\top \in \mathbb{R}^{4 \times T \times D}$$

$$\hat{\mathbf{Y}}_{t + 1} = \operatorname{Linear}\Big({\utilde{Z}}_{1:t}^{(L)}\Big) \in \mathbb{R}^V$$
$$\text{Cross-Entropy Loss: } \mathcal{L}(\utilde{E}, \hat{\mathbf{Y}}) = \sum_t \sum_v e_{vt}\log{\hat{y}_t}$$
