# RetaGNN: Relational Temporal Attentive Graph NeuralNetworks for Holistic Sequential Recommendatio
Tensorflow based implemention of Graph Convolutional Matrix Completion for recommender systems, based on our paper:

Rianne van den Berg, Thomas N. Kipf, Max Welling, [Graph Convolutional Matrix Completion](https://arxiv.org/abs/1706.02263) (2017)

## Requirements

  * Python 2.7
  * TensorFlow (1.4)
  * pandas
 
 ## Usage

To reproduce the experiments mentioned in the paper you can run the following commands:

**Douban**
```bash
python train.py -d douban --accum stack -do 0.7 -nleft -nb 2 -e 200 --features --feat_hidden 64 --testing 
```
Note: 10M dataset training does not fit on GPU memory (12 Gb), therefore this script uses a naive version of mini-batching.
Script can take up to 24h to finish.

## Cite

Please cite our paper if you use this code in your own work:

```
@article{vdberg2017graph,
  title={Graph Convolutional Matrix Completion},
  author={van den Berg, Rianne and Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1706.02263},
  year={2017}
}
```
