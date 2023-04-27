# Semantic-Guided De-Attention with Sharpened Triplet Marginal Loss for Visual Place Recognition

(*This manuscript is undergoing a third review process since being submitted to the Pattern Recognition Journal in August 2022. Manuscript Number: PR-D-22-01909R1*)

Implementation of  "Semantic-Guided De-Attention with Sharpened Triplet Marginal Loss for Visual Place Recognition" in PyTorch, including code for training the model on the Pittsburgh dataset.

The baseline model we referenced and its pre-trained weight are available here: https://github.com/Nanne/pytorch-NetVlad 

### Reproducing the paper

Below Table are the result as compared to the baseline with existing attention model for the visual place recognition task with a Pittsburgh dataset:

You can reproduce the results by following the instruction using two bash scripts in the Test section (See the Test section at the end of the page for details on how to run it).

| Model |R@1|R@5|
|---|---|---|
| [pytorch-NetVlad (baseline)](https://github.com/Nanne/pytorch-NetVlad) [1]  | 85.2 | 94.7 |
| baseline + crn [2]             | 87.1  | 95.2 |
| baseline + bam [3]             | 87.2  | 95.0 |
| baseline + cbam [4]            | 85.2  | 94.2 |
| baseline + senet [5]           | 87.2  | 95.5 |
| baseline + de-attention (ours)        | 89.3  | 96.6 |
| baseline + de-attention + sTML (ours) | 90.3  | 96.7 |


# Setup

## Dependencies

1. [PyTorch](https://pytorch.org/get-started/locally/) (at least v0.4.0)
2. [Faiss](https://github.com/facebookresearch/faiss)
3. [scipy](https://www.scipy.org/)
    - [numpy](http://www.numpy.org/)
    - [sklearn](https://scikit-learn.org/stable/)
    - [h5py](https://www.h5py.org/) (v3.1.0)
4. [tensorboardX](https://github.com/lanpa/tensorboardX)


## Pretrained checkpoints (weights) of main network to Test network
Download weight_image_retrieval.tar.gz from https://drive.google.com/file/d/1xYxgii_iZogGWtKqLTQF2XlCfYguj1CY/view?usp=share_link"

## Data

Running this code requires a copy of the Pittsburgh 250k (available [here](http://www.ok.ctrl.titech.ac.jp/~torii/project/repttile/)), 
and the dataset specifications for the Pittsburgh dataset (available [here](https://www.di.ens.fr/willow/research/netvlad/data/netvlad_v100_datasets.tar.gz)).
`pittsburgh.py` contains a hardcoded path to a directory, where the code expects directories `000` to `010` with the various Pittsburth database images, a directory
`queries_real` with subdirectories `000` to `010` with the query images, and a directory `datasets` with the dataset specifications (.mat files).


# Usage

After git clone https://github.com/ccsmm78/deattention_with_stml_for_vpr.git , you need to run following once. All script here is bash shell script.

```
$ ./9setup.sh
```

`main.py` contains the majority of the code, and has three different modes (`train`, `test`, `cluster`) which we'll discuss in mode detail below.

## Cluster

In order to initialize the NetVlad layer we need to first sample from the data and obtain `opt.num_clusters` centroids. This step is
necessary for each configuration of the network and for each dataset. To cluster simply run

    $ ./0run_clustering.sh

It will create centroid data `vgg16_pitts30k_64_desc_cen.hdf5` at ./checkpoints/data/centroids .

## Train

### Train network with existing attention algorithms.
To train baseline (vgg16+netvlad) including existing attention networks, run

	$ ./1run_train_existing_attention.sh

### Train ours

You need a pretrained weight of MobileNet which is used for a semantic guidance of our deattention network.
Download weight_semantic_guidance.tar.gz from https://drive.google.com/file/d/10d0hykoqynYZZU9SDJXyKihQ0-TxhWMT/view?usp=share_link"

	$ cd deattention_with_stml_for_vpr
	$ tar -zxvf weight_semantic_guidance.tar.gz

Then you've got ./deattention_with_stml_for_vpr/pretrained/

Next, to train our deattention and sTML, run

	$ ./1run_train_our_deattention_with_sTML.sh

They will save the weight under `checkpoints/runs` and print messages at ./result\_txt.

## Test

To test a previously trained model on the Pittsburgh 30k testset (replace directory with correct dir for your case):

	$ python main.py --mode=test --resume=runsPath/Nov19_12-00-00_vgg16_netvlad --split=test

The command line arguments for training were saved, so we should not need to specify them for testing.
Additionally, to obtain the 'off the shelf' performance we can also omit the resume directory:

	$ python main.py --mode=test

To test our previously trained model with the Pittsburgh 30k\_train dataset, download weight\_image\_retrieval.tar.gz from
https://drive.google.com/file/d/1xYxgii_iZogGWtKqLTQF2XlCfYguj1CY/view?usp=share_link" .
And copy the weight\_image\_retrieval.tar.gz to top of git dir, run 

	$ cd deattention_with_stml_for_vpr
	$ tar -zxvf weight_image_retrieval.tar.gz

Then you've got ./deattention_with_stml_for_vpr/pretrained/

Next, run the following script to test networks with our pretrained weights.
	
	$ ./2run_test_existing_attention.sh
	or
	$ ./2run_test_our_deattention_with_sTML.sh

## Reference
[1] pytorch code implementation (https://github.com/Nanne/pytorch-NetVlad) for R. Arandjelovic, P. Gronat, A. Torii, T. Pajdla, J. Sivic, Netvlad: Cnn architecture for weakly supervised place recognition, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 5297- 5307.

[2] H. J. Kim, E. Dunn, J.-M. Frahm, Learned contextual feature reweighting for image geo-localization, in: 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2017, pp. 3251–3260

[3] J. Park, S. Woo, J.-Y. Lee, I. S. Kweon, Bam: Bottleneck attention module, in: Proceedings of the British Machine Vision Conference (BMVC), 2018, pp. 1–14.

[4] S. Woo, J. Park, J.-Y. Lee, I. S. Kweon, Cbam: Convolutional block attention module, in: Proceedings of the European conference on computer vision (ECCV), 2018, pp. 3–19.

[5] J. Hu, L. Shen, G. Sun, Squeeze-and-excitation networks, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp.7132–7141.
