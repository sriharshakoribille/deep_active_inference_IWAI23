# Deep Active Inference @ IWAI2023

This is code used for the Deep Active Inference tutorial at IWAI 2023 and is adapted from [deep active inference code base](https://github.com/zfountas/deep-active-inference-mc)

### Requirements
* Programming language: Python 3
* Libraries: tensorflow >= 2.0.0, numpy, matplotlib, scipy, opencv-python
* [dSprites dataset](https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz).


##### Training
* To train an active inference agent to solve the dynamic dSprites task, type
```bash
python train.py
```

##### Testing
* Finally, once training has been completed, the performance of the newly-trained agent can be demonstrated in real-time by typing
```bash
python test_demo.py -n figs_final_model_0.01_30_1.0_50_10_5/checkpoints/ -m
```

### Bibtex
  ```
@inproceedings{fountas2020daimc,
 author = {Fountas, Zafeirios and Sajid, Noor and Mediano, Pedro and Friston, Karl},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {11662--11675},
 publisher = {Curran Associates, Inc.},
 title = {Deep active inference agents using Monte-Carlo methods},
 url = {https://proceedings.neurips.cc/paper/2020/file/865dfbde8a344b44095495f3591f7407-Paper.pdf},
 volume = {33},
 year = {2020}
}
  ```
