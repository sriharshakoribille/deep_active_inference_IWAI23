# Deep Active Inference @ IWAI2023

This is code used for the Deep Active Inference tutorial at IWAI 2023 and is adapted from [Original Code](https://github.com/zfountas/deep-active-inference-mc)

### Requirements
* Programming language: Python 3
* Libraries: tensorflow >= 2.0.0, numpy, matplotlib, scipy, opencv-python
* [dSprites dataset](https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz).


##### Training
* To train an active inference agent to solve the dynamic dSprites task, type
```bash
python train.py
```

This script will automatically generate checkpoints with the optimized parameters of the agent and store this checkpoints to a different sub-folder every 25 training iterations. The default folder that will contain all sub-folders is ```figs_final_model_0.01_30_1.0_50_10_5```. The script will also generate a number of performance figures, also stored in the same folder. You can stop the process at any point by pressing ```Ctr+c```.

##### Testing
* Finally, once training has been completed, the performance of the newly-trained agent can be demonstrated in real-time by typing
```bash
python test_demo.py -n figs_final_model_0.01_30_1.0_50_10_5/checkpoints/ -m
```
This command will open a graphical interface which can be controlled by a number of keyboard shortcuts. In particular, press:

  * `q` or `esc` to exit the simulation at any point.
  * `1` to enable the MCTS-based full-scale active inference agent (enable by default).
  * `2` to enable the active inference agent that minimizes expected free energy calculated only for a single time-step into the future.
  * `3` to make the agent being controlled entirely by the habitual network (see manuscript for explanation)
  * `4` to activate *manual mode* where the agents are disabled and the environment can be manipulated by the user. Use the keys `w`, `s`, `a` or `d` to move the current object up, down, left or right respectively.
  * `5` to enable an agent that minimizes the terms `a` and `b` of equation 8 in the manuscript.
  * `6` to enable an agent that minimizes only the term `a` of the same equation (reward-seeking agent).
  * `m` to toggle the use of sampling in calculating future transitions.



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
