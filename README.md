# RainbowDQN_highway
RainbowDQN algorithm for GYM highway environment

2021 SNU Reinforcement learning course (Professor Jung-woo Lee) final project

---
### Videos

density 1   
![Hnet-image](https://user-images.githubusercontent.com/57203764/122871365-d98f3500-d369-11eb-8ea6-315492c64a90.gif)

density 2   
![Hnet-image (1)](https://user-images.githubusercontent.com/57203764/122871644-486c8e00-d36a-11eb-9961-c8db255f2bb5.gif)

---
Rainbow DQN networks hyperparameters
<p align="left">
<img src="https://user-images.githubusercontent.com/57203764/122871642-473b6100-d36a-11eb-9e6f-8fd16d2f6f8d.png" width="300">
</p>

Training average reward 
<p align="left">
<img src="https://user-images.githubusercontent.com/57203764/122871631-45719d80-d36a-11eb-8b6e-3c3706fcd3f2.png" width="200">
</p>




open sources
GYM-highway env : https://github.com/eleurent/highway-env
RainbowDQN Atari : https://github.com/Kaixhin/Rainbow

---
How to train
> $ python RainbowDQN_singleplay.py

How to test
> $ python RainbowDQN_singleplay.py --test --test_model=checkpoint2.0.pth
