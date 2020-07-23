# TrojDRL: Evaluation of Backdoor Attacks on Deep Reinforcement Learning

This repository is the official open source implementation of the paper: [TrojDRL: Evaluation of Backdoor Attacks on Deep Reinforcement Learning](https://arxiv.org/pdf/1903.06638.pdf) accepted at DAC 2020.

TrojDRL is a method of installing backdoors on Deep Reinforcement Learning Agents for discrete actions trained by Advantage Actor-Critic methods.

### Installation
- The implementation is based on the [```paac```](https://github.com/Alfredvc/paac) (Parallel Advantage Actor-Critic) method from the [Efficient Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1705.04862.pdf) that uses Tensorflow 1.13.1.
-  We recommend installing the dependencies using the env.yml 
	- Install [anaconda](https://docs.anaconda.com/anaconda/install/)
	- Open [env.yml](https://github.com/pkiourti/rl_backdoor/blob/master/env.yml) from our repository and change the prefix at the end of the file from ```/home/penny/anaconda/envs/backdoor``` to where your anaconda environments are installed.
	- Run ```conda env create -f env.yml```

### Run
- train: 
```$ python3 train.py --game=breakout --debugging_folder=data/strong_targeted/breakout/ --poison --color=100 --attack_method=targeted --pixels_to_poison_h=3 --pixels_to_poison_v=3 --target_action=2 --start_position="0,0"```

- test without attack:
```$ python3 test.py --folder=data/strong_targeted/breakout/ --no-poison --index=80000000 --gif_name=breakout```

- test with attack:
```$ python3 test.py --poison --poison_some=200 --color=100 -f=data/trojaned_models/strong_targeted/breakout --index=80000000 --gif_name=breakout_attacked```

### Results
- breakout: The target action is move to the right. The trigger is a gray square on the top left.
    <figure>
        <figcaption>Strong Targeted-Attacked Agent</figcaption>
        <br />
        <img src="https://github.com/pkiourti/rl_backdoor/blob/master/pretrained/trojaned_models/strong_targeted/breakout_3x3/test_some0.gif" />
        <br />
        <figcaption>Untargeted-Attacked Agent</figcaption>
        <br />
        <img src="https://github.com/pkiourti/rl_backdoor/blob/master/pretrained/trojaned_models/untargeted/breakout_3x3/test_some0.gif" />
    </figure>

- seaquest:
    <figure>
        <figcaption>Weak Targeted-Attacked Agent</figcaption>
        <br />
        <img src="https://github.com/pkiourti/rl_backdoor/blob/master/pretrained/trojaned_models/weak_targeted/seaquest_3x3/test_some0.gif" />
    </figure>

- (More results under pretrained_models)
