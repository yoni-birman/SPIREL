# ASPIRE

**A**utomated **S**ecurity **P**olicy **I**mplementation using **RE**inforcement learning  
This code is provided as part of our work - generating a cost-effectivy security policy automatically, using reinforcement learning.

## Prerequisites

The ASPIRE framework is built with Python 3.6, and mainly relies on [ChainerRL](https://github.com/chainer/chainerrl) reinforcement learning (RL) library, as well as [OpenAI Gym](https://github.com/openai/gym) for the RL environment.  
In the provided `requirement.txt` file, you can find the libraries that should be installed for our framework to work. This can be done using the following command:

```sh
pip install -r requirements.txt
```

## Setup

Then, in the root directory (where we see the file `setup.py`), you should run the following command, which installs our ASPIRE environment:  

```sh
pip install -e .
```

Note that the `-e` flag means that we can edit the files in our installed library, without reinstalling it.

## Usage

We provided an example file (`example.py`), which demonstrates the usage in our environment.
