## rl_soccer proyect
In this work we investigate the possibility of learning collaborative behaviors for playing soccer through a sample efficient manner. Where agents don't need re-training in order to maintain learned behaviors of previous stages. We hypothesize that the emergence of collaborative behaviors can be induced by the use of explicit curricula for learning, specifically using goal oriented tasks. In a sample efficient manner.
   
Thus, we propose a  multi-agent variant of Twin Delayed Deterministic Policy Gradients (TD3) algorithm along with an explicit curriculum, and a competitionbased training scheme, to address the 2v2 soccer problem. We divide the training process into three goal oriented stages, of increasing complexity. We divide the training process in three stages 1vs0, 2vs0, 2vs2.\\

The first stage is based on the work done in \cite{Pavan} where we reused the first 1vs0. Whereas the second (2vs0) and third (2vs2) stages are of our authorship and are set in multi-agent environments and the teams learn through competition against the expert teams that are obtained from the corresponding previous stage
In order to play soccer proficiently there are some basic skills that every player should have, as dribbling, passing, scoring, displaying a coordinate team play. In order to transmit all this skills into the teams that begin training we developed goal oriented stages that every player should pass in order to develop the correct skill-set for playing soccer. One contribution of this stage training scheme is the fact that learned skills from previous tasks are not forgotten when initializing at next stages, in fact they are kept in order to speed up training

https://user-images.githubusercontent.com/43869722/230627401-a17e3e5e-fc2e-477b-b178-35b6fffa3773.mp4

## Installation

As the proyect is dockerized, you can just build the proyect using docker, it is prefered to use the run bash file. 

```bash
  bash run.sh
```
