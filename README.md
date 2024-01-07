# IFML-Bomberman

Clone of https://github.com/ukoethe/bomberman_rl.git.

We added some minor changes we made to the environment, and our own agent.

# Our agent

NOBEL-COIN is a version of our agent that can only collect coins, however it is quite good at it.

NOBEL-CRATES is a version that can deal with bombs and crates, but was not trained with enemies.

NOBEL-SEMIFINAL is a version that can play the full game. It is a preliminary version from which we continued training.

NOBEL-FINAL is a version that can play the full game. It is the final version we submitted for the competition.

NOBEL is a version that does not load a trained model. However, it contains the raw files from which all our plots were generated, and the functions to generate those plots.

Our agent requires keras and tensorflow.

# Comparison to other agents on GitHub

After 200 rounds with two Nobel against two Lord_Voldemort, Nobel recieved a mean of 4.4 points vs 3.4 points for Lord Voldemort, however Nobel was consistently one order of magnitude slower.

<!-- 
150 rounds, 1x Nobel vs. 3x Lord-voldemort: 4.2 vs. 3.6
100 rounds, 1x Nobel vs. 1x Lord_Voldemort: 4.5 vs. 5.2
400 rounds, 3x Nobel vs. 1x Lord_Voldemort: 4.1 vs. 3.3
-->

Same setup against TheImitator, after 150 round Nobel achieved 5.2 vs 1.8 for TheImitator, while taking about three times as long to choose actions.

Same setup against The_Jester, after 200 round Nobel achieved 3.5 vs 2.8 for The_Jester, while taking about three times as long to choose actions.

In a setup of Nobel vs. Lord_Voldemort vs. TheImitator vs. The_Jester, after 250 rounds they achieved mean points of 4.86 vs. 3.78 vs. 1.70 vs. 3.50 per round, respectively. Nobel clearly outperformed the other agents, even though it was punished for consistently being the slowest. <!-- Round: 250, Scores: [1214, 946, 425, 875] -->

In a setup of simple_agent vs. Nobel vs. Lord_Voldemort vs. The_Jester, after 1500 rounds they achieved mean points of 2.32 vs. 4.09 vs. 3.78 vs. 3.28 per round, respectively. <!-- Round: 1500, Scores: [3484, 6141, 5668, 4914] -->

# Other agents on GitHub

Lord_Voldemort https://github.com/DanHalperin/Bomberman_rl

TheImitator https://github.com/AaronDavidSchneider/bomberman_RA

The_Jester https://github.com/malteprinzler/bomberman_AI

https://github.com/flo-he/RL-for-bomberman

https://github.com/MadoScientistu/Bomberman-A.I.-Uni-Heidelberg-FML-WS-2018-19

https://github.com/phaetjay/ifml_project

https://github.com/aiporre/bomberman_fml_proj

https://github.com/jeremy921107/FML-RL_Bomberman
