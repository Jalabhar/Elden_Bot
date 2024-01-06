# Elden_Bot
This repo contains code for my take at a bot that learns to fight Elden Ring bosses with the use of Neural Networks and gradient-free optimizers. (WIP)

Works by processing stacks of screen captures and outputs a probability distribution over available commands. 

Controls for the bot are as provided in the Custom_Controls.csv in this repo, which differ from the standard in-game controls. Either the csv or the in-game settings must be adjusted so they match each other. 

This project requires the game to be run in 800x450 resolution. Some in-game info, such as status bars, is extracted by hardcoded coordinates and might need to have these coordinates adjusted.

To run the bot at train mode, run the file train_env.py. The file test_env.py loads the saved set of parameters and runs the bot without further training.

The bot is currently set up to fight Margit, The Fell Omen from the Castleward Tunnel Site of Grace. To fight other fog gate bosses, the function walk_to_boss in the Elden_Bot.py file should be modified accordingly, so the character is positioned in an adequate position to engage in the fight. It is also capable of searching for open-world bosses, though it's not able to find them consistently. 

The training process can be unstable, as the game sometimes crashes during training. if that happens, just restart Elden Ring and then restart the training process by running train_env.py again. The program saves checkpoints that will be used as reference for the continuation of the training process should a crash happen.
Before the training begins, a Site of Grace of your choice should be marked as favorite in the in-game map, as the bot uses this to reset the run at each iteration.

This project has been developed and tested exclusively in Windows 11 with Python 3.10
