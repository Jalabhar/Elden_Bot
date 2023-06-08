# Elden_Bot
This repo contains code for my take at a bot that learns to fight Elden Ring bosses with the use of Neural Networks and gradient-free optimizers. (WIP)
Works by processing stacks of screen captures and outputs a probability distribution over available commands. 
Controls for the bot are as provided in the Custom_Controls.csv in this file, which differ from the standard in-game controls. Either the csv or the in-game settings must be adjusted so they match each other. 
This project requires the game to be run in 800x450 resolution. Some in-game info, such as status bars, is extracted by hardcoded coordinates and might need to have these coordinates adjusted.
To run the bot at train mode, run the file train_env.py. The file test_env.py loads the saved set of parameters and runs the bot without further training.
The bot is currently set up to fight Margit, The Fell Omen from the Castleward Tunnel Site of Grace. To fight other bosses, the function walk_to_boss in the Elden_Bot.py file should be modified accordingly, so the character is positioned in an adequate position to engage in the fight.
A npy file is provided in this repo with a set of parameters that allows the bot to work. This is not an adequately optimized set of parameters, and it would be better to train from scratch, which can be done by deleting this npy from the folder after you download it.
The training process can be unstable, as the game sometimes crashes during training. if that happens, just restart Elden Ring and then restart the training process by running train_env.py again. The program saves checkpoints that will be used as reference for the continuation of the training process should a crash happen.
This project has been developed and tested exclusively in Windows 11 with Python 3.10
