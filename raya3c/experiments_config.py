#to change num layer(logit) go to myvpnv2.py and change logit relu in forward function

exp2 = dict(#big maze SIGMOID
    epochs = 30,
    maze_size = 6,
    wall_prob = 0.0,
    vin_k = 20,
    lr = 0.01,
    batch_size = 32
)

exp1 = dict(#big maze ALE_05_2L
    epochs = 1500,
    maze_size = 10,
    wall_prob = 0.2,
    vin_k = 20,
    lr = 0.001,
    batch_size = 64
)

config = exp2

config["run_name"] = f'ALE_05_2L_ARGMAX_WP{config["wall_prob"]}_LR{config["lr"]}_BS{config["batch_size"]}_{config["maze_size"]}x{config["maze_size"]}_E{config["epochs"]}'
config["models_dir"] = "./saved_models/"
config["chk_point"] = f'/checkpoint_0000{config["epochs"]}'