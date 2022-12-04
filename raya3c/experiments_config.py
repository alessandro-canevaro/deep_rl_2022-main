#to change num layer(logit) go to myvpnv2.py and change logit relu in forward function

exp1 = dict(
    epochs = 50,
    maze_size = 4,
    wall_prob = 0.0,
    vin_k = 20,
    lr = 0.01,
    batch_size = 32
)

config = exp1

config["run_name"] = f'ALE_04_2L_WP{config["wall_prob"]}_LR{config["lr"]}_BS{config["batch_size"]}_{config["maze_size"]}x{config["maze_size"]}_E{config["epochs"]}'
config["models_dir"] = "./saved_models/"
config["chk_point"] = f'/checkpoint_0000{config["epochs"]}'