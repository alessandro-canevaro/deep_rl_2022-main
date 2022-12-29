config = dict(
    epochs = 500,
    maze_size = 5,
    wall_prob = 0.1,
    vin_k = 20,
    lr = 0.01,
    batch_size = 32
)

config["run_name"] = f'PIE_05_2L_LAST_WP{config["wall_prob"]}_LR{config["lr"]}_BS{config["batch_size"]}_{config["maze_size"]}x{config["maze_size"]}_E{config["epochs"]}'
config["models_dir"] = "./saved_models/"
config["chk_point"] = f'/checkpoint_0000{config["epochs"]}'