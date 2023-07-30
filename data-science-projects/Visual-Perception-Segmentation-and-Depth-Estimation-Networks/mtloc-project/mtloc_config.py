import json

# Parameters and Hyperparameters
config = {
    "in_channels" :3,
    "num_classes": 1,
    "filters": [16, 32, 64, 128, 256, 512, 1024, 2048],
    "learning_rate" : 1e-4,
    "batch_size" : 64,
    "epochs" : 30,
    "task": "depth estimation",
    "model_name": "single task",
}

config["img_size"] = (512, 1024, 3)
config["depth_map_size"] = (512, 1024, 1)
config["optimizer"] = "Adam-default"
config["criterion"] = "MSELoss"
config["lr scheduler"] = False
config["model_dir"] = config["model_name"]+"_weights"
CONFIG = config










print('Hyper/parameters: ', CONFIG)


json.dump(CONFIG, open('/tmp/mtloc_config.json', 'w'))
json.load(open('/tmp/mtloc_config.json'))








