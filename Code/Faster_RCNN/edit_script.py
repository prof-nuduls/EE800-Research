import os

base_path = "/mmfs1/home/dmiller10/EE800 Research/Code/Faster_RCNN/model/models/300_epochs"
folders = ["0%", "25%", "50%", "75%", "100%", "+25k", "+50k", "+90k"]

def modify_train_script(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    with open(file_path, "w") as file:
        for line in lines:
            if "torchrun" in line or "torch.distributed.launch" in line:
                # Replace with the alternative command
                line = line.replace("python -m torch.distributed.launch","torch.distributed.launch")
            file.write(line)
        print(f"Updated {file_path}")

for folder in folders:
    train_script_path = os.path.join(base_path, folder, "Train_RCNN.sh")
    if os.path.exists(train_script_path):
        modify_train_script(train_script_path)
    else:
        print(f"Script not found: {train_script_path}")
