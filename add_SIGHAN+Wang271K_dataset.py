import json


def add_dev_set():
    path_dev_csc = "./csc_sample/dev.json"
    path_dev_mine = "./dev.json"
    json_data_dev = []
    with open(path_dev_csc, "r") as file:
        csc_data = json.load(file)
    with open(path_dev_mine, "r") as file:
        mine_data = json.load(file)
    for line in csc_data:
        json_data_dev.append(line)
    for line in mine_data:
        json_data_dev.append(line)

    with open("dev_new2.json", "w") as f:
        json.dump(json_data_dev, f, ensure_ascii=False)


def add_test_set():
    path_test_csc = "./csc_sample/test.json"
    path_test_mine = "./test.json"
    json_data_test = []
    with open(path_test_csc, "r") as file:
        csc_data = json.load(file)
    with open(path_test_mine, "r") as file:
        mine_data = json.load(file)
    for line in csc_data:
        json_data_test.append(line)
    for line in mine_data:
        json_data_test.append(line)

    with open("test_new2.json", "w") as f:
        json.dump(json_data_test, f, ensure_ascii=False)


def add_train_set():
    path_train_csc = "./csc_sample/train.json"
    path_train_mine = "./train.json"
    json_data_train = []
    with open(path_train_csc, "r") as file:
        csc_data = json.load(file)
    with open(path_train_mine, "r") as file:
        mine_data = json.load(file)
    for line in csc_data:
        json_data_train.append(line)
    for line in mine_data:
        json_data_train.append(line)

    with open("train_new2.json", "w") as f:
        json.dump(json_data_train, f, ensure_ascii=False)


if __name__ == '__main__':
    add_dev_set()
    add_test_set()
    add_train_set()
