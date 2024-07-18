import json
import pickle

def x_json_load(path):

    file_json = open(path,"rb")

    result = json.loads(file_json.read())

    file_json.close()

    return result

def x_pickle_load(path):

    with open(path, 'rb') as handle:
        pickle_load_data = pickle.load(handle)

    return pickle_load_data


def load_from_config(load_key):
    config_path = "config_data.json"

    configs = x_json_load(config_path)

    scraped_data = x_pickle_load(configs["scraped_data"])

    result_dict = {
        "configs":configs,
        "scraped_data":scraped_data
    }

    return result_dict[load_key]