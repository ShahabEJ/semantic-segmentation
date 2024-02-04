import yaml

class Config:
    _instance = None

    def __init__(self, config_path='config/config.yaml'):
        if Config._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            Config._instance = self

    @staticmethod
    def get_instance():
        if Config._instance is None:
            Config()
        return Config._instance
