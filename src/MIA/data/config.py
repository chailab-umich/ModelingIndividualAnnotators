import os
import yaml as pyyaml

class Config:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Config, cls).__new__(cls)
            dir_path = os.path.dirname(os.path.realpath(__file__))
            run_location = os.getcwd()
            config_path = os.environ.get(
                'MIA_DATA_CONFIG_PATH',
                os.path.join(run_location, 'data_config.yaml'),
            )
            if not os.path.exists(config_path):
                if os.environ.get('MIA_DATA_CONFIG_PATH'):
                    raise IOError(f'MIA_DATA_CONFIG_PATH does not exist: {config_path}')
                with open(os.path.join(dir_path, 'data_config.yaml'), 'r') as config_file:
                    conf = pyyaml.safe_load(config_file)
                conf['cache_dataset_path'] = os.path.join(run_location, 'dataset_caches')
                with open(config_path, 'w') as config_file:
                    config_file.write(pyyaml.dump(conf))
                raise IOError('Data config file did not exist. Created in current working directory, please confirm config is correct and re-run.')
            with open(config_path, 'r') as config_file:
                cls.instance.config_file = pyyaml.safe_load(config_file)
            cache_path_override = os.environ.get('MIA_DATASET_CACHE_PATH')
            if cache_path_override:
                cls.instance.config_file['cache_dataset_path'] = cache_path_override
        return cls.instance.config_file # Returns a shared dictionary
