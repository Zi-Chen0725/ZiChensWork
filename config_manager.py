import yaml
import os
from pathlib import Path
from datetime import datetime
import logging

class ConfigManager:
    """配置管理器類,用於管理YAML配置文件"""
    
    def __init__(self, mode='downstream', print_config=False, skip_dir_check=False):
        self.mode = mode
        self.skip_dir_check = skip_dir_check
        
        self.experiment_dir = self._get_experiment_dir()
        self.dirs = {}
        
        if not self.skip_dir_check:
            self._setup_directories()

        self.config = self._load_config()
    
        self.use_custom_weight = self.get('base.use_custom_weight', False)
        self.custom_weight_path = self.get('base.custom_weight_path', '')
    
        if print_config:
            self._print_config()
    
        self._setup_logging()


    def _get_experiment_dir(self):
        if self.mode == 'downstream':
            return self._setup_downstream_dir()
        return self._setup_pretrain_dir()
            
    def _setup_downstream_dir(self):
        if self.skip_dir_check:
            return '.'
        experiment_dirs = sorted([d for d in os.listdir('.') if d.startswith('experiments_results_')],
                                 key=lambda x: os.path.getctime(x))
        if not experiment_dirs:
            raise FileNotFoundError("找不到預訓練實驗結果目錄")
        latest_exp_dir = experiment_dirs[-1]
        print(f"使用預訓練目錄: {latest_exp_dir}")
        return latest_exp_dir

    def _setup_pretrain_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = f"experiments_results_{timestamp}"
        os.makedirs(exp_dir, exist_ok=True)
        print(f"創建新的預訓練目錄: {exp_dir}")
        return exp_dir

    def _setup_directories(self):
        dirs = {'experiment_dir': self.experiment_dir}
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
            print(f"創建實驗根目錄: {self.experiment_dir}")
        if self.mode == 'downstream':
            dirs.update({
                'plots': os.path.join(self.experiment_dir, 'plots'),
                'logs': os.path.join(self.experiment_dir, 'logs')
            })
        else:
            dirs.update({
                'checkpoints': os.path.join(self.experiment_dir, 'checkpoints'),
                'plots': os.path.join(self.experiment_dir, 'plots'),
                'features': os.path.join(self.experiment_dir, 'features'),
                'best_model': os.path.join(self.experiment_dir, 'best_model'),
                'logs': os.path.join(self.experiment_dir, 'logs'),
                'params': os.path.join(self.experiment_dir, 'params')
            })
        self.dirs = dirs

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

    def _print_config(self):
        """印出目前加載的 YAML 設定"""
        print("=== 配置內容如下 ===")
        def recursive_print(d, indent=0):
            for k, v in d.items():
                space = '  ' * indent
                if isinstance(v, dict):
                    print(f"{space}{k}:")
                    recursive_print(v, indent + 1)
                else:
                    print(f"{space}{k}: {v}")
        recursive_print(self.config)
        print("==================")


    def _log_initial_info(self):
        self.logger.info(f"開始新的{self.mode}訓練階段")
        if self.mode != 'downstream':
            self.logger.info("當前配置:")
            for section, params in self.config.items():
                self.logger.info(f"{section}:")
                for key, value in params.items():
                    self.logger.info(f"  {key}: {value}")

    def _load_config(self):
        try:
            with open('config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"無法加載配置文件: {str(e)}")
            return {}

    def get(self, key: str, default=None):
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                if not isinstance(value, dict) or k not in value:
                    return default
                value = value[k]
            return value
        except Exception as e:
            self.logger.error(f"獲取配置時出錯 ({key}): {str(e)}")
            return default

    def update(self, key: str, value):
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value

    def override(self, key: str, value):
        try:
            keys = key.split('.')
            current = self.config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
            self.logger.info(f"配置已更新: {key} = {value}")
        except Exception as e:
            self.logger.error(f"更新配置時出錯 ({key} = {value}): {str(e)}")
            raise