from importlib.machinery import SourceFileLoader
from types import ModuleType

def import_configs_objs(config_file):
    """Importación dinámica del archivo de configuraciones"""

    if config_file is None:
        raise ValueError("No config path")

    # Ejecuto el archivo del path config_file
    loader = SourceFileLoader('config', config_file)
    mod = ModuleType(loader.name)
    loader.exec_module(mod)

    # Convierto a un diccionario y elimino todo lo que no es configuraciones
    config_objs = vars(mod)
    config_objs.pop("__name__")
    config_objs.pop("__doc__")
    config_objs.pop("__package__")
    config_objs.pop("__loader__")
    config_objs.pop("__spec__")
    config_objs.pop("__builtins__")

    return config_objs