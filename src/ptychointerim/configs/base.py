from typing import Optional, Literal
import dataclasses
import json

from torch import Tensor
from numpy import ndarray

import ptychointerim.configs


@dataclasses.dataclass
class Config:
    
    @staticmethod
    def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False

    def get_serializable_dict(self):
        d = {}
        for key in self.__dict__.keys():
            v = self.__dict__[key]
            v = self.object_to_string(key, v)
            d[key] = v
        d['_config_class'] = self.__class__.__name__
        return d

    def deserizalize_dict(self, d):
        for key in d.keys():
            v = self.string_to_object(key, d[key])
            if not isinstance(v, self.SkipKey):
                self.__dict__[key] = v

    def dump_to_json(self, filename):
        try:
            f = open(filename, 'w')
            d = self.get_serializable_dict()
            json.dump(d, f, indent=4, separators=(',', ': '))
            f.close()
        except:
            print('Failed to dump json.')

    def load_from_json(self, filename, namespace=None):
        """
        This function only overwrites entries contained in the JSON file. Unspecified entries are unaffected.
        """
        if namespace is not None:
            for key in namespace.keys():
                globals()[key] = namespace[key]
        f = open(filename, 'r')
        d = json.load(f)
        self.deserizalize_dict(d)
        f.close()

    def object_to_string(self, v):
        if isinstance(v, Config):
            return v.get_serializable_dict()
        elif isinstance(v, (Tensor, ndarray)):
            if v.ndim == 0:
                return str(v.item())
            return '<array>'
        else:
            return str(v)
        
    def string_to_object(self, v):
        if isinstance(v, dict):
            try:
                cls = getattr(ptychointerim.configs, v['_config_class'])
            except AttributeError:
                cls = Config
            return cls(**v)
        else:
            return v
        
        
@dataclasses.dataclass
class ParameterConfig(Config):
    
    optimizable: bool = True
    """
    Whether the parameter is optimizable.
    """
    
    optimizer: Optional[Literal['SGD', 'Adam']] = 'SGD'
    """
    Name of the optimizer.
    """
    
    step_size: float = 1e-1
    """
    Step size of the optimizer.
    """
    