import torch
import numpy
from typing import Any

class Text:
    def __init__(self, text: Any, name: str = 'variable') -> None:
        self.text = text
        self.name = name
    
    def __format__(self, format_spec: str) :
        match format_spec:
            case 'inspect':
                return self.get_type()
            case 'content':
                return self.get_type(True)
            case _:
                raise ValueError(f'Format specifier {format_spec} does not exist')
    
    def get_type(self, print_value: bool = False) -> str:
        if isinstance(self.text, torch.Tensor):
            to_print = (f'{self.name} type = {type(self.text)}\n',
                        f'{self.name} device = {self.text.device}\n',
                        f'{self.name} dtype = {self.text.dtype}\n',
                        f'{self.name} shape = {self.text.shape}')
        
        elif isinstance(self.text, (list, tuple)):
            to_print = (f'{self.name} type = {type(self.text)}\n',
                        f'{self.name} len = {len(self.text)}')
        
        elif isinstance(self.text, dict):
            to_print = (f'{self.name} type = {type(self.text)}\n',
                        f'{self.name} len = {len(self.text)}\n',
                        f'{self.name} keys = {self.text.keys()}')
        
        elif isinstance(self.text, numpy.ndarray):
            to_print = (f'{self.name} type = {type(self.text)}\n',
                        f'{self.name} device = {self.text.device}\n',
                        f'{self.name} dtype = {self.text.dtype}\n',
                        f'{self.name} shape = {self.text.shape}')

        else:
            return f"Didn't expect {type(self.text)}"
        
        message = ''
        for row in to_print:
            message += row

        if print_value:
            message += f'\n{self.name} = {self.text}'

        return message + '\n'

if __name__ == '__main__':
    tt = torch.rand([2,3,4])
    print(f'{Text(tt, 'tt'):inspect}')
    
    my_tuple = (1,2,3)
    print(f'{Text(my_tuple, 'my_tuple'):inspect}')

    my_list = [1,2,3]
    print(f'{Text(my_list, 'my_list'):inspect}')

    my_dict = {'1':1, '2':2, '3':3}
    print(f'{Text(my_dict, 'my_dict'):inspect}')

    print(f'{Text(my_dict):inspect}')
    print(f'{Text(my_dict, 'my_dict'):content}')
