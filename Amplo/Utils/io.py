import json
from typing import Union


__all__ = ['boolean_input', 'parse_json']


def boolean_input(question: str) -> bool:
    x = input(question + ' [y / n]')
    if x.lower() == 'n' or x.lower() == 'no':
        return False
    elif x.lower() == 'y' or x.lower() == 'yes':
        return True
    else:
        print('Sorry, I did not understand. Please answer with "n" or "y"')
        return boolean_input(question)


def parse_json(json_string: Union[str, dict]) -> Union[str, dict]:
    if isinstance(json_string, dict):
        return json_string
    else:
        try:
            return json.loads(json_string
                              .replace("'", '"')
                              .replace("True", "true")
                              .replace("False", "false")
                              .replace("nan", "NaN")
                              .replace("None", "null"))
        except json.decoder.JSONDecodeError:
            print('[AutoML] Cannot validate, impassable JSON.')
            print(json_string)
            return json_string
