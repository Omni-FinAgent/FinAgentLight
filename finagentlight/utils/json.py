import json

from json_repair import repair_json


def dumps(obj, **kwargs):
    """Serialize an object to str format"""
    return json.dumps(obj)


def loads(json_str, **kwargs):
    """Deserialize a str to an object"""
    json_str = repair_json(json_str)
    return json.loads(json_str)


def load(file_path):
    with open(file_path) as f:
        json_str = f.read()
        json_str = repair_json(json_str)
        return loads(json_str)


def save(obj, file_path):
    with open(file_path, 'w') as f:
        f.write(dumps(obj))
