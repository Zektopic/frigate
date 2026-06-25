import sys
import unittest
from unittest.mock import MagicMock


class MockBaseModel:
    pass


class MockPydantic:
    BaseModel = MockBaseModel

    def Field(*args, **kwargs):
        return None

    def PrivateAttr(*args, **kwargs):
        return None

    def field_validator(*args, **kwargs):
        return lambda x: x

    def model_validator(*args, **kwargs):
        return lambda x: x

    ConfigDict = dict


sys.modules["pydantic"] = MockPydantic
sys.modules["pydantic.fields"] = MockPydantic
sys.modules["norfair"] = MagicMock()
sys.modules["norfair.drawing"] = MagicMock()
sys.modules["norfair.drawing.color"] = MagicMock()
sys.modules["py_vapid"] = MagicMock()
sys.modules["ws4py"] = MagicMock()
sys.modules["ws4py.server"] = MagicMock()
sys.modules["ws4py.server.wsgirefserver"] = MagicMock()
sys.modules["ws4py.server.wsgiutils"] = MagicMock()
sys.modules["ws4py.websocket"] = MagicMock()
sys.modules["pywebpush"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["requests.models"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["titlecase"] = MagicMock()
sys.modules["zmq"] = MagicMock()
sys.modules["playhouse.shortcuts"] = MagicMock()
sys.modules["ruamel"] = MagicMock()
sys.modules["ruamel.yaml"] = MagicMock()
sys.modules["psutil"] = MagicMock()
sys.modules["py3nvml"] = MagicMock()

sys.modules["py3nvml.py3nvml"] = MagicMock()
sys.modules["frigate.version"] = MagicMock()

if __name__ == "__main__":
    unittest.main(module=None, argv=["unittest", "discover", "frigate/test"])
