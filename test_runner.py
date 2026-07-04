import sys
import unittest
from unittest.mock import MagicMock


class MockBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                setattr(self, k, MockBaseModel(**v))
            elif isinstance(v, list):
                setattr(self, k, [MockBaseModel(**item) if isinstance(item, dict) else item for item in v])
            else:
                setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()


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
    AfterValidator = MagicMock
    ValidationInfo = MagicMock
    TypeAdapter = MagicMock
    field_serializer = MagicMock
    Json = MagicMock
    parse_obj_as = MagicMock
    StringConstraints = MagicMock
    conlist = MagicMock
    constr = MagicMock


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
class MockModel:
    pass

peewee_mock = MagicMock()
peewee_mock.Model = MockModel
sys.modules["peewee"] = peewee_mock
sys.modules["peewee.DoesNotExists"] = MagicMock()
sys.modules["playhouse"] = MagicMock()
sys.modules["playhouse.sqlite_ext"] = MagicMock()
sys.modules["unidecode"] = MagicMock()
sys.modules["filelock"] = MagicMock()
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
sys.modules["fastapi"] = MagicMock()
sys.modules["fastapi.responses"] = MagicMock()
sys.modules["fastapi.testclient"] = MagicMock()
sys.modules["httpx"] = MagicMock()
sys.modules["norfair.camera_motion"] = MagicMock()
sys.modules["onvif"] = MagicMock()
sys.modules["peewee_migrate"] = MagicMock()
sys.modules["pytz"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.spatial"] = MagicMock()
sys.modules["sherpa_onnx"] = MagicMock()
sys.modules["zeep"] = MagicMock()
sys.modules["zeep.exceptions"] = MagicMock()
sys.modules["zeep.transports"] = MagicMock()
sys.modules["pathvalidate"] = MagicMock()
sys.modules["joserfc"] = MagicMock()
sys.modules["joserfc.jwt"] = MagicMock()
sys.modules["joserfc.jwk"] = MagicMock()
sys.modules["playhouse.sqliteq"] = MagicMock()
sys.modules["ruamel.yaml.constructor"] = MagicMock()
sys.modules["slowapi"] = MagicMock()
sys.modules["slowapi.errors"] = MagicMock()
sys.modules["slowapi.extension"] = MagicMock()
sys.modules["slowapi.util"] = MagicMock()
sys.modules["slowapi.middleware"] = MagicMock()
sys.modules["librosa"] = MagicMock()
sys.modules["prometheus_client"] = MagicMock()
sys.modules["regex"] = MagicMock()
sys.modules["scipy.ndimage"] = MagicMock()
sys.modules["setproctitle"] = MagicMock()
sys.modules["tzlocal"] = MagicMock()
sys.modules["prometheus_client.core"] = MagicMock()
sys.modules["soundfile"] = MagicMock()
sys.modules["starlette_context"] = MagicMock()
sys.modules["starlette_context.middleware"] = MagicMock()
sys.modules["starlette_context.plugins"] = MagicMock()
sys.modules["aiofiles"] = MagicMock()
sys.modules["norfair.drawing.draw_boxes"] = MagicMock()
sys.modules["onnxruntime"] = MagicMock()
sys.modules["requests.exceptions"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["fastapi.encoders"] = MagicMock()
sys.modules["norfair.drawing.drawer"] = MagicMock()
sys.modules["fastapi.params"] = MagicMock()
sys.modules["norfair.filter"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["markupsafe"] = MagicMock()
sys.modules["norfair.tracker"] = MagicMock()
sys.modules["transformers.utils"] = MagicMock()
sys.modules["rich"] = MagicMock()
sys.modules["transformers.utils.logging"] = MagicMock()
sys.modules["pyclipper"] = MagicMock()
sys.modules["rich.console"] = MagicMock()
sys.modules["rapidfuzz"] = MagicMock()
sys.modules["rich.table"] = MagicMock()
sys.modules["rapidfuzz.distance"] = MagicMock()
sys.modules["shapely"] = MagicMock()
sys.modules["shapely.geometry"] = MagicMock()
sys.modules["shapely.geometry.polygon"] = MagicMock()
sys.modules["ai_edge_litert"] = MagicMock()
sys.modules["ai_edge_litert.interpreter"] = MagicMock()
sys.modules["tflite_runtime"] = MagicMock()

class MockPydanticValidationError(Exception):
    pass
MockPydantic.ValidationError = MockPydanticValidationError

if __name__ == "__main__":
    import sys
    argv = ["unittest"] + sys.argv[1:] if len(sys.argv) > 1 else ["unittest", "discover", "frigate/test"]
    unittest.main(module=None, argv=argv)
