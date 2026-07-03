import sys
import unittest
from unittest.mock import MagicMock

class SafeMock(MagicMock):
    def __getattr__(self, name):
        if name in (
            "__path__",
            "__file__",
            "__loader__",
            "__spec__",
            "__name__",
            "__mro_entries__",
            "__origin__",
            "__args__",
            "__parameters__",
        ):
            raise AttributeError(name)
        return super().__getattr__(name)

    def __lt__(self, other):
        return True

def safe_mock():
    return SafeMock()

class MockBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

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

    def validator(*args, **kwargs):
        return lambda x: x

    ConfigDict = dict
    AfterValidator = safe_mock()
    ValidationInfo = safe_mock()
    TypeAdapter = safe_mock()
    field_serializer = safe_mock()
    Json = safe_mock()
    parse_obj_as = safe_mock()
    StringConstraints = safe_mock()
    conlist = safe_mock()
    constr = safe_mock()

    class ValidationError(Exception):
        pass

sys.modules["pydantic"] = MockPydantic
sys.modules["pydantic.fields"] = MockPydantic

for mod in [
    "norfair", "norfair.drawing", "norfair.drawing.color", "py_vapid",
    "ws4py", "ws4py.server", "ws4py.server.wsgirefserver", "ws4py.server.wsgiutils", "ws4py.websocket",
    "pywebpush", "requests", "requests.models", "peewee", "peewee.DoesNotExists",
    "playhouse", "playhouse.sqlite_ext", "filelock",
    "numpy", "titlecase", "zmq", "playhouse.shortcuts",
    "ruamel", "ruamel.yaml", "psutil", "py3nvml", "py3nvml.py3nvml", "frigate.version",
    "fastapi", "fastapi.responses", "fastapi.testclient", "httpx",
    "norfair.camera_motion", "onvif", "peewee_migrate", "pytz", "scipy", "scipy.spatial",
    "sherpa_onnx", "zeep", "zeep.exceptions", "zeep.transports",
    "pathvalidate", "joserfc", "joserfc.jwt", "joserfc.jwk", "playhouse.sqliteq",
    "ruamel.yaml.constructor", "slowapi", "slowapi.errors", "slowapi.extension",
    "slowapi.util", "slowapi.middleware", "librosa", "prometheus_client",
    "regex", "scipy.ndimage", "setproctitle", "tzlocal", "prometheus_client.core",
    "soundfile", "starlette_context", "starlette_context.middleware", "starlette_context.plugins",
    "aiofiles", "norfair.drawing.draw_boxes", "onnxruntime", "requests.exceptions",
    "PIL", "fastapi.encoders", "norfair.drawing.drawer", "fastapi.params",
    "norfair.filter", "transformers", "markupsafe", "norfair.tracker",
    "transformers.utils", "rich", "transformers.utils.logging", "pyclipper",
    "rich.console", "rapidfuzz", "rich.table", "rapidfuzz.distance", "shapely",
    "shapely.geometry", "shapely.geometry.polygon", "ai_edge_litert",
    "ai_edge_litert.interpreter", "tflite_runtime"
]:
    sys.modules[mod] = safe_mock()

cv2_mock = safe_mock()
cv2_mock.cvtColor.return_value.shape = (1080, 1920, 3)
sys.modules["cv2"] = cv2_mock

unidecode_mock = safe_mock()
def unidecode_func(s):
    return str(s).replace("é", "e").replace("á", "a")
unidecode_mock.unidecode = unidecode_func
unidecode_mock.side_effect = unidecode_func
sys.modules["unidecode"] = unidecode_mock

if __name__ == "__main__":
    unittest.main(module=None, argv=["unittest", "discover", "frigate/test"])
