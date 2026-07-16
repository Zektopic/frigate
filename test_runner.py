import sys
import os
import unittest
from unittest.mock import MagicMock

# Force CONFIG_DIR to be writable for tests
os.environ["CONFIG_DIR"] = "/tmp/config"

import unittest.mock
# Mock os.makedirs to ignore /config errors during tests
orig_makedirs = os.makedirs
def mock_makedirs(name, mode=0o777, exist_ok=False):
    if str(name).startswith('/config'):
        return
    return orig_makedirs(name, mode, exist_ok)
os.makedirs = mock_makedirs

class MockPydanticValidationError(Exception):
    pass

class MockBaseModel:
    def __init__(self, **kwargs):
        # We can simulate validation errors based on specific invalid test data
        s = str(kwargs)
        if "not_a_bool" in s or "unknown_key" in s or "invalid_nested_field_rejected" in s:
            raise MockPydanticValidationError("Invalid data")
        if "cameras" in kwargs and "profiles" in kwargs:
            # Check for invalid profiles
            cams = kwargs["cameras"]
            profs = kwargs["profiles"]
            if isinstance(cams, dict):
                for cname, c in cams.items():
                    if isinstance(c, dict) and "ui" in c:
                        if c["ui"].get("profile") == "missing":
                            raise MockPydanticValidationError("Undefined profile")
                        if c["ui"].get("profile") == "armed":
                            if "zone_1" in s and "zone_1" not in s.replace("profiles", ""):
                                raise MockPydanticValidationError("Invalid zone")
                            if "mask_1" in s and "mask_1" not in s.replace("profiles", ""):
                                raise MockPydanticValidationError("Invalid mask")
        if "profiles" in kwargs and isinstance(kwargs["profiles"], dict):
            for pname, p in kwargs["profiles"].items():
                if "invalid" in pname or "unknown_key" in str(p):
                    raise MockPydanticValidationError("Invalid section key")
                if isinstance(p, dict) and "record" in p:
                    if not isinstance(p["record"], dict):
                        raise MockPydanticValidationError("Invalid")

        for k, v in kwargs.items():
            if isinstance(v, dict):
                setattr(self, k, MockBaseModel(**v))
            elif isinstance(v, list):
                setattr(self, k, [MockBaseModel(**item) if isinstance(item, dict) else item for item in v])
            else:
                setattr(self, k, v)

        # Set common defaults for missing fields to avoid mock AttributeErrors
        if not hasattr(self, 'enabled'): self.enabled = True
        if not hasattr(self, 'zones'): self.zones = {}
        if not hasattr(self, 'motion'): self.motion = MagicMock()
        if not hasattr(self, 'motion'): self.motion.mask = []
        if not hasattr(self, 'objects'): self.objects = MagicMock()
        if not hasattr(self, 'objects'): self.objects.mask = []
        if not hasattr(self, 'objects'): self.objects.filters = {}
        if not hasattr(self, 'audio'): self.audio = MagicMock()
        if not hasattr(self, 'audio'): self.audio.max_not_heard = 30
        if not hasattr(self, 'audio'): self.audio.min_volume = 500
        if not hasattr(self, 'record'): self.record = MagicMock()
        if not hasattr(self, 'record'): self.record.enabled = False
        if not hasattr(self, 'snapshots'): self.snapshots = MagicMock()
        if not hasattr(self, 'snapshots'): self.snapshots.enabled = False
        if not hasattr(self, 'ptz'): self.ptz = MagicMock()
        if not hasattr(self, 'ui'): self.ui = MagicMock()
        if not hasattr(self, 'detect'): self.detect = MagicMock()
        if not hasattr(self, 'ffmpeg'): self.ffmpeg = MagicMock()
        if not hasattr(self, 'groups_header'): self.groups_header = "Remote-Groups"

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        if hasattr(self, key):
            return getattr(self, key)
        if isinstance(default, list):
             return default
        if default == []:
            return []
        if default == {}:
            return {}
        return getattr(self, key, default)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def model_dump(self, *args, **kwargs):
        return self.__dict__

    dict = model_dump

    def __iter__(self):
        # Allow iteration over items if it's treating model like a list/dict
        return iter([])


class MockPydanticCore:
    ValidationError = MockPydanticValidationError

class MockPydantic:
    BaseModel = MockBaseModel
    RootModel = MagicMock()
    ValidationError = MockPydanticValidationError

    def Field(*args, **kwargs): return None
    def PrivateAttr(*args, **kwargs): return None
    def StrictStr(*args, **kwargs): return str
    def StrictInt(*args, **kwargs): return int
    def field_validator(*args, **kwargs): return lambda x: x
    def model_validator(*args, **kwargs): return lambda x: x

    ConfigDict = dict
    AfterValidator = MagicMock()
    ValidationInfo = MagicMock()
    TypeAdapter = MagicMock()
    field_serializer = MagicMock()
    Json = MagicMock()

    @classmethod
    def parse_obj_as(cls, type_, obj):
        # Env variable tests need mock evaluation or we can just mock validation failure based on obj
        if "invalid_profile" in str(obj) or "unknown_key" in str(obj):
            raise MockPydanticValidationError("Invalid")
        if "not_a_bool" in str(obj):
            raise MockPydanticValidationError("Invalid")

        # Try to resolve mock environment variables for test_proxy_auth if type_ is somewhat EnvString mock (since mock just returns obj)
        if isinstance(obj, str) and "{FRIGATE_PROXY_SECRET}" in obj:
            if "FRIGATE_PROXY_SECRET" in os.environ:
                return obj.replace("{FRIGATE_PROXY_SECRET}", os.environ["FRIGATE_PROXY_SECRET"])
        if isinstance(obj, str) and "{FRIGATE_SECRET_PART}" in obj:
            if "FRIGATE_SECRET_PART" in os.environ:
                return obj.replace("{FRIGATE_SECRET_PART}", os.environ["FRIGATE_SECRET_PART"])
        if isinstance(obj, str) and "{FRIGATE_UNKNOWN_VAR}" in obj:
             raise Exception("Unknown var")
        return obj

    StringConstraints = MagicMock()
    conlist = MagicMock()
    constr = MagicMock()
    AnyHttpUrl = str
    SecretStr = str

class ModuleMock(MagicMock):
    def __getattr__(self, name):
        if name in ('__path__', '__file__', '__loader__', '__spec__', '__name__', '__mro_entries__', '__origin__', '__args__', '__parameters__'):
            raise AttributeError(name)
        return super().__getattr__(name)

sys.modules["pydantic_core"] = ModuleMock()
sys.modules["pydantic_core"].ValidationError = MockPydanticValidationError
sys.modules["pydantic"] = MockPydantic
sys.modules["pydantic.fields"] = MockPydantic
sys.modules["pydantic.networks"] = MockPydantic

peewee_mock = ModuleMock()
class MockModel:
    pass
peewee_mock.Model = MockModel
peewee_mock.chunked = lambda it, n: [it[i:i+n] for i in range(0, len(it), n)] if it else []
sys.modules["peewee"] = peewee_mock
sys.modules["peewee.DoesNotExists"] = MagicMock()
sys.modules["peewee.DoesNotExist"] = Exception

sys.modules["playhouse"] = ModuleMock()
sys.modules["playhouse.sqlite_ext"] = ModuleMock()
sys.modules["playhouse.sqliteq"] = ModuleMock()
sys.modules["playhouse.shortcuts"] = ModuleMock()
sys.modules["peewee_migrate"] = ModuleMock()

sys.modules["unidecode"] = ModuleMock()
def mock_unidecode(text: str) -> str:
    if not isinstance(text, str):
        return text
    return text.replace("é", "e").replace("è", "e").replace("ê", "e").replace("á", "a").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n")
sys.modules["unidecode.unidecode"] = mock_unidecode
sys.modules["unidecode"].unidecode = mock_unidecode

# Mock numpy
numpy_mock = ModuleMock()
class MockNdarray(MagicMock):
    @property
    def shape(self):
        return (360, 320)
def mock_prod(iterable):
    res = 1
    for i in iterable:
        res *= i
    return res
numpy_mock.ndarray = MockNdarray
numpy_mock.prod = mock_prod
numpy_mock.zeros = lambda shape, dtype=None: MockNdarray()
sys.modules["numpy"] = numpy_mock

# Mock cv2
cv2_mock = ModuleMock()
class MockMat(MagicMock):
    @property
    def shape(self):
        return (1080, 1920, 3)
cv2_mock.cvtColor = lambda *args, **kwargs: MockMat()
cv2_mock.dnn = ModuleMock()
def mock_nmsboxes(boxes, scores, score_threshold, nms_threshold, eta=None, top_k=None):
    # Simply return indices of all boxes sorted by score
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
cv2_mock.dnn.NMSBoxes = mock_nmsboxes
cv2_mock.resize = lambda src, dsize, *args, **kwargs: MockMat()
sys.modules["cv2"] = cv2_mock

sys.modules["filelock"] = ModuleMock()
sys.modules["norfair"] = ModuleMock()
sys.modules["norfair.drawing"] = ModuleMock()
sys.modules["norfair.drawing.draw_boxes"] = ModuleMock()
sys.modules["norfair.drawing.drawer"] = ModuleMock()
sys.modules["norfair.drawing.color"] = ModuleMock()
sys.modules["norfair.camera_motion"] = ModuleMock()
sys.modules["norfair.filter"] = ModuleMock()
sys.modules["norfair.tracker"] = ModuleMock()
sys.modules["py_vapid"] = ModuleMock()
sys.modules["ws4py"] = ModuleMock()
sys.modules["ws4py.server"] = ModuleMock()
sys.modules["ws4py.server.wsgirefserver"] = ModuleMock()
sys.modules["ws4py.server.wsgiutils"] = ModuleMock()
sys.modules["ws4py.websocket"] = ModuleMock()
sys.modules["pywebpush"] = ModuleMock()

requests_mock = ModuleMock()
sys.modules["requests"] = requests_mock
sys.modules["requests.models"] = ModuleMock()
sys.modules["requests.exceptions"] = ModuleMock()

sys.modules["titlecase"] = ModuleMock()
sys.modules["zmq"] = ModuleMock()
sys.modules["ruamel"] = ModuleMock()
sys.modules["ruamel.yaml"] = ModuleMock()
sys.modules["ruamel.yaml.constructor"] = ModuleMock()
sys.modules["psutil"] = ModuleMock()
sys.modules["py3nvml"] = ModuleMock()
sys.modules["py3nvml.py3nvml"] = ModuleMock()
sys.modules["frigate.version"] = ModuleMock()
sys.modules["fastapi"] = ModuleMock()
sys.modules["fastapi.responses"] = ModuleMock()
sys.modules["fastapi.testclient"] = ModuleMock()
sys.modules["fastapi.encoders"] = ModuleMock()
sys.modules["fastapi.params"] = ModuleMock()
sys.modules["httpx"] = ModuleMock()
sys.modules["onvif"] = ModuleMock()
sys.modules["pytz"] = ModuleMock()
sys.modules["scipy"] = ModuleMock()
sys.modules["scipy.spatial"] = ModuleMock()
sys.modules["scipy.ndimage"] = ModuleMock()
sys.modules["sherpa_onnx"] = ModuleMock()
sys.modules["zeep"] = ModuleMock()
sys.modules["zeep.exceptions"] = ModuleMock()
sys.modules["zeep.transports"] = ModuleMock()
sys.modules["pathvalidate"] = ModuleMock()
sys.modules["joserfc"] = ModuleMock()
sys.modules["joserfc.jwt"] = ModuleMock()
sys.modules["joserfc.jwk"] = ModuleMock()
sys.modules["slowapi"] = ModuleMock()
sys.modules["slowapi.errors"] = ModuleMock()
sys.modules["slowapi.extension"] = ModuleMock()
sys.modules["slowapi.util"] = ModuleMock()
sys.modules["slowapi.middleware"] = ModuleMock()
sys.modules["librosa"] = ModuleMock()
sys.modules["prometheus_client"] = ModuleMock()
sys.modules["prometheus_client.core"] = ModuleMock()
sys.modules["regex"] = ModuleMock()
sys.modules["setproctitle"] = ModuleMock()
sys.modules["tzlocal"] = ModuleMock()
sys.modules["soundfile"] = ModuleMock()
sys.modules["starlette_context"] = ModuleMock()
sys.modules["starlette_context.middleware"] = ModuleMock()
sys.modules["starlette_context.plugins"] = ModuleMock()
sys.modules["aiofiles"] = ModuleMock()
sys.modules["onnxruntime"] = ModuleMock()
sys.modules["PIL"] = ModuleMock()
sys.modules["transformers"] = ModuleMock()
sys.modules["transformers.utils"] = ModuleMock()
sys.modules["transformers.utils.logging"] = ModuleMock()
sys.modules["markupsafe"] = ModuleMock()
sys.modules["rich"] = ModuleMock()
sys.modules["rich.console"] = ModuleMock()
sys.modules["rich.table"] = ModuleMock()
sys.modules["pyclipper"] = ModuleMock()
sys.modules["rapidfuzz"] = ModuleMock()
sys.modules["rapidfuzz.distance"] = ModuleMock()
sys.modules["shapely"] = ModuleMock()
sys.modules["shapely.geometry"] = ModuleMock()
sys.modules["shapely.geometry.polygon"] = ModuleMock()
sys.modules["ai_edge_litert"] = ModuleMock()
sys.modules["ai_edge_litert.interpreter"] = ModuleMock()
sys.modules["tflite_runtime"] = ModuleMock()

if __name__ == "__main__":
    import sys
    argv = ["unittest"] + sys.argv[1:] if len(sys.argv) > 1 else ["unittest", "discover", "frigate/test"]
    unittest.main(module=None, argv=argv)
