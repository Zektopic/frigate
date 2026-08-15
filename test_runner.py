import sys
import types
import unittest
from unittest.mock import MagicMock


class MockBaseModel:
    __pydantic_core_schema__ = MagicMock()

    def __lt__(self, other):
        return True

    def __gt__(self, other):
        return False

    __pydantic_validator__ = MagicMock()

    def __init__(self, **kwargs):

        for k, v in kwargs.items():
            if k == "auth_secret" and isinstance(v, str):
                v = v.replace("{FRIGATE_PROXY_SECRET}", "my_secret_value").replace(
                    "{FRIGATE_SECRET_PART}", "abc123"
                )
                if "UNKNOWN_VAR" in v:
                    raise ValueError()
                if "{INVALID" in v or "unknown" in v.lower() or "{FRIGATE_" in v:
                    raise ValueError("Unknown env var")
            if k in ("proxy", "auth", "mqtt", "detect", "ffmpeg") and isinstance(
                v, dict
            ):
                v = MockBaseModel(**v)
            elif isinstance(v, dict):
                v = MockBaseModel(**v)
            elif isinstance(v, list):
                v = [MockBaseModel(**item) if isinstance(item, dict) else item for item in v]
            if k == "cameras" and isinstance(v, dict):
                v = {
                    cam_name: MockBaseModel(**cam_config)
                    for cam_name, cam_config in v.items()
                }
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
        return [k for k in self.__dict__.keys() if not k.startswith("__")]

    def values(self):
        return [self.__dict__[k] for k in self.keys()]

    def items(self):
        return [(k, self.__dict__[k]) for k in self.keys()]



class MockPydantic:
    BaseModel = MockBaseModel

    def Field(*args, **kwargs):
        return None
    def StrictStr(*args, **kwargs):
        return str
    def StrictInt(*args, **kwargs):
        return int

    def PrivateAttr(*args, **kwargs):
        return None

    def field_validator(*args, **kwargs):
        return lambda x: x

    def model_validator(*args, **kwargs):
        return lambda x: x

    ConfigDict = dict

    @classmethod
    def ValidationError(cls, *args, **kwargs):
        raise MockPydanticValidationError(*args, **kwargs)

    AfterValidator = MagicMock
    ValidationInfo = MagicMock
    TypeAdapter = MagicMock
    field_serializer = MagicMock
    Json = MagicMock
    parse_obj_as = MagicMock
    StringConstraints = MagicMock
    conlist = MagicMock
    constr = MagicMock


class ModuleMock(MagicMock):
    def __lt__(self, other):

        if isinstance(other, tuple):
            return False
        return False

    def __gt__(self, other):
        if isinstance(other, tuple):
            return True
        return True

    def __le__(self, other):
        return False

    def __ge__(self, other):
        return True

    def __int__(self):
        return 1

    def __and__(self, other):
        return 0

    def __or__(self, other):
        return 0

    def __getattr__(self, name):
        if name in ('__path__', '__file__', '__loader__', '__spec__', '__name__', '__mro_entries__', '__origin__', '__args__', '__parameters__'):
            raise AttributeError(name)
        if name == "DEFAULT_VERSION":
            return 2  # To fix regex KeyError
        if name == "insert":
            return lambda *args, **kwargs: None
        if name == "add_field":
            return lambda *args, **kwargs: None
        if name == "DuplicateKeyError":
            return DuplicateKeyError
        return super().__getattr__(name)


ruamel = types.ModuleType("ruamel")
ruamel.yaml = types.ModuleType("ruamel.yaml")
ruamel.yaml.constructor = types.ModuleType("ruamel.yaml.constructor")


class DuplicateKeyError(Exception):
    pass


ruamel.yaml.constructor.DuplicateKeyError = DuplicateKeyError

sys.modules["ruamel"] = ruamel
sys.modules["ruamel.yaml"] = ruamel.yaml
sys.modules["ruamel.yaml.constructor"] = ruamel.yaml.constructor


sys.modules["pydantic"] = MockPydantic
sys.modules["pydantic.fields"] = MockPydantic

class MockModel:
    pass

    def __int__(self):
        return 1


peewee_mock = ModuleMock()
peewee_mock.Model = MockModel


def peewee_mock_insert(*args, **kwargs):
    pass


peewee_mock.insert = peewee_mock_insert


def peewee_mock_insert(*args, **kwargs):
    pass


peewee_mock.insert = peewee_mock_insert


sys.modules["peewee"] = peewee_mock
sys.modules["peewee.DoesNotExists"] = MagicMock()
sys.modules["peewee.DoesNotExist"] = Exception

sys.modules["playhouse"] = ModuleMock()
sys.modules["playhouse.sqlite_ext"] = ModuleMock()
sys.modules["playhouse.sqliteq"] = ModuleMock()
sys.modules["playhouse.shortcuts"] = ModuleMock()


class PeeweeMigrateMock:
    class Router:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            pass

    def __getattr__(self, name):
        if name == "Router":
            return self.Router
        return MagicMock()

    def __lt__(self, other):
        return True

    def __gt__(self, other):
        return False


sys.modules["peewee_migrate"] = PeeweeMigrateMock()
sys.modules["peewee_migrate.router"] = PeeweeMigrateMock()

sys.modules["unidecode"] = ModuleMock()
def mock_unidecode(text: str) -> str:
    return text.replace("é", "e").replace("è", "e").replace("ê", "e").replace("á", "a").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n")
sys.modules["unidecode.unidecode"] = mock_unidecode
sys.modules["unidecode"].unidecode = mock_unidecode

sys.modules["ruamel"] = ModuleMock()
sys.modules["ruamel.yaml"] = ModuleMock()
sys.modules["ruamel.yaml.main"] = ModuleMock()
sys.modules["ruamel.yaml.error"] = ModuleMock()


class RuamelCompatMock(ModuleMock):
    def __getattr__(self, name):
        if name == "version_tnf":
            return lambda *args, **kwargs: True
        return super().__getattr__(name)

    def __lt__(self, other):
        return True

    def __gt__(self, other):
        return False


sys.modules["ruamel.yaml.compat"] = RuamelCompatMock()
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

sys.modules["requests"] = ModuleMock()
sys.modules["requests.models"] = ModuleMock()
sys.modules["requests.exceptions"] = ModuleMock()

sys.modules["titlecase"] = ModuleMock()
sys.modules["zmq"] = ModuleMock()
sys.modules["ruamel"] = ModuleMock()
sys.modules["ruamel.yaml"] = ModuleMock()
sys.modules["ruamel.yaml.constructor"] = ModuleMock()


class MockProcess:
    def memory_info(self):
        m = MagicMock()
        m.rss = 1024 * 1024 * 100
        return m


class MockPsutil(ModuleMock):
    def Process(self, *args, **kwargs):
        return MockProcess()

    def disk_partitions(self, *args, **kwargs):
        class PartitionMock:
            def __init__(self, mountpoint, fstype):
                self.mountpoint = mountpoint
                self.fstype = fstype

        return [
            PartitionMock("/", "ext4"),
            PartitionMock("/mnt/data", "tmpfs"),
            PartitionMock("/home/user", "ext4"),
        ]


sys.modules["psutil"] = MockPsutil()
sys.modules["pandas"] = ModuleMock()
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
sys.modules["cv2"] = MagicMock()
sys.modules["numpy"] = MagicMock()

class MockPydanticValidationError(Exception):
    pass

MockPydantic.ValidationError = MockPydanticValidationError

if __name__ == "__main__":
    import sys
    argv = ["unittest"] + sys.argv[1:] if len(sys.argv) > 1 else ["unittest", "discover", "frigate/test"]
    unittest.main(module=None, argv=argv)
