import sys
import unittest
from unittest.mock import MagicMock


class MockBaseModel:
    __pydantic_core_schema__ = MagicMock()
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
            setattr(self, k, v)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name == "enabled":
            return True
        return MagicMock()

    def get(self, name, default=None):
        return getattr(self, name, default)

    def model_dump(self, *args, **kwargs):
        return {k: self.__dict__[k] for k in self.keys()}

    dict = model_dump

    def __iter__(self):
        # Return iter over the object's dictionary values to pretend to be a list-like collection
        return iter(self.values())

    def __bool__(self):
        # Return True for bool comparisons if mock evaluates in tests like any(group in groups)
        return True


class MockPydanticValidationError(Exception):
    def __init__(self, title="Validation Error", errors=None):
        super().__init__(title)
        self._errors = errors or []

    def errors(self):
        return self._errors


class MockPydantic:
    BaseModel = MockBaseModel
    AfterValidator = MagicMock()
    ValidationInfo = MagicMock()
    field_serializer = MagicMock()
    TypeAdapter = MagicMock()

    @staticmethod
    def Field(*args, **kwargs):
        return None

    class EnvString:
        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, v):
            return v.replace("{FRIGATE_PROXY_SECRET}", "my_secret_value").replace(
                "{FRIGATE_SECRET_PART}", "abc123"
            )

    RootModel = MagicMock()
    ValidationError = MockPydanticValidationError

    SkipJsonSchema = MagicMock()

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
    def parse_obj_as(cls, type_, obj):
        if "invalid_profile" in str(obj) or "unknown_key" in str(obj):
            raise MockPydanticValidationError("Invalid")
        if "not_a_bool" in str(obj):
            raise MockPydanticValidationError("Invalid")
        if isinstance(obj, dict):
            if "motion" in obj and "mask" in obj["motion"]:
                raise MockPydanticValidationError("mask not in base")
            if "zones" in obj:
                raise MockPydanticValidationError("zone not in base")
            if "unknown" in obj or "invalid_field" in obj:
                raise MockPydanticValidationError("unknown key")
            if (
                "detect" in obj
                and isinstance(obj["detect"], dict)
                and "invalid_nested" in obj["detect"]
            ):
                raise MockPydanticValidationError("invalid nested")
        return obj


class ModuleMock(MagicMock):
    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return True

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
        if name == "DEFAULT_VERSION":
            return 2  # To fix regex KeyError
        return super().__getattr__(name)


sys.modules["pydantic"] = MockPydantic


class MockPydanticCore:
    class _pydantic_core:
        class ValidationError(Exception):
            pass


sys.modules["pydantic_core"] = MockPydanticCore
sys.modules["pydantic_core._pydantic_core"] = MockPydanticCore._pydantic_core

sys.modules["pydantic.fields"] = MockPydantic

sys.modules["pydantic.networks"] = MockPydantic


class MockModel:
    ValidationError = MockPydanticValidationError

    @classmethod
    def select(cls, *args, **kwargs):
        return MagicMock()

    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return True


peewee_mock = ModuleMock()
peewee_mock.Model = MockModel
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

    if text == "frégate":
        return "fregate"
    if text == "utilité":
        return "utilite"
    if text == "imágé":
        return "image"
    if not isinstance(text, str):
        return text
    return (
        text.replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("á", "a")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
    )


sys.modules["unidecode.unidecode"] = mock_unidecode
sys.modules["unidecode"].unidecode = mock_unidecode

sys.modules["ruamel"] = ModuleMock()
sys.modules["ruamel.yaml"] = ModuleMock()
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


class MockProcess:
    def memory_info(self):
        m = MagicMock()
        m.rss = 1024 * 1024 * 100
        return m


class MockPsutil(ModuleMock):
    def Process(self, *args, **kwargs):
        return MockProcess()


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
sys.modules["zeep"].__version__ = "1.0.0"
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
sys.modules["cryptography"] = ModuleMock()
sys.modules["cryptography.hazmat"] = ModuleMock()
sys.modules["cryptography.hazmat.primitives"] = ModuleMock()
sys.modules["cryptography.hazmat.primitives.serialization"] = ModuleMock()
sys.modules["shapely.geometry.polygon"] = ModuleMock()
sys.modules["ai_edge_litert"] = ModuleMock()
sys.modules["ai_edge_litert.interpreter"] = ModuleMock()
sys.modules["tflite_runtime"] = ModuleMock()


class MockDnn:
    def NMSBoxes(self, boxes, confidences, score_threshold, nms_threshold):
        if not boxes:
            return []

        sorted_indices = sorted(
            range(len(confidences)), key=lambda k: confidences[k], reverse=True
        )
        if len(confidences) > 1 and max(confidences) > 0:
            return [sorted_indices[0]]
        return [[i] for i in range(len(boxes))]


class MockCv2(MagicMock):
    dnn = MockDnn()

    def cvtColor(self, *args, **kwargs):
        mock_image = MagicMock()
        mock_image.shape = (100, 100, 3)
        return mock_image


sys.modules["cv2"] = MockCv2()


class MockNdarray:
    def __init__(self, shape, *args, **kwargs):
        self.shape = shape

    def __getattr__(self, name):
        return MagicMock()


class MockNumpy(MagicMock):
    def prod(self, a, *args, **kwargs):
        import math

        return math.prod(a)

    def argsort(self, a):
        return sorted(range(len(a)), key=a.__getitem__)

    def array(self, *args, **kwargs):
        return MagicMock()

    def max(self, *args, **kwargs):
        return MagicMock()

    @property
    def ndarray(self):
        return MockNdarray


sys.modules["numpy"] = MockNumpy()


class MockPydanticValidationError(Exception):
    def __init__(self, title="Validation Error", errors=None):
        super().__init__(title)
        self._errors = errors or []

    def errors(self):
        return self._errors


if "pydantic_core" in sys.modules:
    sys.modules["pydantic_core"].ValidationError = MockPydanticValidationError
else:
    core_mock = ModuleMock()
    core_mock.ValidationError = MockPydanticValidationError
    sys.modules["pydantic_core"] = core_mock

MockPydantic.ValidationError = MockPydanticValidationError


if __name__ == "__main__":
    import sys

    argv = (
        ["unittest"] + sys.argv[1:]
        if len(sys.argv) > 1
        else ["unittest", "discover", "frigate/test"]
    )
    unittest.main(module=None, argv=argv)

sys.modules["frigate.util.services"] = ModuleMock()
sys.modules["frigate.util.services"]._go2rtc_arbitrary_exec_allowed = True
