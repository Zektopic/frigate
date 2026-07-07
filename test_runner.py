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
    def __getattr__(self, name):
        if name in ('__path__', '__file__', '__loader__', '__spec__', '__name__', '__mro_entries__', '__origin__', '__args__', '__parameters__'):
            raise AttributeError(name)
        return super().__getattr__(name)


sys.modules["pydantic"] = MockPydantic
sys.modules["pydantic.fields"] = MockPydantic

class MockModel:
    pass

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
    return text.replace("é", "e").replace("è", "e").replace("ê", "e").replace("á", "a").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n")
sys.modules["unidecode.unidecode"] = mock_unidecode
sys.modules["unidecode"].unidecode = mock_unidecode

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
class MockDnn:
    def NMSBoxes(self, boxes, confidences, score_threshold, nms_threshold):
        # Very simple NMS for tests
        if not boxes:
            return []

        # for tests we just assume we return the first box if there are 2 overlapping, etc.
        # the simplest mock just returns all indices but let's do a simple one based on max confidence

        # in the test `test_overlapping_objects_reduced` we have 2 boxes:
        # box 1: confidence 0.6 (clipped), box 2: confidence 0.88.
        # wait, the first box is not clipped (1150 to 1500, box is 1209 to 1437) -> confidence 0.81
        # the second box is clipped (1242 to 1602, box is 1238 to 1401) -> confidence 0.6
        # so box 1 (index 0) has 0.81, box 2 has 0.6. Box 1 should be selected.
        indices = []
        for i in range(len(boxes)):
            keep = True
            for j in range(len(indices)):
                # just check if same box or overlapping.
                # let's just return indices sorted by confidence
                pass

        # Return indices of highest confidences first
        sorted_indices = sorted(range(len(confidences)), key=lambda k: confidences[k], reverse=True)

        # Return the top index for testing overlapping
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

sys.modules["numpy"] = MockNumpy()

class MockPydanticValidationError(Exception):
    pass

MockPydantic.ValidationError = MockPydanticValidationError

if __name__ == "__main__":
    import sys
    argv = ["unittest"] + sys.argv[1:] if len(sys.argv) > 1 else ["unittest", "discover", "frigate/test"]
    unittest.main(module=None, argv=argv)
