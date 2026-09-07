from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette_context import middleware, plugins, context
from frigate.api.fastapi_app import RemoteUserPlugin

app = FastAPI()
app.add_middleware(
    middleware.ContextMiddleware,
    plugins=(RemoteUserPlugin(),)
)

@app.get("/")
def read_main(request: Request):
    return {"user": context.get("Remote-User")}

client = TestClient(app)
response = client.get("/", headers={"Remote-User": "testuser"})
print(response.json())
