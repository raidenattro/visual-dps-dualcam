from app_factory import create_app
import uvicorn

app, APP_CONFIG = create_app()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=APP_CONFIG["server"].get("host", "0.0.0.0"),
        port=int(APP_CONFIG["server"].get("port", 8045)),
    )
