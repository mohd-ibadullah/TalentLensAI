import sys
import os
import starlette.middleware.gzip

# 1. Apply Starlette monkeypatch before importing Streamlit
starlette.middleware.gzip.DEFAULT_EXCLUDED_CONTENT_TYPES = ("text/event-stream",)

class MockIdentityResponder:
    def __init__(self, app, minimum_size):
        self.app = app
        self.minimum_size = minimum_size
        self.send = None
        
    async def __call__(self, scope, receive, send):
        self.send = send
        await self.app(scope, receive, self.send_with_compression)
        
    async def send_with_compression(self, message):
        await self.send(message)

starlette.middleware.gzip.IdentityResponder = MockIdentityResponder

# 2. Programmatically launch Streamlit
import streamlit.web.cli as stcli

if __name__ == "__main__":
    app_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "app",
        "streamlit_app.py"
    )
    
    # Configure arguments for headless running
    sys.argv = [
        "streamlit", 
        "run", 
        app_path, 
        "--server.port", "8501", 
        "--server.headless", "true"
    ]
    
    print(f"Launching Streamlit App from: {app_path}")
    sys.exit(stcli.main())
