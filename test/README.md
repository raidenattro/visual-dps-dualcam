# Callback Test Utilities

This folder contains a mock backend endpoint for testing collision callback reporting.

## Run mock backend

From project root:

```bash
python test/mock_java_backend.py
```

If your PDF defines a different callback path, run with environment variable:

```bash
$env:MOCK_CALLBACK_PATH="/your/pdf/callback/path"; python test/mock_java_backend.py
```

The mock service listens on `http://127.0.0.1:8080`.

## Endpoints

- `GET /health` health check
- `POST $MOCK_CALLBACK_PATH` callback endpoint (match PDF path exactly)
- `POST /api/pick/finish` legacy compatible alias
- `GET /api/pick/records` view received payloads

## How to verify callback flow

1. Set callback reporting enabled in `app_config.json`:
   - `reporting.enabled = true`
   - `reporting.callback_url = "http://127.0.0.1:8080/api/pick/finish"`
2. Start this mock backend.
3. Start your main app and trigger collisions.
4. Check:
   - Main app console logs (`[CALLBACK][SEND]`, `[CALLBACK][ACK]`, etc.)
   - `GET http://127.0.0.1:8080/api/pick/records`

## Simulate callback failure

Include `simulateStatus` in callback payload (for testing only), for example `500`.
The mock service will return that HTTP status to trigger retry/failure logic.
