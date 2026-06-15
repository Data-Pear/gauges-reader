from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("GAUGE_API_HOST", "0.0.0.0")
    port = int(os.getenv("GAUGE_API_PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port)


if __name__ == "__main__":
    main()

