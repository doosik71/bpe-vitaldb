"""
Serve the data/ directory over HTTP using a multi-threaded server.

Python's built-in ``http.server`` is single-threaded: while one large file
is being transferred, the server cannot accept any other connection.  This
script uses ``ThreadingHTTPServer`` so every client request is handled in
its own thread, allowing concurrent downloads.

Usage:
    uv run python scripts/share-data.py [--port PORT] [--bind ADDRESS]

Options:
    --port      TCP port to listen on        (default: 8888)
    --bind      IP address to bind to        (default: 0.0.0.0)
    --data-dir  Directory to serve           (default: data)
"""

import argparse
import logging
import socket
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-threaded HTTP file server for the data/ directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--port",     type=int,  default=8888,
                   help="TCP port (default: 8888)")
    p.add_argument("--bind",     default="0.0.0.0",
                   help="Bind address (default: 0.0.0.0)")
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="Directory to serve (default: data)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        log.error("Directory not found: %s", args.data_dir)
        raise SystemExit(1)

    # Resolve the directory once so the handler always serves the right path
    serve_dir = str(args.data_dir.resolve())

    def handler_factory(*a, **kw):
        return SimpleHTTPRequestHandler(*a, directory=serve_dir, **kw)

    with ThreadingHTTPServer((args.bind, args.port), handler_factory) as httpd:
        # Show all local IPv4 addresses so the user knows where to connect
        hostname = socket.gethostname()
        try:
            lan_ip = socket.gethostbyname(hostname)
        except OSError:
            lan_ip = args.bind

        log.info("Serving %s", serve_dir)
        log.info("Listening on %s:%d", args.bind, args.port)
        log.info("Local  : http://127.0.0.1:%d/",  args.port)
        if lan_ip not in ("0.0.0.0", "127.0.0.1"):
            log.info("Network: http://%s:%d/", lan_ip, args.port)
        log.info("Press Ctrl+C to stop.")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            log.info("Server stopped.")


if __name__ == "__main__":
    main()
