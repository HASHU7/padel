"""
High-Performance Padel AI Video & Analytics HTTP Server
Supports HTTP 206 Partial Content (Byte-Range Requests)
Enables instant scrubbing & seeking across large 2.9GB/4.2GB video match recordings.
"""
import sys
import os
import http.server
import socketserver
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PadelVideoRangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def end_headers(self):
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

    def send_head(self):
        if 'Range' not in self.headers:
            return super().send_head()

        path = self.translate_path(self.path)
        f = None
        try:
            f = open(path, 'rb')
        except OSError:
            self.send_error(http.server.HTTPStatus.NOT_FOUND, 'File not found')
            return None

        fs = os.fstat(f.fileno())
        total_len = fs.st_size
        range_header = self.headers.get('Range').strip()

        if not range_header.startswith('bytes='):
            self.send_error(http.server.HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            f.close()
            return None

        range_val = range_header[6:].split('-')
        start = int(range_val[0]) if range_val[0] else 0
        end = int(range_val[1]) if len(range_val) > 1 and range_val[1] else total_len - 1

        if start >= total_len or end >= total_len or start > end:
            self.send_error(http.server.HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            f.close()
            return None

        self.send_response(http.server.HTTPStatus.PARTIAL_CONTENT)
        self.send_header('Content-type', self.guess_type(path))
        self.send_header('Content-Range', f'bytes {start}-{end}/{total_len}')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Last-Modified', self.date_time_string(fs.st_mtime))
        self.end_headers()

        f.seek(start)

        class SubFile:
            def __init__(self, file_obj, length):
                self.f = file_obj
                self.remaining = length

            def read(self, size=-1):
                if self.remaining <= 0:
                    return b''
                read_size = min(size, self.remaining) if size > 0 else self.remaining
                data = self.f.read(read_size)
                self.remaining -= len(data)
                return data

            def close(self):
                self.f.close()

        return SubFile(f, end - start + 1)


if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


from http.server import ThreadingHTTPServer


def run_server(port=8888):
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, PadelVideoRangeRequestHandler)
    print(f"Padel Video & AI Coach Server running at http://127.0.0.1:{port}/live_preview.html")
    print("Multi-threaded HTTP 206 Partial Content enabled for ultra-fast match video seeking.")
    httpd.serve_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    run_server(port)
