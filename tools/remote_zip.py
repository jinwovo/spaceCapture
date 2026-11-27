"""HTTP Range 요청으로 원격 ZIP에서 필요한 파일만 읽는다(전체 다운로드 없이)."""
import io, urllib.request, zipfile


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "spacecapture-eval"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos+off if whence == 1 else self.size+off
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size-self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.size, self.pos+n)-1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": "spacecapture-eval"})
        with urllib.request.urlopen(req, timeout=120) as r: data = r.read()
        self.pos += len(data); return data
    def readinto(self, b):
        data = self.read(len(b)); b[:len(data)] = data; return len(data)


def open_remote(url):
    return zipfile.ZipFile(io.BufferedReader(RangeFile(url), buffer_size=1 << 20))
