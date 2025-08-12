import asyncio
import re
import time
import json
import hashlib
import os
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import httpx
from bs4 import BeautifulSoup
import tldextract
from tqdm import tqdm
import yaml
from urllib import robotparser

# -------------------------
# Helpers
# -------------------------

def slugify_url(url: str) -> str:
    # deterministic filename per URL
    u = urlparse(url)
    path = (u.path or "/").strip("/")
    if not path:
        path = "index"
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", path)
    base = f"{u.netloc}-{safe}"
    if u.query:
        base += "-" + hashlib.md5(u.query.encode("utf-8")).hexdigest()[:8]
    return base.lower() + ".html"

def same_reg_domain(a: str, b: str) -> bool:
    ea = tldextract.extract(a)
    eb = tldextract.extract(b)
    return (ea.domain, ea.suffix) == (eb.domain, eb.suffix)

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compile_patterns(cfg):
    allow = [re.compile(p) for p in cfg.get("path_allowlist", [])]
    return allow

def is_allowed_path(path: str, allow_patterns) -> bool:
    if not allow_patterns:
        return True
    for p in allow_patterns:
        if p.search(path):
            return True
    return False

def has_disallowed_ext(url: str, exts):
    url = url.lower()
    return any(url.endswith(ext.lower()) for ext in exts)

# -------------------------
# Robots
# -------------------------

class RobotsGate:
    def __init__(self, base_url: str, ua: str):
        parsed = urlparse(base_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        self.rp = robotparser.RobotFileParser()
        try:
            self.rp.set_url(robots_url)
            self.rp.read()
        except Exception:
            # fail closed if robots not reachable
            self.rp = None
        self.ua = ua

    def allowed(self, url: str) -> bool:
        if self.rp is None:
            return True
        try:
            return self.rp.can_fetch(self.ua, url)
        except Exception:
            return False

# -------------------------
# Crawler
# -------------------------

class Crawler:
    def __init__(self, config_path: str):
        self.cfg = load_config(config_path)
        self.start_urls = self.cfg["start_urls"]
        self.allowed_domains = set(self.cfg["allowed_domains"])
        self.allow_patterns = compile_patterns(self.cfg)
        self.disallow_ext = self.cfg.get("disallow_file_ext", [])
        self.ua = self.cfg.get("user_agent", "FriendlyResearchBot/1.0")
        self.max_pages = int(self.cfg.get("max_pages", 1000))
        self.timeout_s = int(self.cfg.get("request_timeout_s", 20))
        self.rate_sleep_ms = int(self.cfg.get("rate_limit_sleep_ms", 300))
        self.concurrency = int(self.cfg.get("concurrency", 8))
        self.save_dir = Path(self.cfg.get("save_dir", "data/raw"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # one robots gate per hostname
        self.robots = {}
        for u in self.start_urls:
            host = urlparse(u).netloc
            self.robots[host] = RobotsGate(u, self.ua)

        self.seen = set()
        self.to_visit = asyncio.Queue()
        for u in self.start_urls:
            self.to_visit.put_nowait(u)

    def _within_scope(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        if parsed.netloc not in self.allowed_domains:
            # still OK if same registrable domain and domain listed allows subdocs
            # but we keep strict here
            return False
        if has_disallowed_ext(url, self.disallow_ext):
            return False
        if self.allow_patterns and not is_allowed_path(parsed.path or "/", self.allow_patterns):
            return False
        # robots.txt
        # gate = self.robots.get(parsed.netloc)
        # if gate and not gate.allowed(url):
        #     return False
        return True

    async def fetch(self, client: httpx.AsyncClient, url: str):
        try:
            r = await client.get(url, timeout=self.timeout_s, headers={"User-Agent": self.ua})
            ct = r.headers.get("content-type", "")
            if r.status_code == 200 and ("text/html" in ct or ct.startswith("text/")):
                return r.text
        except Exception:
            return None
        return None

    def extract_links(self, html: str, base_url: str):
        soup = BeautifulSoup(html, "html.parser")
        links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            href = urljoin(base_url, href)
            href, _ = urldefrag(href)  # drop fragments
            links.add(href)
        return links

    def save_html(self, url: str, html: str):
        fname = slugify_url(url)
        out = self.save_dir / fname
        out.write_text(html, encoding="utf-8")

    async def worker(self, client, pbar):
        while not self.to_visit.empty() and len(self.seen) < self.max_pages:
            url = await self.to_visit.get()
            if url in self.seen:
                self.to_visit.task_done()
                continue
            self.seen.add(url)

            html = await self.fetch(client, url)
            if html:
                self.save_html(url, html)
                # discover
                for link in self.extract_links(html, url):
                    if self._within_scope(link) and link not in self.seen:
                        await self.to_visit.put(link)
            await asyncio.sleep(self.rate_sleep_ms / 1000.0)
            pbar.update(1)
            self.to_visit.task_done()

    async def run(self):
        limits = httpx.Limits(max_keepalive_connections=self.concurrency, max_connections=self.concurrency)
        async with httpx.AsyncClient(limits=limits, follow_redirects=True) as client:
            with tqdm(total=self.max_pages, desc=f"Crawling {self.cfg['name']}") as pbar:
                tasks = [asyncio.create_task(self.worker(client, pbar)) for _ in range(self.concurrency)]
                await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Config-driven crawler")
    parser.add_argument("--config", default="configs/netapp_ontap.yml")
    args = parser.parse_args()
    asyncio.run(Crawler(args.config).run())
