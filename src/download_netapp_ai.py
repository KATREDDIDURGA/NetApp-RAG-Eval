from requests_html import HTMLSession
from pathlib import Path

urls = [
    "https://docs.netapp.com/us-en/netapp-solutions/ai/wp-genai.html",
    "https://docs.netapp.com/us-en/netapp-solutions/ai/ai-minipod.html",
    "https://docs.netapp.com/us-en/netapp-solutions/ai/vector-database-solution-with-netapp.html",
    "https://docs.netapp.com/us-en/netapp-solutions/ai/aicp_introduction.html",
    "https://docs.netapp.com/us-en/netapp-solutions/ai/ai-edge-introduction.html"
]

save_dir = Path("data/raw/netapp-ai")
save_dir.mkdir(parents=True, exist_ok=True)

session = HTMLSession()

for url in urls:
    filename = url.split("/")[-1] or "index.html"
    file_path = save_dir / filename
    print(f"Downloading {url} -> {file_path}")
    try:
        r = session.get(url)
        r.html.render(timeout=30)  # JS rendering
        file_path.write_text(r.html.html, encoding="utf-8")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

print("✅ Done. Files saved to", save_dir)
