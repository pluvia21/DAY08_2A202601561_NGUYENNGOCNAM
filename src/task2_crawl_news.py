"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: thông báo tuyển sinh, sự kiện, dịch vụ thư viện, hỗ trợ sinh viên, học bổng.
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_latest_news_urls(limit=6):
    import requests
    from bs4 import BeautifulSoup
    
    print("Fetching latest news from UET homepage...")
    try:
        r = requests.get('https://uet.vnu.edu.vn/', timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        
        urls = []
        for a in soup.select('h3.entry-title a') + soup.select('.post-title a'):
            href = a.get('href')
            if href and href.startswith('https://uet.vnu.edu.vn/') and href not in urls:
                urls.append(href)
                if len(urls) >= limit:
                    break
                    
        return [{"url": u, "category": "news"} for u in urls]
    except Exception as e:
        print(f"[WARNING] Failed to fetch dynamic URLs: {e}")
        return []

ARTICLE_URLS = [
    {
        "url": "https://uet.vnu.edu.vn/trieu-tap-sinh-vien-tham-du-su-kien-uet-job-fair-2026/?utm_source=chatgpt.com",
        "category": "event"
    },
    {
        "url": "https://uet.vnu.edu.vn/ho-tro-cho-nguoi-hoc-nhan-dip-tet-nguyen-dan-2026/?utm_source=chatgpt.com",
        "category": "student_support"
    },
    {
        "url": "https://uet.vnu.edu.vn/chuong-trinh-hoc-bong-dinh-thien-ly-nam-hoc-2026-2027-danh-cho-sinh-vien-nam-cuoi-va-sinh-vien-nam-cuoi-co-hoan-canh-dac-biet/?utm_source=chatgpt.com",
        "category": "scholarship"
    },
    {
        "url": "https://uet.vnu.edu.vn/hoc-bong-vallet-nam-2026/?utm_source=chatgpt.com",
        "category": "scholarship"
    },
    {
        "url": "https://uet.vnu.edu.vn/gioi-thieu-chung/",
        "category": "event"
    },
    {
        "url": "https://uet.vnu.edu.vn/thu-ngo-cua-hieu-truong/",
        "category": "event"
    }
]

def safe_filename(title: str, date_iso: str) -> str:
    """Generate a safe, readable filename from title and date."""
    date_prefix = date_iso.split('T')[0]
    safe_title = re.sub(r'[^a-zA-Z0-9\s-]', '', title).strip().lower()
    safe_title = re.sub(r'[\s]+', '-', safe_title)
    if not safe_title:
        safe_title = "uet-news"
    safe_title = safe_title[:50]
    return f"{date_prefix}-{safe_title}.json"


async def crawl_article(item: dict, crawler=None, use_fallback=False) -> dict:
    url = item["url"]
    category = item["category"]

    try:
        title = ""
        content = ""
        
        if not use_fallback and crawler:
            result = await crawler.arun(url=url)
            
            if hasattr(result, 'metadata') and result.metadata and isinstance(result.metadata, dict):
                title = result.metadata.get('title', '')
            
            if not title and hasattr(result, 'markdown') and result.markdown:
                for line in result.markdown.split('\n'):
                    if line.strip().startswith('# '):
                        title = line.strip('# ').strip()
                        break
                        
            content = result.markdown if hasattr(result, 'markdown') else ""
            crawler_name = "crawl4ai"
        else:
            # Fallback using requests + markdownify
            import requests
            from bs4 import BeautifulSoup
            from markdownify import markdownify as md
            
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            if soup.title:
                title = soup.title.string.strip()
                
            # Extract main content roughly
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
            content = md(str(main_content), heading_style="ATX") if main_content else ""
            crawler_name = "requests_fallback"
        
        if not title:
            title = "Untitled UET News"

        crawled_at = datetime.now().isoformat()
        
        # Create unique ID from URL
        article_id = f"uet-news-{abs(hash(url)) % 1000000:06d}"
        
        # Remove url parameters for saving cleanly
        clean_url = url.split("?")[0]
        
        data = {
            "id": article_id,
            "title": title,
            "source_url": clean_url,
            "source_domain": "uet.vnu.edu.vn" if "uet.vnu.edu.vn" in url else "unknown",
            "category": category,
            "crawled_at": crawled_at,
            "content_format": "markdown",
            "content": content,
            "metadata": {
                "crawler": crawler_name,
                "status": "success"
            }
        }

        return data

    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return {
            "id": f"uet-news-{abs(hash(url)) % 1000000:06d}",
            "title": "Error",
            "source_url": url,
            "category": category,
            "status": "error",
            "error": str(e)
        }


async def crawl_all():
    from crawl4ai import AsyncWebCrawler
    
    setup_directory()
    
    results = []
    
    browser_config = None
    try:
        from crawl4ai.async_configs import BrowserConfig
        browser_config = BrowserConfig(channel="msedge")
    except ImportError:
        pass

    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    
    use_fallback = False
    crawler_context = None
    
    try:
        crawler_context = AsyncWebCrawler(config=browser_config, executable_path=edge_path)
        # Try to initialize playwright to see if it fails
        await crawler_context.start()
    except Exception as e:
        if "Playwright" in str(e) or "Executable doesn't exist" in str(e):
            print("[WARNING] Playwright browser missing or not installed properly. Using requests fallback.")
            use_fallback = True
            crawler_context = None
            
    try:
        for i, item in enumerate(ARTICLE_URLS, 1):
            url = item["url"]
            print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
            
            article_data = await crawl_article(item, crawler=crawler_context, use_fallback=use_fallback)
            
            if article_data.get("status") == "error":
                print(f"  [FAIL] Failed: {url} - {article_data.get('error')}")
                continue
                
            filename = safe_filename(article_data["title"], article_data["crawled_at"])
            filepath = DATA_DIR / filename
            
            counter = 1
            original_filepath = filepath
            while filepath.exists():
                filepath = original_filepath.with_name(original_filepath.stem + f"-{counter}.json")
                counter += 1
                
            filepath.write_text(json.dumps(article_data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [OK] Saved: {filepath.name}")
            
            article_data["_filepath"] = filepath.name
            results.append(article_data)
    except Exception as e:
        print(f"Global crawling error: {e}")

    # Save manifest
    manifest_data = {
        "total_articles": len(results),
        "crawled_at": datetime.now().isoformat(),
        "articles": []
    }
    
    for r in results:
        manifest_data["articles"].append({
            "id": r["id"],
            "title": r["title"],
            "source_url": r["source_url"],
            "category": r["category"],
            "file": r["_filepath"],
            "status": "success"
        })
        
    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest saved to {manifest_path.name}")
    print(f"Total crawled successfully: {len(results)}/{len(ARTICLE_URLS)}")
    
    if crawler_context and not use_fallback:
        await crawler_context.close()


if __name__ == "__main__":
    urls_to_crawl = ARTICLE_URLS + get_latest_news_urls(limit=4)
    # Deduplicate by url
    seen = set()
    unique_urls = []
    for item in urls_to_crawl:
        if item["url"] not in seen:
            seen.add(item["url"])
            unique_urls.append(item)
            
    ARTICLE_URLS = unique_urls
    
    if not ARTICLE_URLS:
        print("[WARNING] Không có bài viết nào để crawl!")
    else:
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass
        asyncio.run(crawl_all())
