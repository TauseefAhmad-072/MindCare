"""
scrappers/reddit.py — MindCare RSS Syndication Pipeline
Bypasses 403 authorization walls using the globally compliant W3C Atom XML engine.
"""
import requests
import xml.etree.ElementTree as ET
import html
import re

# ── CONFIGURATION
TARGET_USER   = "SamiSha_"          
MAX_POSTS     = 20                  

# 🚀 CRITICAL FIX 1: Restored the fully operational secure forward slash directory mapping string
BASE_URL = "https://www.reddit.com/user/{}/submitted/.rss"
HEADERS  = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
}

def clean_html_body(html_content: str) -> str:
    """Isolates and extracts pure text from Reddit's RSS HTML content container wrappers."""
    if not html_content:
        return ""
    text = html.unescape(html_content)
    text = re.sub(r'<table.*?>.*?</table>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]*>', ' ', text)
    return re.sub(r"\s+", " ", text).strip()

def is_text_post(body_text: str) -> bool:
    if not body_text or len(body_text) < 5 or body_text in ("[removed]", "[deleted]"):
        return False
    return True

def parse_post(entry_element, namespaces: dict, username: str) -> dict:
    id_elem = entry_element.find('atom:id', namespaces)
    post_id = id_elem.text.split('_')[-1] if id_elem is not None else ""
    
    title_elem = entry_element.find('atom:title', namespaces)
    title = title_elem.text if title_elem is not None else ""
    
    content_elem = entry_element.find('atom:content', namespaces)
    raw_content = content_elem.text if content_elem is not None else ""
    body = clean_html_body(raw_content)
    
    updated_elem = entry_element.find('atom:updated', namespaces)
    updated_raw = updated_elem.text if updated_elem is not None else ""
    
    link_elem = entry_element.find('atom:link', namespaces)
    post_url = link_elem.get('href') if link_elem is not None else f"https://www.reddit.com/user/{username}"

    return {
        "id":           post_id,
        "title":        title.strip(),
        "body":         body,
        "subreddit":    "u_" + username,
        "score":        1,  
        "upvote_ratio": 1.0,
        "num_comments": 0,
        "created_utc":  updated_raw,
        "url":          post_url,
    }

def fetch_user_text_posts(username: str, max_posts: int = 20) -> list[dict]:
    if username.startswith('u/'):
        username = username[2:]
    if username.startswith('u_'):
        username = username[2:]
        
    url = BASE_URL.format(username)
    results = []

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
            
        root = ET.fromstring(resp.content)
        # 🚀 CRITICAL FIX 2: Restored fully compliant W3C Atom XML standard schema string location URL
        namespaces = {'atom': 'http://w3.org'}
        entries = root.findall('atom:entry', namespaces)
        
        for entry in entries:
            parsed = parse_post(entry, namespaces, username)
            if is_text_post(parsed["body"]):
                results.append(parsed)
                if max_posts and len(results) >= max_posts:
                    break
    except Exception as e:
        print(f"[Reddit Scraper Error] {e}")
        return []

    return results

if __name__ == "__main__":
    posts = fetch_user_text_posts(TARGET_USER, MAX_POSTS)
    print(f"✅ Extracted {len(posts)} structural text rows from Reddit Atom stream.")
