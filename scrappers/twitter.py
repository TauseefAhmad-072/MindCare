# scrappers/twitter.py
"""
X/Twitter scraper using twscrape — account pool with cookie auth.
Requires: pip install twscrape
Setup:    twscrape add_cookie my_account "auth_token=xxx; ct0=yyy"
"""
import asyncio
from twscrape import API, gather

# Use the default accounts.db in the project root
_api = API()


async def _fetch_tweets_async(username: str, max_tweets: int) -> list[dict]:
    """Async fetch — twscrape is async-only."""
    # Search "from:username" is the most reliable endpoint for user timelines
    query = f"from:{username}"
    tweets = await gather(_api.search(query, limit=max_tweets))

    results = []
    for t in tweets:
        results.append({
            "id": str(t.id),
            "body": t.rawContent,
            "created_utc": t.date.isoformat() if t.date else "",
            "url": t.url or f"https://x.com/{username}/status/{t.id}",
        })
    return results


def fetch_tweets(username: str, max_tweets: int = 20) -> list[dict]:
    """
    Synchronous wrapper — matches the exact signature your app.py expects.
    Returns [{id, body, created_utc, url}, ...]
    """
    if username.startswith('@'):
        username = username[1:]

    print(f"\n🔍 Scraping tweets from X for @{username} using twscrape ...")

    try:
        return asyncio.run(_fetch_tweets_async(username, max_tweets))
    except Exception as e:
        print(f"❌ twscrape Error: {e}")
        return []


if __name__ == "__main__":
    tweets = fetch_tweets("realDonaldTrump", max_tweets=5)
    print(f"\n📊 Retrieved {len(tweets)} tweets.")
    for t in tweets:
        print(f"--- {t['created_utc']} ---\n{t['body'][:100]}...\n")