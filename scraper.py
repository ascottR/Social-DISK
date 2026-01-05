import asyncio
import httpx
import pandas as pd
import random

# HEADERS (Using a standard browser header)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/"
}

async def fetch_posts(client, subreddit, query, limit=100):
    all_posts = []
    after_token = None
    
    print(f"--- Starting scrape for r/{subreddit} ---")
    
    while len(all_posts) < limit:
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {
            "q": query,
            "restrict_sr": "1",
            "limit": "25",
            "sort": "new"
        }
        
        if after_token:
            params["after"] = after_token

        try:
            response = await client.get(url, headers=HEADERS, params=params, timeout=10.0, follow_redirects=True)
            
            if response.status_code == 429:
                print(f"⚠️ Rate limited on r/{subreddit}. Sleeping...")
                await asyncio.sleep(5)
                continue
            
            if response.status_code != 200:
                break

            data = response.json()
            children = data.get("data", {}).get("children", [])
            
            if not children:
                break
            
            for child in children:
                post = child["data"]
                all_posts.append({
                    "source_subreddit": subreddit,
                    "title": post.get("title"),
                    "selftext": post.get("selftext"),
                    "upvotes": post.get("score"),
                    "comments_count": post.get("num_comments"),
                    "created_utc": post.get("created_utc"),
                    "url": post.get("url"),
                    "id": post.get("id")
                })
            
            after_token = data.get("data", {}).get("after")
            if not after_token:
                break
            
            # Sleep to avoid bans
            await asyncio.sleep(random.uniform(1, 2))
            
        except Exception as e:
            print(f"Crash: {e}")
            break
            
    return all_posts[:limit]

async def run_scraper(query, subreddits, max_posts_per_sub):
    """
    Main function to be called by the Streamlit App.
    """
    async with httpx.AsyncClient(http2=True) as client:
        tasks = [fetch_posts(client, sub, query, max_posts_per_sub) for sub in subreddits]
        results = await asyncio.gather(*tasks)
    
    # Flatten list
    flat_data = [post for sub_result in results for post in sub_result]
    df = pd.DataFrame(flat_data)
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['created_utc'], unit='s')
        
    return df