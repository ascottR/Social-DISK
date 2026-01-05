import asyncio
import httpx
import pandas as pd
import random

# --- CONFIGURATION ---

# List of User-Agents to rotate (Look like real browsers)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
]

# Base Headers
BASE_HEADERS = {
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/"
}

async def fetch_posts(client, subreddit, query, limit=100):
    all_posts = []
    after_token = None
    
    # Pick a random "identity" for this specific subreddit scrape
    headers = BASE_HEADERS.copy()
    headers["User-Agent"] = random.choice(USER_AGENTS)
    
    print(f"--- Starting scrape for r/{subreddit} ---")
    
    retries = 0
    max_retries = 3
    
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
            response = await client.get(url, headers=headers, params=params, timeout=10.0, follow_redirects=True)
            
            # --- RATE LIMIT HANDLING (Exponential Backoff) ---
            if response.status_code == 429:
                wait_time = (2 ** retries) + random.uniform(1, 3) # Wait 2s, 5s, 9s...
                print(f"⚠️ Rate limit (429) on r/{subreddit}. Cooling down for {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                retries += 1
                if retries > max_retries:
                    print(f"❌ Max retries reached for r/{subreddit}. Stopping.")
                    break
                continue # Try again
            
            # Stop if other errors occur (404, 403, 500)
            if response.status_code != 200:
                print(f"❌ Error {response.status_code} on r/{subreddit}")
                break

            # Reset retries on success
            retries = 0 

            data = response.json()
            children = data.get("data", {}).get("children", [])
            
            if not children:
                print(f"✅ Finished r/{subreddit} (No more posts).")
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
            
            # Pagination
            after_token = data.get("data", {}).get("after")
            if not after_token:
                break
            
            # Human-like delay between pages (prevents instant bans)
            await asyncio.sleep(random.uniform(1.5, 3.0))
            
        except Exception as e:
            print(f"Crash on r/{subreddit}: {e}")
            break
            
    return all_posts[:limit]

async def run_scraper(query, subreddits, max_posts_per_sub):
    """
    Main function to be called by the Streamlit App.
    """
    async with httpx.AsyncClient(http2=True) as client:
        # Create a task for each subreddit to run concurrently
        tasks = [fetch_posts(client, sub, query, max_posts_per_sub) for sub in subreddits]
        results = await asyncio.gather(*tasks)
    
    # Flatten list of lists
    flat_data = [post for sub_result in results for post in sub_result]
    df = pd.DataFrame(flat_data)
    
    if not df.empty:
        # Convert timestamp to readable date
        df['date'] = pd.to_datetime(df['created_utc'], unit='s')
        
    return df

# For testing this file directly (optional)
if __name__ == "__main__":
    print("Testing Scraper...")
    df = asyncio.run(run_scraper("protein", ["nutrition"], 10))
    print(df.head())