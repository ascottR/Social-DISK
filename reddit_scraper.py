import asyncio
import httpx
import pandas as pd
import random
import time

# CONFIGURATION
# ---------------------------------------------------------
SEARCH_QUERY = "protein bar"  # What you are looking for
SUBREDDITS = ["nutrition", "supplements", "frugal", "fitness"] # Where to look
MAX_POSTS_PER_SUB = 100       # How much data you want
# ---------------------------------------------------------

# PREVENTING BLOCKS
# Reddit blocks default Python user agents. We pretend to be a Mac using Chrome.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/"
}

async def fetch_posts(client, subreddit, query, limit=100):
    all_posts = []
    after_token = None # This is the "cursor" for the next page
    
    print(f"--- Starting scrape for r/{subreddit} ---")
    
    while len(all_posts) < limit:
        # Build the URL: Search for the query INSIDE the subreddit
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {
            "q": query,
            "restrict_sr": "1", # Restrict search to this subreddit only
            "limit": "50",      # Max allowed per request is usually 25-100
            "sort": "new" # Can be 'new', 'top', 'relevance'
        }
        
        if after_token:
            params["after"] = after_token

        try:
            response = await client.get(url, headers=HEADERS, params=params, timeout=10.0)
            
            # CASE 1: Rate Limited (Too many requests)
            if response.status_code == 429:
                print(f"⚠️ Rate limited on r/{subreddit}. Sleeping for 10 seconds...")
                await asyncio.sleep(10)
                continue # Retry
            
            # CASE 2: Success
            if response.status_code == 200:
                data = response.json()
                children = data.get("data", {}).get("children", [])
                
                if not children:
                    print(f"No more results in r/{subreddit}.")
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
                
                # Update the 'after' token to get the next page
                after_token = data.get("data", {}).get("after")
                
                print(f"   Collected {len(all_posts)} posts so far...")
                
                if not after_token:
                    break # Stop if there are no more pages
                
                # IMPORTANT: Sleep slightly to avoid ban
                await asyncio.sleep(random.uniform(1, 3))
                
            else:
                print(f"Error {response.status_code}: {response.text}")
                break
                
        except Exception as e:
            print(f"Crash: {e}")
            break
            
    return all_posts[:limit]

async def main():
    async with httpx.AsyncClient(http2=True) as client:
        tasks = [fetch_posts(client, sub, SEARCH_QUERY, MAX_POSTS_PER_SUB) for sub in SUBREDDITS]
        results = await asyncio.gather(*tasks)
    
    # Flatten the list of lists
    flat_data = [post for sub_result in results for post in sub_result]
    
    # Save to DataFrame
    df = pd.DataFrame(flat_data)
    
    # Basic Cleaning: Convert timestamp to readable date
    if not df.empty:
        df['date'] = pd.to_datetime(df['created_utc'], unit='s')
        print(f"\n✅ Scraping Complete. Total Posts: {len(df)}")
        print(df[['source_subreddit', 'title', 'upvotes']].head())
        
        # Save to CSV
        filename = "reddit_protein_bar_data.csv"
        df.to_csv(filename, index=False)
        print(f"Saved to {filename}")
    else:
        print("No data found.")

if __name__ == "__main__":
    asyncio.run(main())