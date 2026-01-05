import streamlit as st
import pandas as pd
import json
import asyncio
from groq import AsyncGroq
from neo4j import GraphDatabase
import scraper  # Import the local scraper module

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SocialDisk: Local", page_icon="🧬", layout="wide")

# CSS Styling (Dark Mode & Neon)
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .neon-header { border-bottom: 2px solid #FF00FF; padding-bottom: 10px; margin-bottom: 20px; color: #FF00FF; font-size: 24px; font-weight: bold; }
    .novel-card { background-color: #2b1111; color: #ffb3b3; padding: 15px; border-radius: 8px; border-left: 5px solid #ff4b4b; margin-bottom: 10px; }
    .known-card { background-color: #0f291e; color: #b3ffcc; padding: 15px; border-radius: 8px; border-left: 5px solid #00cc96; margin-bottom: 10px; }
    .unknown-card { background-color: #2b2111; color: #ffeeb3; padding: 15px; border-radius: 8px; border-left: 5px solid #ff9800; margin-bottom: 10px; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #00FFCC; }
</style>
""", unsafe_allow_html=True)

# --- CREDENTIALS & CONFIG ---
# Load secrets safely from .streamlit/secrets.toml
try:
    NEO4J_URI = st.secrets["NEO4J_URI"]
    NEO4J_USERNAME = st.secrets["NEO4J_USERNAME"]
    NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
    NEO4J_AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception as e:
    st.error(f"⚠️ Missing Credentials! Please set up `.streamlit/secrets.toml`. Error: {e}")
    st.stop()

# --- NEO4J LOGIC ---
@st.cache_resource
def get_neo4j_driver():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        driver.verify_connectivity()
        return driver
    except Exception as e:
        st.error(f"❌ Neo4j Connection Failed: {e}")
        return None

def verify_claim_logic(tx, supplement, symptom):
    query = """
    MATCH (s:Ingredient)-[r:ASSOCIATED_WITH]->(e:Symptom)
    WHERE toLower(s.name) CONTAINS toLower($supp)
      AND toLower(e.name) CONTAINS toLower($symp)
      AND r.type = 'has_adverse_reaction'
    RETURN type(r) as existing_relation
    """
    relation = tx.run(query, supp=supplement, symp=symptom).single()
    return "KNOWN" if relation else "NOVEL"

def run_verification_process(input_data):
    driver = get_neo4j_driver()
    if not driver: return pd.DataFrame()
    results = []
    with driver.session() as session:
        for post in input_data:
            if "Relations" in post:
                for rel in post["Relations"]:
                    supp = rel.get("Subject", "Unknown")
                    symp = rel.get("Object", "Unknown")
                    status = session.execute_read(verify_claim_logic, supp, symp)
                    results.append({"Ingredient": supp, "Symptom": symp, "Status": status, "Source Text": post.get("text", "N/A")})
    return pd.DataFrame(results)

# --- GROQ AI LOGIC ---
class RedditIntelligenceAgent:
    def __init__(self, api_key):
        self.client = AsyncGroq(api_key=api_key)

    async def analyze_data(self, text, mode="trend"):
        if mode == "trend":
            system_prompt = """You are a Market Researcher. Output JSON: { "sentiment_score": (1-10), "taste_complaints": [], "price_complaints": [], "feature_requests": [] }"""
        else:
            system_prompt = """You are a Safety Officer. Output JSON: { "risk_level": "Low/Medium/High", "identified_symptoms": [], "severe_reports": [], "safety_summary": "string" }"""
        
        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Data:\n{text[:15000]}"}],
                model="llama-3.3-70b-versatile", response_format={"type": "json_object"}
            )
            return json.loads(chat_completion.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}

# --- MAIN APP ---
def main():
    st.sidebar.title("🧬 SocialDisk: Local")
    st.sidebar.markdown("---")
    
    # NAVIGATION
    app_mode = st.sidebar.radio("Select Module:", ["🕷️ Live Scraper (Local)", "📊 Analyze Data (AI)", "🛡️ Signal Verifier (Neo4j)"])

    # 1. LIVE SCRAPER (New Feature)
    if app_mode == "🕷️ Live Scraper (Local)":
        st.markdown("<div class='neon-header'>🕷️ LIVE REDDIT SCRAPER</div>", unsafe_allow_html=True)
        st.info("Since this is running locally, Reddit won't block you! (Unless you spam)")

        c1, c2 = st.columns(2)
        query = c1.text_input("Search Query", "protein bar")
        subs = c2.text_input("Subreddits (comma separated)", "nutrition,supplements,frugal")
        limit = st.slider("Max Posts Per Sub", 10, 100, 50)

        if st.button("🚀 Start Scraping", type="primary"):
            sub_list = [s.strip() for s in subs.split(",")]
            
            with st.spinner(f"Scraping Reddit for '{query}'..."):
                # Run the async scraper
                df = asyncio.run(scraper.run_scraper(query, sub_list, limit))
            
            if not df.empty:
                st.success(f"✅ Collected {len(df)} posts!")
                st.dataframe(df[['source_subreddit', 'title', 'upvotes']].head())
                
                # Save automatically
                df.to_csv("reddit_data.csv", index=False)
                st.toast("Saved to 'reddit_data.csv'")
            else:
                st.error("No data found.")

    # 2. ANALYZE DATA
    elif app_mode == "📊 Analyze Data (AI)":
        st.markdown("<div class='neon-header'>📊 AI ANALYZER</div>", unsafe_allow_html=True)
        
        # Option to load the just-scraped file or upload new
        data_source = st.radio("Data Source:", ["📂 Load 'reddit_data.csv' (Last Scrape)", "⬆️ Upload New CSV/JSON"])
        
        df = None
        if "Last Scrape" in data_source:
            try:
                df = pd.read_csv("reddit_data.csv")
                st.success("Loaded local 'reddit_data.csv'")
            except:
                st.warning("No local file found. Go scrape something first!")
        else:
            uploaded_file = st.file_uploader("Upload File", type=["csv", "json"])
            if uploaded_file:
                if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
                else: df = pd.DataFrame(json.load(uploaded_file))

        if df is not None:
            analysis_type = st.radio("Mode", ["Market Trends", "Safety Monitor"], horizontal=True)
            if st.button("🤖 Analyze with Groq"):
                # Prepare text
                raw_text = ""
                for _, row in df.head(30).iterrows(): # Limit to 30 for speed
                    raw_text += f"TITLE: {row.get('title', '')}\nBODY: {row.get('selftext', '')}\n---\n"
                
                agent = RedditIntelligenceAgent(GROQ_API_KEY)
                mode_key = "safety" if "Safety" in analysis_type else "trend"
                
                with st.spinner("Analyzing..."):
                    report = asyncio.run(agent.analyze_data(raw_text, mode_key))
                
                st.json(report)

    # 3. VERIFIER
    elif app_mode == "🛡️ Signal Verifier (Neo4j)":
        st.markdown("<div class='neon-header'>🛡️ SIGNAL VERIFIER</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Relations JSON", type=["json"])
        if uploaded_file and st.button("Verify"):
            data = json.load(uploaded_file)
            df = run_verification_process(data)
            st.dataframe(df)

if __name__ == "__main__":
    main()