import streamlit as st
import pandas as pd
import json
import asyncio
from datetime import datetime, timedelta
import plotly.express as px

from groq import AsyncGroq
from neo4j import GraphDatabase

import scraper 

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Social DSS", page_icon="🧬", layout="wide")

# --- CSS STYLING (Dark Mode & Neon) ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    /* Neon Headers */
    .neon-header { border-bottom: 2px solid #FF00FF; padding-bottom: 10px; margin-bottom: 20px; color: #FF00FF; font-size: 24px; font-weight: bold; }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] { font-size: 28px; color: #00FFCC; }
    
    /* Status Labels */
    .status-novel { background-color: #451a1a; color: #ff9999; padding: 4px 8px; border-radius: 4px; border: 1px solid #ff4b4b; font-weight: bold; }
    .status-known { background-color: #0f291e; color: #99ffcc; padding: 4px 8px; border-radius: 4px; border: 1px solid #00cc96; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- CREDENTIALS ---
# Load from .streamlit/secrets.toml
try:
    NEO4J_URI = st.secrets["NEO4J_URI"]
    NEO4J_USERNAME = st.secrets["NEO4J_USERNAME"]
    NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
    NEO4J_AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception as e:
    st.error(f"⚠️ Missing Credentials! Please set up `.streamlit/secrets.toml`. Error: {e}")
    st.stop()

# --- BACKEND LOGIC: NEO4J ---
@st.cache_resource
def get_neo4j_driver():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        driver.verify_connectivity()
        return driver
    except Exception as e:
        st.error(f"❌ Neo4j Connection Failed: {e}")
        return None

def verify_claim_logic(tx, subject, object_entity):
    """
    Checks if a relationship exists in the Knowledge Graph.
    Returns: 'KNOWN' (exists) or 'NOVEL' (new signal).
    """
    query = """
    MATCH (s)-[r]->(o)
    WHERE toLower(s.name) CONTAINS toLower($sub)
      AND toLower(o.name) CONTAINS toLower($obj)
    RETURN type(r) as relation_type
    """
    result = tx.run(query, sub=subject, obj=object_entity).single()
    return "KNOWN" if result else "NOVEL"

def batch_verify_signals(relations_list):
    driver = get_neo4j_driver()
    if not driver: return []
    
    verified_results = []
    with driver.session() as session:
        for rel in relations_list:
            subj = rel.get("Subject")
            obj = rel.get("Object")
            pred = rel.get("Predicate", "RELATED_TO")
            
            if subj and obj:
                status = session.execute_read(verify_claim_logic, subj, obj)
                verified_results.append({
                    "Subject": subj,
                    "Predicate": pred,
                    "Object": obj,
                    "Type": rel.get("Type", "General"),
                    "Status": status
                })
    return verified_results

# --- BACKEND LOGIC: GROQ AI ---
class RedditIntelligenceAgent:
    def __init__(self, api_key):
        self.client = AsyncGroq(api_key=api_key)

    async def extract_safety_signals(self, text):
        """
        Extracts structured Entities & Relations (JSON) from raw text.
        """
        system_prompt = """
        You are an expert Pharmacovigilance AI. 
        Your goal is to extract safety signals from consumer text.
        
        OUTPUT JSON FORMAT:
        {
          "Entities": [
            { "text": "entity_name", "type": "BRAND/SYMPTOM/INGREDIENT" }
          ],
          "Relations": [
            { 
              "Subject": "Entity A", 
              "Predicate": "CAUSES/CONTAINS/DOES_NOT_CONTAIN", 
              "Object": "Entity B", 
              "Type": "Adverse Event/Composition/Effect", 
              "Sentiment": "Negative/Neutral" 
            }
          ],
          "User_Intent": "Summary of what the user wants",
          "Recommendation_Request": boolean
        }

        RULES:
        1. Map slang to standard medical terms (e.g., "the runs" -> "Diarrhea").
        2. Identify Brands (Subject) vs Symptoms (Object).
        3. Only output valid JSON.
        """
        
        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this text:\n{text[:4000]}"}
                ],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(chat_completion.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}

# --- MAIN DASHBOARD ---
def main():
    st.sidebar.title("🧬 SocialDisk DSS")
    st.sidebar.markdown("---")
    app_mode = st.sidebar.radio("Select Module:", ["🕷️ Market Pulse (Scraper)", "🛡️ Safety Intelligence Hub"])

    # ==========================================
    # MODULE 1: MARKET PULSE (Business Intel)
    # ==========================================
    if app_mode == "🕷️ Market Pulse (Scraper)":
        st.markdown("<div class='neon-header'>🕷️ MARKET PULSE</div>", unsafe_allow_html=True)
        st.markdown("Monitor brand health and detect emerging trends in real-time.")

        # 1. CONTROL PANEL
        with st.container():
            c1, c2, c3 = st.columns([1, 2, 1])
            
            with c1:
                search_mode = st.radio("Search Scope", ["🛡️ Brand Monitor", "🔭 Category Explorer"])
            
            with c2:
                if search_mode == "🛡️ Brand Monitor":
                    search_input = st.text_input("Target Brand", placeholder="e.g., Optimum Nutrition, AG1")
                else:
                    search_input = st.text_input("Product Category", placeholder="e.g., Pre-workout, Creatine Gummies")
            
            with c3:
                time_filter = st.selectbox("Time Horizon", ["Last 30 Days", "Last 60 Days", "Last 90 Days", "All Time"])

        with st.expander("⚙️ Advanced Scraper Settings"):
            subs = st.text_input("Subreddits to Scan", "supplements,nutrition,frugal,gymsnark,biohackers")
            limit = st.slider("Max Posts per Sub", 50, 300, 100)

        # 2. EXECUTION
        if st.button("🚀 Run Market Analysis", type="primary"):
            if not search_input:
                st.error("Please enter a Brand or Category.")
            else:
                sub_list = [s.strip() for s in subs.split(",")]
                
                with st.spinner(f"📡 Scanning Reddit frequencies for '{search_input}'..."):
                    try:
                        # Call local scraper module
                        df = asyncio.run(scraper.run_scraper(search_input, sub_list, limit))
                        
                        if not df.empty:
                            # Date Filtering
                            df['date'] = pd.to_datetime(df['created_utc'], unit='s')
                            
                            if time_filter != "All Time":
                                days_map = {"Last 30 Days": 30, "Last 60 Days": 60, "Last 90 Days": 90}
                                cutoff = datetime.now() - timedelta(days=days_map.get(time_filter, 30))
                                df = df[df['date'] > cutoff]

                            if df.empty:
                                st.warning(f"No data found in the {time_filter}.")
                            else:
                                # 3. DASHBOARD METRICS
                                st.success(f"✅ Analyzed {len(df)} discussions from {time_filter}.")
                                
                                kpi1, kpi2, kpi3 = st.columns(3)
                                kpi1.metric("Volume (Mentions)", len(df))
                                kpi2.metric("Engagement (Votes+Comments)", int(df['upvotes'].sum() + df['comments_count'].sum()))
                                
                                # Simple keyword detection for "Risk" context
                                negative_keywords = ['sick', 'nausea', 'vomit', 'headache', 'rash', 'bloating']
                                risk_posts = df[df['selftext'].str.contains('|'.join(negative_keywords), case=False, na=False)]
                                kpi3.metric("Potential Adverse Events", len(risk_posts), delta_color="inverse")

                                # 4. TREND CHART
                                st.subheader("📈 Discussion Velocity")
                                daily_counts = df.groupby(df['date'].dt.date).size().reset_index(name='Posts')
                                fig = px.line(daily_counts, x='date', y='Posts', title=f"Mentions of '{search_input}'", template="plotly_dark")
                                fig.update_traces(line_color='#00FFCC', line_width=3)
                                st.plotly_chart(fig, use_container_width=True)

                                # 5. DRILL DOWN (Top Posts)
                                st.subheader("🔥 Top Discussions")
                                top_posts = df.sort_values(by='upvotes', ascending=False).head(5)
                                
                                for _, row in top_posts.iterrows():
                                    with st.expander(f"[{row['upvotes']}▲] {row['title']}"):
                                        st.write(row['selftext'])
                                        st.caption(f"r/{row['source_subreddit']} • {row['date'].strftime('%Y-%m-%d')}")
                                        
                                        # "Send to Safety Hub" Logic (Simulated via Session State)
                                        if st.button("Analyze for Safety Signals", key=row['id']):
                                            st.session_state['analysis_text'] = row['selftext']
                                            st.info("Text copied! Switch to 'Safety Intelligence Hub' tab to process.")

                                # Save Data
                                df.to_csv("reddit_data.csv", index=False)
                                st.toast("Dataset saved to reddit_data.csv")

                        else:
                            st.warning("Scraper returned no results.")
                            
                    except Exception as e:
                        st.error(f"System Error: {e}")

    # ==========================================
    # MODULE 2: SAFETY INTELLIGENCE HUB (AI + Graph)
    # ==========================================
    elif app_mode == "🛡️ Safety Intelligence Hub":
        st.markdown("<div class='neon-header'>🛡️ SAFETY INTELLIGENCE HUB</div>", unsafe_allow_html=True)
        st.markdown("**Workflow:** Extract Signals (AI) $\\rightarrow$ Filter (Adverse Events) $\\rightarrow$ Verify Novelty (Neo4j)")

        # 1. INPUT
        with st.expander("📂 Data Input", expanded=True):
            input_method = st.radio("Source:", ["Paste Text", "Upload File (CSV/JSON)", "Load Last Scrape"], horizontal=True)
            
            raw_texts = []
            
            if input_method == "Paste Text":
                default_text = st.session_state.get('analysis_text', "")
                txt = st.text_area("Post Content:", value=default_text, height=150)
                if txt: raw_texts = [txt]
            
            elif input_method == "Load Last Scrape":
                try:
                    df = pd.read_csv("reddit_data.csv")
                    # Join title + body for better context
                    raw_texts = (df['title'].fillna('') + " " + df['selftext'].fillna('')).tolist()
                    st.info(f"Loaded {len(raw_texts)} posts from last scrape.")
                except:
                    st.warning("No previous scrape found.")

            else:
                uploaded_file = st.file_uploader("Upload File", type=["json", "csv"])
                if uploaded_file:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        raw_texts = (df['title'].fillna('') + " " + df['selftext'].fillna('')).tolist()
                    else:
                        data = json.load(uploaded_file)
                        if isinstance(data, list):
                            raw_texts = [d.get('selftext', str(d)) for d in data]

            # --- NEW: LIMIT CONTROLLER ---
            st.markdown("---")
            # Default is 5 for testing, can increase to 100+ for full runs
            analysis_limit = st.number_input("⚠️ Max Posts to Analyze (Save API Credits)", min_value=1, max_value=5000, value=5)

        # 2. PROCESSING
        if raw_texts and st.button(f"🔍 Verify Signals (First {analysis_limit})", type="primary"):
            
            # --- APPLY LIMIT HERE ---
            texts_to_process = raw_texts[:analysis_limit]
            
            agent = RedditIntelligenceAgent(GROQ_API_KEY)
            all_extractions = []
            all_verifications = []
            
            progress = st.progress(0)
            status_t = st.empty()
            
            total = len(texts_to_process)
            
            for i, text in enumerate(texts_to_process):
                if len(text) < 20: continue # Skip short junk
                
                status_t.text(f"AI Analyzing Post {i+1}/{total}...")
                
                # A. Extract
                ai_result = asyncio.run(agent.extract_safety_signals(text))
                
                if "error" not in ai_result:
                    ai_result["Source_Summary"] = text[:50] + "..."
                    all_extractions.append(ai_result)
                    
                    # B. Filter & Verify
                    if "Relations" in ai_result and ai_result["Relations"]:
                        
                        # STRICT FILTER: Only 'Adverse Event' with valid Subject & Object
                        adverse_events = [
                            rel for rel in ai_result["Relations"] 
                            if rel.get("Type") == "Adverse Event" 
                            and rel.get("Subject") 
                            and rel.get("Object")
                        ]
                        
                        if adverse_events:
                            verified = batch_verify_signals(adverse_events)
                            # Add context back
                            for v in verified: v["Source"] = text[:50] + "..."
                            all_verifications.extend(verified)
                
                progress.progress((i + 1) / total)
            
            status_t.text("✅ Analysis Complete.")
            
            # 3. RESULTS DISPLAY
            tab1, tab2 = st.tabs(["🚦 Verification Matrix", "🧠 AI Extraction Log"])
            
            with tab1:
                st.subheader("Signal Novelty Check (Adverse Events Only)")
                if all_verifications:
                    df_v = pd.DataFrame(all_verifications)
                    
                    def color_status(val):
                        if val == "NOVEL": return 'background-color: #451a1a; color: #ff9999; font-weight: bold;'
                        return 'background-color: #0f291e; color: #99ffcc; font-weight: bold;'

                    st.dataframe(
                        df_v.style.map(color_status, subset=['Status']),
                        column_order=["Subject", "Predicate", "Object", "Status", "Source"],
                        use_container_width=True
                    )
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Total Adverse Events", len(df_v))
                    c2.metric("⚠️ Novel Risks", len(df_v[df_v['Status']=='NOVEL']))
                else:
                    st.info(f"No Adverse Events found in the first {analysis_limit} posts.")

            with tab2:
                st.json(all_extractions)
if __name__ == "__main__":
    main()