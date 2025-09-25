#!/usr/bin/env python3
"""
Complete YouTube Analytics Platform with Pagination
Combines intelligent niche research, viral video finder, and channel finder
Advanced analytics with mathematical models for YouTube success
"""

import streamlit as st
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from scipy import stats
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
try:
    from textstat import flesch_reading_ease
except ImportError:
    def flesch_reading_ease(text):
        return 50  # Default score if textstat not available
import re
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="YouTube Complete Analytics Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
/* --- Refined & Professional YouTube Analytics Platform CSS --- */

/* --- Keyframe Animations --- */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes shimmer {
    0% { background-position: -200px 0; }
    100% { background-position: calc(200px + 100%) 0; }
}

/* --- Root Variables for a Clean & Consistent Theme --- */
:root {
    --primary-color: #6366F1; /* A modern, friendly indigo */
    --secondary-color: #8B5CF6; /* A complementary violet */
    --accent-color: #10B981; /* A vibrant but clean emerald green */
    --text-color: #374151; /* A dark gray for readability */
    --text-color-light: #6B7280; /* A lighter gray for subtext */
    --bg-color: #F9FAFB; /* A very light gray background */
    --card-bg-color: #FFFFFF;
    --border-color: #E5E7EB;
    --border-radius: 12px;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

body {
    background-color: var(--bg-color);
}

/* --- Main Header --- */
.main-header {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    padding: 2.5rem 2rem;
    border-radius: var(--border-radius);
    margin-bottom: 2rem;
    text-align: center;
    color: white;
    box-shadow: var(--shadow-lg);
    animation: fadeIn 0.8s ease-out;
}
.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
}
.main-header h3 {
    font-size: 1.25rem;
    font-weight: 400;
    opacity: 0.9;
}

/* --- General Content Cards (Metric, Result, etc.) --- */
.metric-card, .result-card, .question-box, .prediction-box, .optional-section {
    background-color: var(--card-bg-color);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-md);
    transition: var(--transition);
    animation: fadeIn 0.6s ease-out forwards;
    opacity: 0;
    animation-delay: 0.2s;
}
.metric-card:hover, .result-card:hover, .question-box:hover, .prediction-box:hover, .optional-section:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
}

/* Specific Card Styling */
.question-box {
    border-left: 4px solid var(--primary-color);
}
.question-box h4 {
    color: var(--primary-color);
    margin-bottom: 0.5rem;
}

.prediction-box {
    background-color: #F0FFF4;
    border-color: var(--accent-color);
    border-left: 4px solid var(--accent-color);
}
.prediction-box h4 {
    color: var(--accent-color);
}

.optional-section {
    background: #FFFBEB;
    border-color: #FBBF24;
}
.optional-section h4 {
    color: #B45309;
}

/* --- Buttons --- */
.stButton > button {
    background-color: var(--primary-color);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: 600;
    transition: var(--transition);
    box-shadow: var(--shadow-md);
}
.stButton > button:hover {
    background-color: var(--secondary-color);
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}
.stButton > button:active {
    transform: translateY(0);
}

/* --- Tabs --- */
.stTabs [data-baseweb="tab-list"] {
    background-color: rgba(255, 255, 255, 0.7);
    padding: 6px;
    border-radius: 10px;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-color);
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 20px;
    border-radius: 8px;
    transition: var(--transition);
    font-weight: 500;
    color: var(--text-color-light);
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #F3F4F6;
    color: var(--text-color);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    box-shadow: var(--shadow-md);
}

/* --- Input Fields --- */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    border-radius: 8px;
    border: 1px solid var(--border-color);
    transition: var(--transition);
    background-color: var(--card-bg-color);
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
}

/* --- Sidebar --- */
.css-1d391kg {
    background-color: var(--card-bg-color);
    border-right: 1px solid var(--border-color);
}

/* --- Expander / Accordion --- */
.streamlit-expander {
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius);
    overflow: hidden;
    transition: var(--transition);
    box-shadow: var(--shadow-sm);
    animation: fadeIn 0.5s ease-out;
}
.streamlit-expander:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--primary-color);
}
.streamlit-expander > summary {
    padding: 1rem 1.25rem;
    font-weight: 500;
    background-color: #F9FAFB;
}
.streamlit-expander > summary:hover {
    background-color: #F3F4F6;
}

/* --- Dataframe & Tables --- */
.stDataFrame {
    border-radius: var(--border-radius);
    overflow: hidden;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-color);
}

/* --- Metric Display --- */
.stMetric {
    background-color: var(--card-bg-color);
    padding: 1.5rem;
    border-radius: var(--border-radius);
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-md);
    transition: var(--transition);
}
.stMetric:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-lg);
}

/* --- Progress Bar --- */
.stProgress > div > div > div > div {
    background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
    border-radius: 10px;
}

/* --- Alerts and Messages --- */
.stAlert {
    border-radius: var(--border-radius);
    animation: fadeIn 0.5s ease-out;
    box-shadow: var(--shadow-md);
    border-left-width: 4px;
    border-left-style: solid;
}
.st-emotion-cache-1wivapv.e1nzilvr5 { /* Success Alert */
    border-left-color: var(--accent-color);
}
.st-emotion-cache-1wivapv.e1nzilvr4 { /* Info Alert */
    border-left-color: var(--primary-color);
}
.st-emotion-cache-1wivapv.e1nzilvr3 { /* Warning Alert */
    border-left-color: #FBBF24;
}
.st-emotion-cache-1wivapv.e1nzilvr2 { /* Error Alert */
    border-left-color: #F87171;
}

/* --- Responsive Design --- */
@media (max-width: 992px) {
    .main-header {
        padding: 2rem 1.5rem;
    }
    .main-header h1 {
        font-size: 2rem;
    }
}

@media (max-width: 768px) {
    .main-header {
        padding: 2rem 1rem;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 1.75rem;
    }
    .main-header h3 {
        font-size: 1.1rem;
    }
    .metric-card, .result-card, .question-box, .prediction-box, .optional-section {
        padding: 1rem;
    }
    .stButton > button {
        width: 100%;
        padding: 14px 20px;
    }
    .stTabs [data-baseweb="tab"] {
        flex-grow: 1;
        text-align: center;
    }
}

@media (max-width: 480px) {
    .main-header h1 {
        font-size: 1.5rem;
    }
    .main-header h3 {
        font-size: 1rem;
    }
    :root {
        --border-radius: 10px;
    }
}

/* --- Dark Mode Support --- */
@media (prefers-color-scheme: dark) {
    :root {
        --text-color: #E2E8F0;
        --text-color-light: #94A3B8;
        --bg-color: #1A202C;
        --card-bg-color: #2D3748;
        --border-color: #4A5568;
    }
    .st-emotion-cache-1wivapv.e1nzilvr5, /* Success Alert */
    .prediction-box {
        background-color: rgba(16, 185, 129, 0.1);
        border-color: var(--accent-color);
    }
    .optional-section {
        background-color: rgba(251, 191, 36, 0.1);
        border-color: #FBBF24;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(0, 0, 0, 0.2);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #4A5568;
    }
    .streamlit-expander > summary {
        background-color: #1A202C;
    }
    .streamlit-expander > summary:hover {
        background-color: #2D3748;
    }
}

/* --- Accessibility Improvements --- */
*:focus-visible {
    outline: 3px solid var(--primary-color);
    outline-offset: 2px;
    border-radius: 4px;
}

@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* --- Print Styles --- */
@media print {
    .main-header, .stButton, .stTabs, .sidebar, .st-emotion-cache-16txtl3 {
        display: none !important;
    }
    .metric-card, .result-card, .question-box, .prediction-box, .optional-section {
        box-shadow: none !important;
        border: 1px solid #ccc !important;
        break-inside: avoid;
    }
    body {
        background-color: #FFFFFF !important;
    }
}
</style>
""", unsafe_allow_html=True)

# --- Mathematical Constants & Configuration ---
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
EULER_NUMBER = np.e
PI = np.pi

# API Configuration
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEO_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_CHANNEL_URL = "https://www.googleapis.com/youtube/v3/channels"

# --- Advanced Mathematical Models ---
class GrowthAnalyzer:
    @staticmethod
    def exponential_growth_model(t, a, b, c):
        return a * np.exp(b * t) + c

    @staticmethod
    def logistic_growth_model(t, L, k, t0):
        return L / (1 + np.exp(-k * (t - t0)))

    @staticmethod
    def power_law_model(x, a, b):
        return a * np.power(x, b)

    @staticmethod
    def calculate_growth_velocity(data_points, time_intervals):
        if len(data_points) < 2:
            return 0
        velocities = np.gradient(data_points, time_intervals)
        return np.mean(velocities)

    @staticmethod
    def calculate_growth_acceleration(data_points, time_intervals):
        if len(data_points) < 3:
            return 0
        velocities = np.gradient(data_points, time_intervals)
        accelerations = np.gradient(velocities, time_intervals)
        return np.mean(accelerations)

class ViralityPredictor:
    @staticmethod
    def calculate_viral_coefficient(views, time_since_publish, subscriber_count):
        if subscriber_count == 0:
            return 0
        decay_constant = 0.1
        time_factor = np.exp(-decay_constant * time_since_publish.days)
        viral_ratio = views / max(subscriber_count, 1)
        return viral_ratio * time_factor

    @staticmethod
    def engagement_quality_score(likes, comments, views, video_duration):
        if views == 0:
            return 0
        duration_factor = max(1, video_duration / 300)
        like_rate = (likes / views) * 100
        comment_rate = (comments / views) * 100
        engagement_score = (like_rate + 3 * comment_rate) / duration_factor
        return 10 / (1 + np.exp(-engagement_score + 2))

class NetworkAnalyzer:
    @staticmethod
    def build_topic_network(channels_data):
        G = nx.Graph()
        for channel in channels_data:
            G.add_node(
                channel.get('Channel Name', ''),
                subscribers=channel.get('Subscribers', 0),
                niche=channel.get('Found Via Niche', channel.get('Found Via Keyword', ''))
            )
        niches = defaultdict(list)
        for channel in channels_data:
            niche_key = channel.get('Found Via Niche', channel.get('Found Via Keyword', ''))
            niches[niche_key].append(channel.get('Channel Name', ''))
        for niche, channels in niches.items():
            for i, channel1 in enumerate(channels):
                for channel2 in channels[i+1:]:
                    G.add_edge(channel1, channel2, weight=1.0, niche=niche)
        return G

    @staticmethod
    def calculate_network_centrality(G, node):
        try:
            betweenness = nx.betweenness_centrality(G)[node]
            closeness = nx.closeness_centrality(G)[node]
            degree = nx.degree_centrality(G)[node]
            return {
                'betweenness': betweenness,
                'closeness': closeness,
                'degree': degree,
                'influence_score': (betweenness + closeness + degree) / 3
            }
        except Exception:
            return {'betweenness': 0, 'closeness': 0, 'degree': 0, 'influence_score': 0}

@st.cache_data(ttl=3600)
def fetch_youtube_data(url, params):
    """Fetch data from YouTube API with caching"""
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        # Increment API call count
        if 'daily_api_calls' not in st.session_state:
            st.session_state.daily_api_calls = 0
        st.session_state.daily_api_calls += 1
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

def parse_youtube_duration(duration_str):
    """Parse YouTube duration format (PT1H2M3S) to seconds"""
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration_str)
    if not match:
        return 0
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0
    return hours * 3600 + minutes * 60 + seconds

def format_number(num):
    """Format large numbers for display"""
    if num >= 1000000000:
        return f"{num/1000000000:.1f}B"
    elif num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    else:
        return str(num)

def perform_advanced_analysis(api_key, channel_id, channel_data, analysis_depth):
    """Perform advanced analysis on a single channel"""
    analyzer = GrowthAnalyzer()
    predictor = ViralityPredictor()
    analysis_results = {
        "Engagement Score": 0, "Viral Potential": 0, "Growth Velocity": 0,
        "Growth Acceleration": 0, "Content Consistency": 0, "Monetization Signals": [],
        "Readability Score": 0, "Topic Coherence": 0, "Optimal Upload Times": [],
        "Predicted Growth Trajectory": "Stable"
    }
    
    try:
        video_search_params = {
            "part": "snippet", "channelId": channel_id, "order": "date",
            "maxResults": 25, "key": api_key
        }
        video_response = fetch_youtube_data(YOUTUBE_SEARCH_URL, video_search_params)
        if not video_response or not video_response.get("items"): 
            return analysis_results
        
        video_ids = [item["id"]["videoId"] for item in video_response["items"] if "videoId" in item.get("id", {})]
        if not video_ids: 
            return analysis_results
            
        video_details_params = {
            "part": "statistics,snippet,contentDetails", "id": ",".join(video_ids), "key": api_key
        }
        details_response = fetch_youtube_data(YOUTUBE_VIDEO_URL, video_details_params)
        if not details_response or not details_response.get("items"): 
            return analysis_results

        videos_data = details_response.get("items", [])
        metrics = defaultdict(list)
        for video in videos_data:
            stats, snippet, content_details = video.get("statistics", {}), video.get("snippet", {}), video.get("contentDetails", {})
            metrics['views'].append(int(stats.get("viewCount", 0)))
            metrics['likes'].append(int(stats.get("likeCount", 0)))
            metrics['comments'].append(int(stats.get("commentCount", 0)))
            metrics['titles'].append(snippet.get("title", ""))
            metrics['descriptions'].append(snippet.get("description", ""))
            metrics['durations'].append(parse_youtube_duration(content_details.get("duration", "PT0S")))
            if snippet.get("publishedAt"):
                metrics['publish_dates'].append(datetime.fromisoformat(snippet["publishedAt"].replace("Z", "+00:00")))
        
        if len(metrics['views']) > 2:
            engagement_scores, viral_coefficients = [], []
            subscriber_count = int(channel_data.get("statistics", {}).get("subscriberCount", 1))

            for i in range(len(metrics['views'])):
                if metrics['views'][i] > 0:
                    engagement_scores.append(predictor.engagement_quality_score(
                        metrics['likes'][i], metrics['comments'][i], 
                        metrics['views'][i], metrics['durations'][i]
                    ))
                    time_since = datetime.now(metrics['publish_dates'][i].tzinfo) - metrics['publish_dates'][i]
                    viral_coefficients.append(predictor.calculate_viral_coefficient(
                        metrics['views'][i], time_since, subscriber_count
                    ))
            
            if engagement_scores: 
                analysis_results["Engagement Score"] = np.mean(engagement_scores)
            if viral_coefficients: 
                analysis_results["Viral Potential"] = np.mean(viral_coefficients) * 100

            if len(metrics['publish_dates']) > 3:
                sorted_data = sorted(zip(metrics['publish_dates'], metrics['views']))
                dates, views = zip(*sorted_data)
                time_deltas = [(d - dates[0]).days for d in dates]
                analysis_results["Growth Velocity"] = analyzer.calculate_growth_velocity(views, time_deltas)
                analysis_results["Growth Acceleration"] = analyzer.calculate_growth_acceleration(views, time_deltas)

            if np.mean(metrics['views']) > 0:
                view_cv = np.std(metrics['views']) / np.mean(metrics['views'])
                analysis_results["Content Consistency"] = max(0, 100 - (view_cv * 100))

        all_text = " ".join(metrics['titles'] + metrics['descriptions'])
        if all_text:
            try:
                analysis_results["Readability Score"] = flesch_reading_ease(all_text)
            except Exception:
                analysis_results["Readability Score"] = 50

    except Exception as e:
        st.warning(f"Partial analysis due to: {e}")

    channel_description = channel_data.get("snippet", {}).get("description", "")
    monetization_patterns = {
        'Affiliate': r'affiliate|commission', 
        'Sponsorship': r'sponsor|brand deal', 
        'Merchandise': r'merch|store', 
        'Course': r'course|masterclass', 
        'Patreon': r'patreon|ko-fi'
    }
    detected_signals = [
        sig_type for sig_type, pattern in monetization_patterns.items() 
        if re.search(pattern, channel_description.lower())
    ]
    analysis_results["Monetization Signals"] = detected_signals

    return analysis_results

def find_viral_new_channels_enhanced(api_key, niche_ideas_list, video_type="Any", analysis_depth="Deep"):
    """Find viral new channels for niche research"""
    viral_channels = []
    current_year = datetime.now().year
    progress_bar = st.progress(0)
    status_text = st.empty()
    processed_channel_ids = set()

    for i, niche in enumerate(niche_ideas_list):
        status_text.text(f"🔬 Analyzing niche '{niche}'... ({i + 1}/{len(niche_ideas_list)})")
        progress_bar.progress((i + 1) / len(niche_ideas_list))
        
        search_params = {
            "part": "snippet", "q": niche, "type": "video", "order": "relevance",
            "publishedAfter": (datetime.utcnow() - timedelta(days=120)).isoformat("T") + "Z",
            "maxResults": 30, "key": api_key
        }
        if video_type != "Any":
            search_params['videoDuration'] = 'short' if video_type == "Shorts Channel" else 'long'
        
        search_response = fetch_youtube_data(YOUTUBE_SEARCH_URL, search_params)
        if not search_response or not search_response.get("items"): 
            continue

        new_channel_ids = list({
            item["snippet"]["channelId"] for item in search_response["items"]
        } - processed_channel_ids)
        if not new_channel_ids: 
            continue

        for batch_start in range(0, len(new_channel_ids), 50):
            batch_ids = new_channel_ids[batch_start:batch_start + 50]
            channel_params = {"part": "snippet,statistics", "id": ",".join(batch_ids), "key": api_key}
            channel_response = fetch_youtube_data(YOUTUBE_CHANNEL_URL, channel_params)
            if not channel_response or not channel_response.get("items"): 
                continue

            for channel in channel_response["items"]:
                published_date = datetime.fromisoformat(channel["snippet"]["publishedAt"].replace("Z", "+00:00"))
                if published_date.year >= current_year - 1:
                    stats_data = channel.get("statistics", {})
                    subs = int(stats_data.get("subscriberCount", 0))
                    views = int(stats_data.get("viewCount", 0))
                    video_count = int(stats_data.get("videoCount", 0))
                    subscriber_velocity = subs / max((datetime.now(published_date.tzinfo) - published_date).days, 1)
                    view_to_video_ratio = views / max(video_count, 1)
                    
                    if (subs > 500 and views > 25000 and 3 < video_count < 200 and 
                        subscriber_velocity > 5 and view_to_video_ratio > 1000):
                        
                        channel_id = channel['id']
                        analysis_data = perform_advanced_analysis(api_key, channel_id, channel, analysis_depth)
                        viral_channels.append({
                            "Channel Name": channel["snippet"]["title"],
                            "URL": f"https://www.youtube.com/channel/{channel_id}",
                            "Subscribers": subs,
                            "Total Views": views,
                            "Video Count": video_count,
                            "Creation Date": published_date.strftime("%Y-%m-%d"),
                            "Channel Age (Days)": (datetime.now(published_date.tzinfo) - published_date).days,
                            "Found Via Niche": niche,
                            "Subscriber Velocity": round(subscriber_velocity, 2),
                            "View-to-Video Ratio": round(view_to_video_ratio, 0),
                            **analysis_data
                        })
                        processed_channel_ids.add(channel_id)

    progress_bar.empty()
    status_text.empty()
    if viral_channels:
        return apply_advanced_ranking(viral_channels)
    return viral_channels

def find_channels_with_criteria(api_key, search_params, results_container, enable_unlimited_split=False):
    """Find YouTube channels based on user-defined criteria with unlimited pagination and optional yearly split to bypass API limits"""
    
    keywords = search_params.get('keywords', '')
    channel_type = search_params.get('channel_type', 'Any')
    creation_year = search_params.get('creation_year', None)
    
    description_keyword = search_params.get('description_keyword', '')
    min_subscribers = search_params.get('min_subscribers', 0)
    max_subscribers = search_params.get('max_subscribers', float('inf'))
    min_videos = search_params.get('min_videos', 0)
    max_videos = search_params.get('max_videos', float('inf'))
    min_views = search_params.get('min_views', 0)
    max_views = search_params.get('max_views', float('inf'))
    country = search_params.get('country', 'Any')
    
    found_channels = []
    processed_channel_ids = set()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    search_terms = [term.strip() for term in keywords.split(',') if term.strip()]
    
    total_terms = len(search_terms)
    channels_found = 0
    total_queries = 0
    
    # If unlimited split enabled and no specific creation year, split by years to get more results
    year_ranges = []
    if enable_unlimited_split and (creation_year is None or creation_year == 0):
        current_year = datetime.now().year
        for y in range(2005, current_year + 1):
            year_ranges.append((y, y + 1))
        st.info(f"🔄 Unlimited mode: Splitting search across {len(year_ranges)} years to bypass API limits (up to 500 results/year).")
    else:
        year_ranges = [(None, None)]  # Single query
    
    for term_idx, term in enumerate(search_terms):
        for year_idx, (start_y, end_y) in enumerate(year_ranges):
            total_queries += 1
            page_token = None
            page_count = 0
            
            while True:
                status_text.text(f"🔍 Term: '{term}' | Year: {start_y if start_y else 'All'} | Page: {page_count + 1} | Channels: {channels_found}")
                
                # Progress: rough estimate based on terms * years * pages
                estimated_total = total_terms * len(year_ranges) * 10  # Assume 10 pages max per query
                progress = (total_queries * 10 + page_count) / estimated_total
                progress_bar.progress(min(progress, 1.0))
                
                search_query_params = {
                    "part": "snippet",
                    "q": term,
                    "type": "video",
                    "order": "relevance",
                    "maxResults": 50,
                    "key": api_key
                }
                
                if channel_type == "Short":
                    search_query_params['videoDuration'] = 'short'
                elif channel_type == "Long":
                    search_query_params['videoDuration'] = 'long'
                
                # Apply year range for video publish dates to split queries
                if start_y:
                    start_date = f"{start_y}-01-01T00:00:00Z"
                    end_date = f"{end_y}-01-01T00:00:00Z"
                    search_query_params['publishedAfter'] = start_date
                    search_query_params['publishedBefore'] = end_date
                
                # If specific creation year, apply to channel later (not video)
                if creation_year and creation_year > 1900 and not start_y:
                    start_date = f"{creation_year}-01-01T00:00:00Z"
                    end_date = f"{creation_year + 1}-01-01T00:00:00Z"
                    search_query_params['publishedAfter'] = start_date
                    search_query_params['publishedBefore'] = end_date
                
                if country != "Any":
                    search_query_params['regionCode'] = country
                
                if page_token:
                    search_query_params['pageToken'] = page_token
                
                search_response = fetch_youtube_data(YOUTUBE_SEARCH_URL, search_query_params)
                if not search_response or not search_response.get("items"):
                    break
                
                channel_ids = list(set([
                    item["snippet"]["channelId"] 
                    for item in search_response["items"] 
                    if item["snippet"]["channelId"] not in processed_channel_ids
                ]))
                
                if not channel_ids:
                    break
                
                for batch_start in range(0, len(channel_ids), 50):
                    batch_ids = channel_ids[batch_start:batch_start + 50]
                    
                    channel_params = {
                        "part": "snippet,statistics", "id": ",".join(batch_ids), "key": api_key
                    }
                    
                    channel_response = fetch_youtube_data(YOUTUBE_CHANNEL_URL, channel_params)
                    if not channel_response or not channel_response.get("items"):
                        continue
                    
                    for channel in channel_response["items"]:
                        try:
                            snippet = channel.get("snippet", {})
                            stats = channel.get("statistics", {})
                            
                            channel_name = snippet.get("title", "Unknown")
                            channel_description = snippet.get("description", "")
                            published_date_str = snippet.get("publishedAt", "")
                            if not published_date_str:
                                continue
                            published_date = datetime.fromisoformat(published_date_str.replace("Z", "+00:00"))
                            channel_country = snippet.get("country", "Unknown")
                            
                            subscribers = int(stats.get("subscriberCount", 0))
                            total_views = int(stats.get("viewCount", 0))
                            video_count = int(stats.get("videoCount", 0))
                            
                            # Apply creation year filter on channel creation date
                            if creation_year and creation_year > 1900 and published_date.year != creation_year:
                                continue
                            
                            # Apply subscriber filters
                            if not (min_subscribers <= subscribers <= max_subscribers):
                                continue
                            
                            # Apply video count filters
                            if not (min_videos <= video_count <= max_videos):
                                continue
                            
                            # Apply total views filters
                            if not (min_views <= total_views <= max_views):
                                continue
                            
                            # Apply description keyword filter only if provided
                            if description_keyword and description_keyword.strip() and description_keyword.lower() not in channel_description.lower():
                                continue
                            
                            # Apply country filter only if specified
                            if country != "Any" and channel_country != country:
                                continue
                            
                            channel_age_days = (datetime.now(published_date.tzinfo) - published_date).days
                            avg_views_per_video = total_views / max(video_count, 1)
                            subscriber_velocity = subscribers / max(channel_age_days, 1)
                            
                            channel_data = {
                                "Channel Name": channel_name,
                                "URL": f"https://www.youtube.com/channel/{channel['id']}",
                                "Subscribers": subscribers,
                                "Total Views": total_views,
                                "Video Count": video_count,
                                "Creation Date": published_date.strftime("%Y-%m-%d"),
                                "Channel Age (Days)": channel_age_days,
                                "Found Via Keyword": term,
                                "Subscriber Velocity": round(subscriber_velocity, 4),
                                "Avg Views per Video": round(avg_views_per_video, 0),
                                "Description": channel_description[:200] + "..." if len(channel_description) > 200 else channel_description,
                                "Country": channel_country,
                                "Search Year": start_y if start_y else "All"
                            }
                            
                            found_channels.append(channel_data)
                            processed_channel_ids.add(channel['id'])
                            channels_found += 1
                            
                            # Display channel incrementally
                            with results_container:
                                with st.expander(f"#{channels_found} {channel_data['Channel Name']} • {format_number(channel_data['Subscribers'])} subs | Year: {channel_data.get('Search Year', 'All')}", expanded=(channels_found <= 3)):
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("📊 Subscribers", format_number(channel_data['Subscribers']))
                                    col2.metric("👀 Total Views", format_number(channel_data['Total Views']))
                                    col3.metric("🎬 Videos", channel_data['Video Count'])
                                    
                                    col4, col5, col6 = st.columns(3)
                                    col4.metric("📅 Created", channel_data['Creation Date'])
                                    col5.metric("⏱️ Age", f"{channel_data['Channel Age (Days)']} days")
                                    col6.metric("🔍 Found via", channel_data['Found Via Keyword'])
                                    
                                    col7, col8, col9 = st.columns(3)
                                    col7.metric("🌍 Country", channel_data['Country'])
                                    col8.metric("🚀 Subscriber Velocity", f"{channel_data['Subscriber Velocity']:.4f}")
                                    col9.metric("👀 Avg Views/Video", format_number(channel_data['Avg Views per Video']))
                                    
                                    if channel_data.get('Description'):
                                        st.text_area("📝 Description:", channel_data['Description'], height=80, key=f"desc_finder_inc_{channels_found}_{term_idx}_{year_idx}")
                                    
                                    st.markdown(f"[🔗 Visit Channel]({channel_data['URL']})")
                        except (ValueError, KeyError) as e:
                            continue
                
                page_token = search_response.get('nextPageToken')
                page_count += 1
                if not page_token:
                    break
    
    progress_bar.empty()
    status_text.empty()
    
    return found_channels

def apply_advanced_ranking(channels):
    """Apply advanced ranking algorithm to channels"""
    weights = {
        'subscriber_velocity': 0.25, 'engagement': 0.20, 'viral_potential': 0.20, 
        'growth_velocity': 0.15, 'consistency': 0.10, 'monetization': 0.10
    }
    
    features = []
    for ch in channels:
        features.append([
            ch.get('Subscriber Velocity', 0), ch.get('Engagement Score', 0), 
            ch.get('Viral Potential', 0), ch.get('Growth Velocity', 0), 
            ch.get('Content Consistency', 0), len(ch.get('Monetization Signals', [])) * 10
        ])
    
    if not features: 
        return channels
    
    features_normalized = StandardScaler().fit_transform(features)
    
    for i, channel in enumerate(channels):
        score = np.dot(features_normalized[i], list(weights.values()))
        channel['Intelligence_Score'] = round(score * 100, 2)
        if score > 0.8: 
            channel['Ranking_Tier'] = "🏆 Elite"
        elif score > 0.6: 
            channel['Ranking_Tier'] = "🥇 Excellent"
        elif score > 0.4: 
            channel['Ranking_Tier'] = "🥈 Good"
        else: 
            channel['Ranking_Tier'] = "📈 Emerging"
            
    return sorted(channels, key=lambda x: x.get('Intelligence_Score', 0), reverse=True)

def perform_advanced_channel_analysis(api_key, channels_data):
    """Perform advanced analysis on found channels"""
    
    analyzer = GrowthAnalyzer()
    predictor = ViralityPredictor()
    
    enhanced_channels = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, channel_data in enumerate(channels_data):
        status_text.text(f"🧠 Analyzing: {channel_data['Channel Name']} ({i + 1}/{len(channels_data)})")
        progress_bar.progress((i + 1) / len(channels_data))
        
        try:
            channel_id = channel_data['URL'].split('/')[-1]
            
            video_search_params = {
                "part": "snippet",
                "channelId": channel_id,
                "order": "date",
                "maxResults": 10,
                "key": api_key
            }
            
            video_response = fetch_youtube_data(YOUTUBE_SEARCH_URL, video_search_params)
            
            if video_response and video_response.get("items"):
                video_ids = [
                    item["id"]["videoId"] 
                    for item in video_response["items"] 
                    if "videoId" in item.get("id", {})
                ]
                
                if video_ids:
                    video_details_params = {
                        "part": "statistics,snippet,contentDetails",
                        "id": ",".join(video_ids[:5]),
                        "key": api_key
                    }
                    
                    details_response = fetch_youtube_data(YOUTUBE_VIDEO_URL, video_details_params)
                    
                    if details_response and details_response.get("items"):
                        videos_data = details_response.get("items", [])
                        
                        total_views = sum([int(v.get("statistics", {}).get("viewCount", 0)) for v in videos_data])
                        total_likes = sum([int(v.get("statistics", {}).get("likeCount", 0)) for v in videos_data])
                        total_comments = sum([int(v.get("statistics", {}).get("commentCount", 0)) for v in videos_data])
                        
                        if total_views > 0:
                            engagement_rate = ((total_likes + total_comments) / total_views) * 100
                        else:
                            engagement_rate = 0
                        
                        channel_data["Recent Engagement Rate"] = round(engagement_rate, 3)
                        channel_data["Recent Avg Views"] = round(total_views / len(videos_data), 0)
                        channel_data["Analysis Status"] = "✅ Complete"
                    else:
                        channel_data["Analysis Status"] = "⚠️ Limited Data"
                else:
                    channel_data["Analysis Status"] = "❌ No Videos"
            else:
                channel_data["Analysis Status"] = "❌ Access Denied"
                
        except Exception as e:
            channel_data["Analysis Status"] = f"❌ Error: {str(e)[:30]}"
        
        enhanced_channels.append(channel_data)
    
    progress_bar.empty()
    status_text.empty()
    
    return enhanced_channels

# --- Main Application UI ---

st.markdown("""
<div class="main-header">
    <h1>🚀 YouTube Complete Analytics Platform</h1>
    <h3>Advanced Intelligence Engine for YouTube Success</h3>
    <p>Comprehensive channel discovery, viral content research, and analytics</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("🔧 Configuration Panel")
    api_key = st.text_input("YouTube Data API Key:", type="password", help="Get your API key from Google Cloud Console")
    if api_key: 
        st.success("✅ API Key Configured")
    else: 
        st.error("❌ API Key Required")
    
    st.divider()
    analysis_depth = st.selectbox("Analysis Depth:", ["Quick", "Standard", "Deep"], index=2)
    
    st.markdown("---")
    st.header("🛠️ Advanced Settings")
    
    # Cache management
    if st.button("🗑️ Clear Cache", help="Clear cached API responses"):
        st.cache_data.clear()
        st.success("Cache cleared!")
    
    # Data export format
    export_format = st.selectbox(
        "Default Export Format:",
        ["CSV", "JSON", "Excel"],
        help="Choose default format for data exports"
    )
    
    # Display options
    show_advanced_metrics = st.checkbox(
        "Show Advanced Metrics",
        value=True,
        help="Display engagement rates and growth analytics"
    )
    
    # Search limits
    st.subheader("🔢 Search Limits")
    default_max_results = st.slider(
        "Suggested Max Channels (for guidance):",
        min_value=10,
        max_value=200,
        value=50,
        help="This is now unlimited; use for reference only."
    )
    
    # API usage tracking
    st.subheader("📊 API Usage")
    if 'api_calls_made' not in st.session_state:
        st.session_state.api_calls_made = 0
    
    # Track API calls in cached function
    @st.cache_data(ttl=3600)
    def get_api_call_count():
        if 'daily_api_calls' not in st.session_state:
            st.session_state.daily_api_calls = 0
        return st.session_state.daily_api_calls
    
    current_calls = get_api_call_count()
    st.metric("API Calls Today", current_calls)
    
    # Progress bar for API usage (assuming 10,000 daily limit)
    progress_value = min(current_calls / 10000, 1.0)
    st.progress(progress_value)
    
    if current_calls > 8000:
        st.warning("⚠️ Approaching API daily limit!")
    elif current_calls > 9500:
        st.error("🚫 Very close to API limit!")
    
    # Help and support
    st.markdown("---")
    st.subheader("❓ Help & Support")
    
    with st.expander("📚 Documentation"):
        st.markdown("""
        **Common Issues:**
        
        • **No results found**: Try broader keywords or adjust filters
        • **API errors**: Check your API key and quota limits
        • **Slow searches**: Unlimited mode splits by years—increases API calls
        • **Missing data**: Some channels may have private statistics
        
        **Best Practices:**
        
        • Start with broad searches, then refine
        • Use specific niches for targeted results
        • Check channel creation dates for trends
        • Export data for further analysis
        """)
    
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **If something goes wrong:**
        
        1. **Refresh the page** - Clears temporary issues
        2. **Check API key** - Ensure it's valid and has quota
        3. **Clear cache** - Use the button above
        4. **Monitor API usage** - Unlimited search with split can hit quota fast
        5. **Try different keywords** - Some terms may be restricted
        
        **Error Codes:**
        - 403: API key issue or quota exceeded
        - 400: Invalid search parameters
        - 404: Channel not found or deleted
        """)
    
    # Version and Updates
    st.markdown("---")
    st.markdown("""
    <div style="color: #888; font-size: 0.8em;">
        <p>YouTube Analytics Platform v2.2 (Truly Unlimited)<br>
        Built with Streamlit & YouTube Data API v3<br>
        Last Updated: September 2025</p>
    </div>
    """, unsafe_allow_html=True)

# List of countries with ISO 3166-1 alpha-2 codes
COUNTRY_CODES = {
    "Any": "Any",
    "United States": "US",
    "Canada": "CA",
    "United Kingdom": "GB",
    "Australia": "AU",
    "India": "IN",
    "Germany": "DE",
    "France": "FR",
    "Brazil": "BR",
    "Japan": "JP",
    "South Korea": "KR",
    "Mexico": "MX",
    "Russia": "RU",
    "China": "CN",
    "Italy": "IT",
    "Spain": "ES",
    "Netherlands": "NL",
    "Sweden": "SE",
    "Norway": "NO",
    "South Africa": "ZA",
    "Argentina": "AR",
    "Egypt": "EG",
    "Indonesia": "ID",
    "Turkey": "TR",
    "Saudi Arabia": "SA",
    "Pakistan": "PK",
    "Bangladesh": "BD",
    "Nigeria": "NG",
    "Vietnam": "VN",
    "Philippines": "PH",
    "Thailand": "TH",
    "Malaysia": "MY",
    "Singapore": "SG",
    "New Zealand": "NZ",
    "United Arab Emirates": "AE",
    "Israel": "IL",
    "Ukraine": "UA",
    "Poland": "PL",
    "Switzerland": "CH",
    "Belgium": "BE",
    "Austria": "AT",
    "Denmark": "DK",
    "Finland": "FI",
    "Ireland": "IE",
    "Portugal": "PT",
    "Greece": "GR",
    "Chile": "CL",
    "Colombia": "CO",
    "Peru": "PE",
    "Venezuela": "VE",
    "Ecuador": "EC",
    "Morocco": "MA",
    "Algeria": "DZ",
    "Kenya": "KE",
    "Ghana": "GH",
    "Ethiopia": "ET",
    "Iraq": "IQ",
    "Iran": "IR",
    "Syria": "SY",
    "Jordan": "JO",
    "Lebanon": "LB",
    "Qatar": "QA",
    "Kuwait": "KW",
    "Oman": "OM",
    "Bahrain": "BH",
    "Sri Lanka": "LK",
    "Nepal": "NP",
    "Myanmar": "MM",
    "Cambodia": "KH",
    "Laos": "LA",
    "Mongolia": "MN",
    "Afghanistan": "AF",
    "Yemen": "YE",
    "Sudan": "SD",
    "Uganda": "UG",
    "Angola": "AO",
    "Zimbabwe": "ZW",
    "Zambia": "ZM",
    "Malawi": "MW",
    "Mozambique": "MZ",
    "Botswana": "BW",
    "Namibia": "NA",
    "Tunisia": "TN",
    "Libya": "LY",
    "Jamaica": "JM",
    "Costa Rica": "CR",
    "Panama": "PA",
    "Cuba": "CU",
    "Haiti": "HT",
    "Dominican Republic": "DO",
    "Guatemala": "GT",
    "Honduras": "HN",
    "El Salvador": "SV",
    "Nicaragua": "NI",
    "Paraguay": "PY",
    "Uruguay": "UY",
    "Bolivia": "BO",
    "Serbia": "RS",
    "Croatia": "HR",
    "Bosnia and Herzegovina": "BA",
    "Albania": "AL",
    "Macedonia": "MK",
    "Slovenia": "SI",
    "Montenegro": "ME",
    "Bulgaria": "BG",
    "Romania": "RO",
    "Hungary": "HU",
    "Czech Republic": "CZ",
    "Slovakia": "SK",
    "Latvia": "LV",
    "Lithuania": "LT",
    "Estonia": "EE",
    "Belarus": "BY",
    "Moldova": "MD",
    "Georgia": "GE",
    "Armenia": "AM",
    "Azerbaijan": "AZ",
    "Kazakhstan": "KZ",
    "Uzbekistan": "UZ",
    "Turkmenistan": "TM",
    "Kyrgyzstan": "KG",
    "Tajikistan": "TJ"
}

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Intelligent Niche Research", 
    "🔍 Channel Finder", 
    "🔥 Viral Video Finder",
    "📊 Results Dashboard"
])

# Tab 1: Intelligent Niche Research
with tab1:
    st.header("🚀 Intelligent Niche Research Engine")
    
    video_type_choice = st.radio(
        "Channel Type Focus:", 
        ('Any Content', 'Shorts-Focused', 'Long-Form Content'), 
        horizontal=True
    )
    
    suggested_niches = {
        "AI & Technology": ["AI Tools for Creators", "No-Code SaaS", "Crypto DeFi Explained"],
        "Personal Development": ["Productivity for ADHD", "Financial Independence", "Minimalist Lifestyle"],
        "Entertainment": ["Gaming Reviews", "Movie Reactions", "Comedy Skits"],
        "Education": ["Science Experiments", "History Explained", "Language Learning"],
    }
    
    niche_category = st.selectbox("Choose a Category for Suggestions:", list(suggested_niches.keys()))
    user_niche_input = st.text_area(
        "Enter Niche Ideas (one per line):", 
        "\n".join(suggested_niches[niche_category]), 
        height=150
    )

    if st.button("🚀 Launch Intelligent Analysis", type="primary", use_container_width=True):
        if not api_key:
            st.error("🔐 Please configure your API key in the sidebar.")
        else:
            niche_ideas = [n.strip() for n in user_niche_input.split('\n') if n.strip()]
            if not niche_ideas:
                st.warning("⚠️ Please enter at least one niche idea.")
            else:
                with st.spinner("🔬 Applying advanced mathematical models..."):
                    video_type_map = {
                        'Any Content': 'Any', 
                        'Shorts-Focused': 'Shorts Channel', 
                        'Long-Form Content': 'Long Video Channel'
                    }
                    st.session_state.niche_results = find_viral_new_channels_enhanced(
                        api_key, niche_ideas, video_type_map[video_type_choice], analysis_depth
                    )
                
                if st.session_state.niche_results:
                    st.success(f"🎉 Analysis Complete! Found {len(st.session_state.niche_results)} high-potential channels.")
                else:
                    st.warning("🔍 No channels found matching the criteria. Try adjusting your search.")

    # Display niche research results
    if 'niche_results' in st.session_state and st.session_state.niche_results:
        st.subheader("🔬 Individual Channel Intelligence Reports")
        for i, channel in enumerate(st.session_state.niche_results):
            with st.expander(
                f"#{i+1} {channel['Channel Name']} • {channel.get('Ranking_Tier', 'Unranked')} • Score: {channel.get('Intelligence_Score', 0):.1f}", 
                expanded=(i < 3)
            ):
                col1, col2, col3 = st.columns(3)
                col1.metric("Subscribers", f"{channel['Subscribers']:,}")
                col2.metric("Total Views", f"{channel['Total Views']:,}")
                col3.metric("Videos", channel['Video Count'])
                
                col4, col5, col6 = st.columns(3)
                col4.metric("Engagement Score", f"{channel.get('Engagement Score', 0):.2f}")
                col5.metric("Viral Potential", f"{channel.get('Viral Potential', 0):.2f}%")
                col6.metric("Growth Velocity", f"{channel.get('Growth Velocity', 0):.2f}")
                
                st.markdown(f"[🔗 Visit Channel]({channel['URL']})")

# Tab 2: Channel Finder
with tab2:
    st.header("🎯 Truly Unlimited Channel Discovery & Search")
    
    st.markdown("""
    <div class="question-box">
        <h4>📋 Required Information</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        keywords = st.text_input(
            "🔤 Enter keywords (e.g. 'AI', 'cat', 'funny'):",
            placeholder="gaming, tutorial, cooking",
            help="Separate multiple keywords with commas"
        )
        
        channel_type = st.selectbox(
            "📺 Which channel do you find?",
            ["Any", "Long", "Short"],
            help="Long = Long videos, Short = Short videos/Shorts"
        )
    
    with col2:
        creation_year = st.number_input(
            "📅 Channel Creation Year? (e.g. 2023, or 0 for All):",
            min_value=0,
            max_value=2025,
            value=0,
            help="0 = All years (enables unlimited split)"
        )
        
        enable_unlimited_split = st.checkbox(
            "🔄 Enable Unlimited Split Mode?",
            value=True,
            help="Splits search by years to bypass YouTube's 500-result limit per query. Increases API usage but finds more channels."
        )
    
    st.markdown("""
    <div class="optional-section">
        <h4>🎛️ Optional Filters (Leave blank to skip)</h4>
        <p><em>These filters will only be applied if you enter values. Leave empty to ignore the filter.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        description_keyword = st.text_input(
            "📝 Channel Description Keyword (Optional):",
            value="",
            placeholder="Leave empty to skip this filter",
            help="Find channels with specific words in their description. Leave blank to skip."
        )
        
        min_subscribers = st.number_input(
            "👥 Minimum Channel Subscribers:",
            min_value=0,
            value=0,
            help="Set to 0 to include all channels regardless of subscriber count"
        )
        
        min_videos = st.number_input(
            "🎬 Minimum Channel Videos:",
            min_value=0,
            value=0,
            help="Set to 0 to include all channels regardless of video count"
        )
        
        min_views = st.number_input(
            "👀 Minimum Channel Total Views:",
            min_value=0,
            value=0,
            help="Set to 0 to include all channels regardless of total views"
        )
    
    with col4:
        max_subscribers = st.number_input(
            "👥 Maximum Channel Subscribers:",
            min_value=1,
            value=10000000,
            help="Set high value to include channels with any subscriber count"
        )
        
        max_videos = st.number_input(
            "🎬 Maximum Channel Videos:",
            min_value=1,
            value=100000,
            help="Set high value to include channels with any video count"
        )
        
        max_views = st.number_input(
            "👀 Maximum Channel Total Views:",
            min_value=1,
            value=10000000000,
            help="Set high value to include channels with any total views"
        )
        
        country = st.selectbox(
            "🌍 Channel Country (Optional):",
            list(COUNTRY_CODES.keys()),
            index=0,
            help="Select a country to filter channels by their location. Choose 'Any' to skip this filter."
        )

    results_container = st.container()

    if st.button("🚀 Start Truly Unlimited Channel Discovery", type="primary", use_container_width=True):
        if not api_key:
            st.error("🔐 Please configure your YouTube API key in the sidebar first!")
        elif not keywords:
            st.error("🔤 Please enter at least one keyword to search for!")
        else:
            search_params = {
                'keywords': keywords,
                'channel_type': channel_type,
                'creation_year': creation_year if creation_year > 0 else None,
                'description_keyword': description_keyword,
                'min_subscribers': min_subscribers,
                'max_subscribers': max_subscribers,
                'min_videos': min_videos,
                'max_videos': max_videos,
                'min_views': min_views,
                'max_views': max_views,
                'country': COUNTRY_CODES[country]
            }
            
            with st.spinner("🔍 Starting truly unlimited search... Results appear live. (May take time with split mode)"):
                channels = find_channels_with_criteria(
                    api_key, search_params, results_container, 
                    enable_unlimited_split and (creation_year == 0)
                )
            
            if channels:
                if analysis_depth == "Deep":
                    with st.spinner("🧠 Performing advanced analysis..."):
                        channels = perform_advanced_channel_analysis(api_key, channels)
                
                st.session_state.channel_finder_results = channels
                with results_container:
                    st.success(f"🎉 Truly unlimited discovery complete! Found {len(channels)} channels (across all years/pages until exhausted).")
                    
                    preview_df = pd.DataFrame(channels)
                    st.dataframe(
                        preview_df[['Channel Name', 'Subscribers', 'Video Count', 'Creation Date', 'Country', 'Search Year']].head(10),
                        use_container_width=True
                    )
            else:
                with results_container:
                    st.warning("😔 No channels found matching your criteria. Try adjusting your filters or keywords.")

    # Display channel finder results
    if 'channel_finder_results' in st.session_state and st.session_state.channel_finder_results:
        with results_container:
            st.subheader("📊 Channel Discovery Results")
            channels_data = st.session_state.channel_finder_results
            
            for i, channel in enumerate(channels_data[:20]):  # Show first 20
                with st.expander(f"#{i+1} {channel['Channel Name']} • {format_number(channel['Subscribers'])} subscribers"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("📊 Subscribers", format_number(channel['Subscribers']))
                    col2.metric("👀 Total Views", format_number(channel['Total Views']))
                    col3.metric("🎬 Videos", channel['Video Count'])
                    
                    col4, col5, col6 = st.columns(3)
                    col4.metric("📅 Created", channel['Creation Date'])
                    col5.metric("⏱️ Age", f"{channel['Channel Age (Days)']} days")
                    col6.metric("🔍 Found via", channel['Found Via Keyword'])
                    
                    col7, col8, col9 = st.columns(3)
                    col7.metric("🌍 Country", channel['Country'])
                    col8.metric("📅 Search Year", channel.get('Search Year', 'All'))
                    if 'Recent Engagement Rate' in channel:
                        col9.metric("💝 Engagement Rate", f"{channel['Recent Engagement Rate']:.3f}%")
                    elif 'Recent Avg Views' in channel:
                        col9.metric("📈 Recent Avg Views", format_number(channel['Recent Avg Views']))
                    
                    if channel.get('Description'):
                        st.text_area("📝 Description:", channel['Description'], height=80, key=f"desc_finder_{i}")
                    
                    st.markdown(f"[🔗 Visit Channel]({channel['URL']})")

# Tab 3: Viral Video Finder
with tab3:
    st.header("🔥 Viral Video Discovery Engine")
    
    col1, col2 = st.columns(2)
    with col1:
        days = st.number_input("📅 Days to search back (1-30):", min_value=1, max_value=30, value=7)
    with col2:
        max_subs = st.number_input("👥 Max channel subscribers:", min_value=100, max_value=10000, value=3000)
    
    keywords = st.text_area(
        "🔤 Enter Keywords (one per line):",
        "AI tutorial\nCoding for beginners\nProductivity hacks\nMinecraft builds",
        height=120
    )

    if st.button("🔍 Find Viral Videos", type="primary", use_container_width=True):
        if not api_key:
            st.error("🔐 Please configure your API key in the sidebar.")
        else:
            keyword_list = [k.strip() for k in keywords.split('\n') if k.strip()]
            if not keyword_list:
                st.warning("⚠️ Please enter at least one keyword.")
            else:
                with st.spinner("🔍 Searching for viral videos..."):
                    start_date = (datetime.utcnow() - timedelta(days=int(days))).isoformat("T") + "Z"
                    all_results = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, keyword in enumerate(keyword_list):
                        status_text.text(f"🔍 Searching for: {keyword}")
                        progress_bar.progress((i + 1) / len(keyword_list))
                        
                        search_params = {
                            "part": "snippet",
                            "q": keyword,
                            "type": "video",
                            "order": "viewCount",
                            "publishedAfter": start_date,
                            "maxResults": 10,
                            "key": api_key,
                        }
                        
                        search_response = fetch_youtube_data(YOUTUBE_SEARCH_URL, search_params)
                        if not search_response or not search_response.get("items"):
                            continue
                        
                        videos = search_response["items"]
                        video_ids = [v["id"]["videoId"] for v in videos if "id" in v and "videoId" in v["id"]]
                        channel_ids = [v["snippet"]["channelId"] for v in videos if "snippet" in v]
                        
                        if not video_ids or not channel_ids:
                            continue
                        
                        # Get video statistics
                        stats_params = {"part": "statistics", "id": ",".join(video_ids), "key": api_key}
                        stats_response = fetch_youtube_data(YOUTUBE_VIDEO_URL, stats_params)
                        if not stats_response or not stats_response.get("items"):
                            continue
                        
                        # Get channel statistics
                        channel_params = {"part": "statistics", "id": ",".join(set(channel_ids)), "key": api_key}
                        channel_response = fetch_youtube_data(YOUTUBE_CHANNEL_URL, channel_params)
                        if not channel_response or not channel_response.get("items"):
                            continue
                        
                        # Create channel lookup
                        channel_lookup = {ch["id"]: ch for ch in channel_response["items"]}
                        
                        # Process results
                        for video, stat in zip(videos, stats_response["items"]):
                            channel_id = video["snippet"]["channelId"]
                            channel_data = channel_lookup.get(channel_id)
                            
                            if not channel_data:
                                continue
                                
                            subs = int(channel_data["statistics"].get("subscriberCount", 0))
                            if subs <= max_subs:
                                title = video["snippet"].get("title", "N/A")
                                description = video["snippet"].get("description", "")[:200]
                                video_url = f"https://www.youtube.com/watch?v={video['id']['videoId']}"
                                views = int(stat["statistics"].get("viewCount", 0))
                                
                                all_results.append({
                                    "Title": title,
                                    "Description": description,
                                    "URL": video_url,
                                    "Views": views,
                                    "Subscribers": subs,
                                    "Keyword": keyword
                                })
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if all_results:
                        st.session_state.viral_results = all_results
                        st.success(f"🎉 Found {len(all_results)} viral videos from small channels!")
                        
                        # Display top results
                        sorted_results = sorted(all_results, key=lambda x: x['Views'], reverse=True)
                        for i, result in enumerate(sorted_results[:15]):
                            with st.expander(f"#{i+1} {result['Title'][:60]}... • {format_number(result['Views'])} views"):
                                st.markdown(f"**👥 Channel Subscribers:** {format_number(result['Subscribers'])}")
                                st.markdown(f"**👀 Views:** {format_number(result['Views'])}")
                                st.markdown(f"**🔍 Found via:** {result['Keyword']}")
                                st.markdown(f"**📝 Description:** {result['Description']}")
                                st.markdown(f"[🔗 Watch Video]({result['URL']})")
                    else:
                        st.warning("😔 No viral videos found matching your criteria.")

# Tab 4: Results Dashboard
with tab4:
    st.header("📊 Comprehensive Results Dashboard")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    niche_count = len(st.session_state.get('niche_results', []))
    channel_count = len(st.session_state.get('channel_finder_results', []))
    viral_count = len(st.session_state.get('viral_results', []))
    total_channels = niche_count + channel_count
    
    col1.metric("🔬 Niche Research Channels", niche_count)
    col2.metric("🔍 Channel Finder Results", channel_count)
    col3.metric("🔥 Viral Videos Found", viral_count)
    col4.metric("📊 Total Channels Analyzed", total_channels)
    
    # Export all results
    if total_channels > 0 or viral_count > 0:
        st.subheader("💾 Export All Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download Channel Data as CSV", use_container_width=True):
                all_channel_data = []
                if 'niche_results' in st.session_state:
                    all_channel_data.extend(st.session_state.niche_results)
                if 'channel_finder_results' in st.session_state:
                    all_channel_data.extend(st.session_state.channel_finder_results)
                
                if all_channel_data:
                    df_export = pd.DataFrame(all_channel_data)
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"youtube_channels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("🎬 Download Viral Videos as CSV", use_container_width=True):
                if 'viral_results' in st.session_state:
                    df_viral = pd.DataFrame(st.session_state.viral_results)
                    csv_viral = df_viral.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Viral CSV",
                        data=csv_viral,
                        file_name=f"viral_videos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with col3:
            if st.button("📋 Download Complete Report", use_container_width=True):
                report_data = {
                    'summary': {
                        'niche_channels': niche_count,
                        'found_channels': channel_count,
                        'viral_videos': viral_count,
                        'total_channels': total_channels,
                        'generated_at': datetime.now().isoformat()
                    }
                }
                
                if 'niche_results' in st.session_state:
                    report_data['niche_research'] = st.session_state.niche_results
                if 'channel_finder_results' in st.session_state:
                    report_data['channel_finder'] = st.session_state.channel_finder_results
                if 'viral_results' in st.session_state:
                    report_data['viral_videos'] = st.session_state.viral_results
                
                import json
                json_str = json.dumps(report_data, indent=2, default=str)
                st.download_button(
                    label="📥 Download JSON Report",
                    data=json_str,
                    file_name=f"youtube_complete_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Network analysis if we have multiple channels
        if total_channels > 5:
            st.subheader("🌐 Network Analysis")
            
            try:
                # Combine all channel data
                all_channels = []
                if 'niche_results' in st.session_state:
                    all_channels.extend(st.session_state.niche_results)
                if 'channel_finder_results' in st.session_state:
                    all_channels.extend(st.session_state.channel_finder_results)
                
                # Build network
                network_analyzer = NetworkAnalyzer()
                G = network_analyzer.build_topic_network(all_channels)
                
                if G.number_of_nodes() > 0:
                    # Calculate network metrics
                    centrality_scores = {}
                    for node in G.nodes():
                        centrality_scores[node] = network_analyzer.calculate_network_centrality(G, node)
                    
                    # Display top influential channels
                    st.markdown("**🏆 Most Influential Channels in Network:**")
                    sorted_influence = sorted(
                        centrality_scores.items(), 
                        key=lambda x: x[1]['influence_score'], 
                        reverse=True
                    )
                    
                    for i, (channel, scores) in enumerate(sorted_influence[:10]):
                        st.markdown(f"{i+1}. **{channel}** - Influence Score: {scores['influence_score']:.3f}")
                
            except Exception as e:
                st.info(f"🌐 Network analysis failed: {str(e)[:100]}")
    
    else:
        st.info("📊 Run analyses in other tabs to see comprehensive dashboard results.")

# Additional Features Section
st.markdown("---")
st.header("🔧 Additional Tools & Features")

col1, col2, col3 = st.columns(3)

with col1:
    with st.expander("🎯 Search Tips"):
        st.markdown("""
        **Effective Search Strategies:**
        
        • Use specific, relevant keywords
        • Combine multiple related terms with commas
        • Set creation year to 0 + enable split for truly unlimited results
        • Adjust subscriber limits to find your target audience size
        • Use description keywords to filter by niche topics
        • Monitor API quota—split mode uses more calls
        
        **Examples:**
        - "AI tutorial, machine learning, python coding"
        - "cooking recipes, healthy meals, quick dinner"
        - "gaming review, indie games, retro gaming"
        """)

with col2:
    with st.expander("📊 Understanding Metrics"):
        st.markdown("""
        **Key Metrics Explained:**
        
        • **Subscribers**: Total channel followers
        • **Total Views**: Cumulative views across all videos
        • **Video Count**: Number of uploaded videos
        • **Engagement Rate**: (Likes + Comments) / Views ratio
        • **Subscriber Velocity**: Subscribers gained per day
        • **Channel Age**: Days since channel creation
        • **Search Year**: Year range used in the split query
        
        **Growth Indicators:**
        - High engagement rate = Active audience
        - High subscriber velocity = Fast growing
        - Good view-to-video ratio = Consistent quality
        """)

with col3:
    with st.expander("🔗 API Setup Guide"):
        st.markdown("""
        **Getting Your YouTube API Key:**
        
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project or select existing
        3. Enable YouTube Data API v3
        4. Create credentials (API Key)
        5. Copy and paste the key in sidebar
        
        **API Limits:**
        - 10,000 requests per day (free tier)
        - Each search uses ~3-5 quota units
        - Split mode: ~50x more calls for full history
        - Be cautious with unlimited!
        
        **Cost:** Free up to daily quota limit
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <h4>🚀 Complete YouTube Analytics Platform (Truly Unlimited Edition)</h4>
    <p>This comprehensive platform combines intelligent niche research, advanced channel discovery, 
    and viral video detection. It uses mathematical models and machine learning algorithms to 
    provide deep insights into YouTube ecosystem dynamics.</p>
    <br>
    <p><strong>✨ Features:</strong> Niche Research • Truly Unlimited Channel Discovery (Yearly Split) • Viral Video Detection • Advanced Analytics</p>
    <p><em>Powered by YouTube Data API v3, advanced mathematical models, and AI-driven intelligence</em></p>
</div>
""", unsafe_allow_html=True)
