import streamlit as st
import pandas as pd
import plotly.express as px
import ast
from pathlib import Path

st.set_page_config(layout="wide")
st.title('Hashtag Clusters Explorer')

# --- Helper Function for Cleaning Rules ---
def clean_itemset_string(itemset_str):
    """
    Safely converts a string representation of a set/list (e.g., "frozenset({'#ai'})")
    into a clean comma-separated string (e.g., "#ai").
    """
    if not isinstance(itemset_str, str):
        return str(itemset_str)
    
    # Remove "frozenset({" and "})" wrapper if present
    clean_str = itemset_str.replace("frozenset({", "").replace("})", "").replace("{", "").replace("}", "")
    
    # Evaluate if it looks like a list/tuple string
    try:
        # If it's a list string "['#ai', '#ml']", parse it
        if clean_str.startswith("[") and clean_str.endswith("]"):
            items = ast.literal_eval(clean_str)
            return ", ".join(sorted(items))
    except (ValueError, SyntaxError):
        pass

    # Fallback: manual string cleaning for single quotes
    clean_str = clean_str.replace("'", "").replace('"', "")
    return clean_str

# --- Load Data ---
@st.cache_data
def load_data(output_dir):
    cluster_file = output_dir / 'clustered_tweets.csv'
    summary_file = output_dir / 'cluster_summary.csv'
    rules_file = output_dir / 'apriori_rules_ALL_raw.csv' 
    umap_html = output_dir / 'interactive_umap_plot.html'

    # Load Clusters
    try:
        df = pd.read_csv(cluster_file)
    except FileNotFoundError:
        st.error(f"Could not find {cluster_file}")
        return None, None, None, None, None

    # Load Summary
    summary = pd.read_csv(summary_file) if summary_file.exists() else None
    
    # Load Rules (with safe parsing)
    rules = None
    if rules_file.exists():
        try:
            rules = pd.read_csv(rules_file)
            # Apply cleaning immediately after load
            if 'antecedents' in rules.columns:
                rules['antecedents'] = rules['antecedents'].apply(clean_itemset_string)
            if 'consequents' in rules.columns:
                rules['consequents'] = rules['consequents'].apply(clean_itemset_string)
        except Exception as e:
            st.warning(f"Could not load rules file: {e}")
    
    # Load UMAP HTML
    umap_html_content = umap_html.read_text() if umap_html.exists() else None
    
    # Find per-cluster plots
    cluster_plot_files = sorted(list(output_dir.glob('top_hashtags_cluster_*.html')))
    
    return df, summary, rules, umap_html_content, cluster_plot_files

# --- Main App Execution ---
# Use current directory
APP_OUTPUT_DIR = Path('.') 

df, summary, rules, umap_html_content, cluster_plot_files = load_data(APP_OUTPUT_DIR)

if df is None:
    st.stop()

clusters = sorted(df['cluster'].unique())

# --- Sidebar ---
st.sidebar.title("Controls")
default_index = 0
if 0 in clusters:
    default_index = clusters.index(0)
elif -1 in clusters:
    default_index = clusters.index(-1)

selected_cluster = st.sidebar.selectbox('Select Cluster:', clusters, index=default_index)

# --- Main Page ---

# 1. UMAP Plot
st.header("Interactive Cluster Map")
if umap_html_content:
    st.components.v1.html(umap_html_content, height=600, scrolling=True)
else:
    st.warning("Could not find 'interactive_umap_plot.html'.")

# 2. Cluster Summary Table
if summary is not None:
    st.header("Cluster Summary")
    st.dataframe(summary)

# 3. Association Rules Scatter Plot
st.header("Association Rules Explorer")
rules_plot_file = APP_OUTPUT_DIR / 'apriori_rules_scatterplot.html'
if rules_plot_file.exists():
    st.components.v1.html(rules_plot_file.read_text(), height=500, scrolling=True)
else:
    st.warning("Could not find 'apriori_rules_scatterplot.html'.")


# 4. Selected Cluster Details
st.header(f"Details for Cluster: {selected_cluster}")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    # 5. Top Hashtags Plot
    st.subheader("Top Hashtags")
    plot_path = APP_OUTPUT_DIR / f'top_hashtags_cluster_{selected_cluster}.html'
    if plot_path.exists():
        st.components.v1.html(plot_path.read_text(), height=400)
    else:
        st.write("No top hashtags plot found for this cluster.")

    # 6. Rules Table
    st.subheader("Top Association Rules")
    if rules is not None:
        # Filter for selected cluster
        cluster_rules = rules[rules['cluster'] == selected_cluster].copy()
        
        if cluster_rules.empty:
            st.write("No association rules found for this cluster.")
        else:
            # Sort by Lift
            cluster_rules = cluster_rules.sort_values('lift', ascending=False)
            st.dataframe(cluster_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20))
    else:
        st.write("No rules data available.")

with col2:
    # 7. Sample Tweets
    st.subheader("Sample Posts")
    sub_df = df[df['cluster'] == selected_cluster]
    st.write(f"Total posts in cluster: {len(sub_df)}")
    
    try:
        # Safe display of hashtags
        if 'hashtags_str' not in sub_df.columns:
             # Try to clean up the list string if it exists
             sub_df['hashtags_str'] = sub_df['hashtags'].astype(str).apply(lambda x: x.replace("[","").replace("]","").replace("'",""))
        
        st.dataframe(sub_df[['clean_text', 'hashtags_str']].head(50), height=600)
    except Exception as e:
        st.error(f"Error display tweets: {e}")
        st.dataframe(sub_df.head(10))
