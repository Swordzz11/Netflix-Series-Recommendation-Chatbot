# app.py — Smart Netflix recommender (starter list + export/import + negative feedback)
import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Config & API Key
# -------------------------
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_ENABLED = bool(TMDB_API_KEY)

# -------------------------
# Load data & model (cached)
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df = df.dropna(subset=['title', 'description', 'listed_in']).reset_index(drop=True)
    return df

@st.cache_resource
def load_model_and_embeddings(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embs = model.encode(df['description'].tolist(), convert_to_tensor=False)
    return model, np.array(embs)

df = load_data()
model, embeddings = load_model_and_embeddings(df)
title_to_index = {t.lower(): i for i, t in enumerate(df['title'].tolist())}

# -------------------------
# Session state init
# -------------------------
if 'watched' not in st.session_state:
    st.session_state['watched'] = []        # list of titles
if 'ratings' not in st.session_state:
    st.session_state['ratings'] = []        # parallel list of int ratings (1-5)
if 'hidden' not in st.session_state:
    st.session_state['hidden'] = set()      # titles user marked "Not Interested"
if 'likes' not in st.session_state:
    st.session_state['likes'] = set()       # titles the user liked from recs

# -------------------------
# TMDb helpers (cached)
# -------------------------
@st.cache_data
def fetch_tmdb_for_title(title):
    """Return (rating, popularity, poster_url) for a title using TMDb search (or None values)."""
    if not TMDB_ENABLED:
        return None, None, None
    try:
        resp = requests.get("https://api.themoviedb.org/3/search/multi",
                            params={"api_key": TMDB_API_KEY, "query": title}, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        if data.get("results"):
            first = data["results"][0]
            return first.get("vote_average"), first.get("popularity"), (f"https://image.tmdb.org/t/p/w300{first.get('poster_path')}") if first.get("poster_path") else None
    except Exception:
        return None, None, None
    return None, None, None

@st.cache_data
def fetch_trending_titles_from_tmdb(limit=30):
    """Return list of trending titles from TMDb (title/name)."""
    if not TMDB_ENABLED:
        return []
    try:
        resp = requests.get("https://api.themoviedb.org/3/trending/all/week",
                            params={"api_key": TMDB_API_KEY, "language": "en-US"}, timeout=8)
        resp.raise_for_status()
        items = resp.json().get("results", [])[:limit]
        titles = []
        for it in items:
            t = it.get("title") or it.get("name")
            if t:
                titles.append(t)
        return titles
    except Exception:
        return []

# -------------------------
# Utility: starter list creation
# -------------------------
STATIC_STARTERS = [
    "Stranger Things", "Money Heist", "Black Mirror", "The Crown", "Narcos",
    "Ozark", "Dark", "Mindhunter", "Peaky Blinders", "The Queen's Gambit",
    "Lupin", "Bridgerton", "Squid Game", "The Witcher"
]

def map_titles_to_dataset(candidate_list, max_items=8):
    mapped = []
    for t in candidate_list:
        idx = title_to_index.get(t.lower())
        if idx is not None:
            mapped.append(df.at[idx, 'title'])
        if len(mapped) >= max_items:
            break
    return mapped

def get_starter_titles(max_items=8):
    # prefer TMDb trending if available
    if TMDB_ENABLED:
        trending = fetch_trending_titles_from_tmdb(limit=40)
        mapped = map_titles_to_dataset(trending, max_items)
        if len(mapped) >= max_items:
            return mapped
    # fallback static list
    return map_titles_to_dataset(STATIC_STARTERS, max_items)

# -------------------------
# Small search helpers
# -------------------------
def search_titles(query, top_n=12):
    if not query:
        return []
    mask = df['title'].str.contains(query, case=False, na=False)
    matches = df[mask]['title'].unique().tolist()
    if len(matches) < 6:
        tokens = [t for t in query.split() if len(t) > 2]
        if tokens:
            mask2 = df['title'].str.contains("|".join(tokens), case=False, na=False)
            for t in df[mask2]['title'].unique().tolist():
                if t not in matches:
                    matches.append(t)
    return matches[:top_n]

# -------------------------
# Profile vector (weighted by user ratings)
# -------------------------
def compute_profile_vector(watched_titles, watched_ratings):
    idxs = []
    weights = []
    for t, r in zip(watched_titles, watched_ratings):
        idx = title_to_index.get(t.lower())
        if idx is not None:
            idxs.append(idx)
            weights.append(r)
    if not idxs:
        return None, []
    arr = embeddings[np.array(idxs)]
    w = np.array(weights).astype(float)
    profile = (arr * w[:,None]).sum(axis=0) / (w.sum())
    return profile, idxs

# -------------------------
# Recommendation pipeline
# -------------------------
def get_candidate_indices(profile_vec, exclude_idxs, top_k=300):
    sims = cosine_similarity([profile_vec], embeddings)[0]
    top_idxs = sims.argsort()[-top_k:][::-1]
    candidates = [i for i in top_idxs if i not in exclude_idxs]
    sim_vals = sims[candidates]
    return candidates, sim_vals

def enrich_candidates(cand_idxs, sim_vals):
    # normalize sim
    mn, mx = float(sim_vals.min()), float(sim_vals.max())
    sim_norm = (sim_vals - mn) / (mx - mn + 1e-9) if mx != mn else np.ones_like(sim_vals)
    candidates = []
    # fetch tmdb info for each
    tmdb_ratings = []
    tmdb_pops = []
    for i, s_val, s_norm in zip(cand_idxs, sim_vals, sim_norm):
        title = df.at[i,'title']
        rating, pop, poster = fetch_tmdb_for_title(title) if TMDB_ENABLED else (None, None, None)
        r = rating if rating else 0.0
        p = pop if pop else 0.0
        candidates.append({
            "idx": i,
            "title": title,
            "type": df.at[i,'type'],
            "genres": df.at[i,'listed_in'],
            "cast": df.at[i,'cast'] if pd.notna(df.at[i,'cast']) else "",
            "description": df.at[i,'description'],
            "sim": float(s_val),
            "sim_norm": float(s_norm),
            "rating": float(r),
            "popularity": float(p),
            "poster": poster
        })
        tmdb_ratings.append(r)
        tmdb_pops.append(p)
    # normalize ratings / popularity
    rating_arr = np.array(tmdb_ratings)
    pop_arr = np.array(tmdb_pops)
    rating_norm = rating_arr / 10.0
    pop_div = pop_arr.max() if pop_arr.max() > 0 else 1.0
    pop_norm = pop_arr / pop_div
    for i, c in enumerate(candidates):
        c["rating_norm"] = float(rating_norm[i])
        c["pop_norm"] = float(pop_norm[i])
    return candidates

def score_and_sort(candidates, sim_w, rating_w, pop_w):
    # normalize weights
    total = sim_w + rating_w + pop_w
    if total > 0:
        sim_w, rating_w, pop_w = sim_w/total, rating_w/total, pop_w/total
    else:
        sim_w, rating_w, pop_w = 1.0, 0.0, 0.0
    for c in candidates:
        c["final_score"] = sim_w * c["sim_norm"] + rating_w * c["rating_norm"] + pop_w * c["pop_norm"]
    candidates = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
    return candidates

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Netflix Recommender — Upgraded", page_icon="🎬", layout="wide")
st.title("🎬 Netflix Recommender — Starter, Export/Import, and Feedback")

# Left column: profile builder
left, right = st.columns([3,5])
with left:
    st.header("1) Build / Manage Profile")
    st.write("Search and add titles you watched; rate them 1 (disliked) to 5 (loved).")
    q = st.text_input("Search title to add", key="search_q")
    matches = search_titles(q) if q else []
    picked = None
    if matches:
        picked = st.selectbox("Matches", ["-- pick --"] + matches, key="match_picker")
    else:
        picked = "-- pick --"

    if st.button("Add selected to watched"):
        if picked and picked != "-- pick --":
            if picked in st.session_state['watched']:
                st.warning("Already in watched.")
            else:
                st.session_state['watched'].append(picked)
                st.session_state['ratings'].append(5)
                st.success(f"Added {picked}")
        else:
            st.warning("Pick a title first.")

    # quick starter list
    if st.button("Add popular starter list (one-click)"):
        starter = get_starter_titles(max_items=8)
        added = 0
        for t in starter:
            if t not in st.session_state['watched']:
                st.session_state['watched'].append(t)
                st.session_state['ratings'].append(4)  # default liked starter
                added += 1
        st.success(f"Added {added} titles to your watched list.")

    # show watched list with rating sliders and remove button
    st.markdown("**Your watched list:**")
    if not st.session_state['watched']:
        st.info("No titles yet — add some or use the starter list.")
    else:
        for i, title in enumerate(list(st.session_state['watched'])):
            cols = st.columns([3,1,1])
            cols[0].write(f"**{title}**")
            idx = title_to_index.get(title.lower())
            if idx is not None:
                cols[0].write(f"_Genres:_ {df.at[idx,'listed_in']}  \n_Cast:_ {df.at[idx,'cast'] if pd.notna(df.at[idx,'cast']) else 'N/A'}")
            new_rating = cols[1].slider("Rating", 1, 5, value=st.session_state['ratings'][i], key=f"r_{i}")
            st.session_state['ratings'][i] = int(new_rating)
            if cols[2].button("Remove", key=f"rem_{i}"):
                st.session_state['watched'].pop(i); st.session_state['ratings'].pop(i)
                st.experimental_rerun()

    # export / import / clear
    st.markdown("---")
    st.download_button("Export profile (JSON)", data=json.dumps({
        "watched": st.session_state['watched'],
        "ratings": st.session_state['ratings']
    }), file_name="netflix_profile.json", mime="application/json")

    uploaded = st.file_uploader("Import profile (JSON)", type=['json'])
    if uploaded:
        try:
            payload = json.load(uploaded)
            w = payload.get("watched", [])
            r = payload.get("ratings", [])
            if isinstance(w, list) and isinstance(r, list) and len(w) == len(r):
                st.session_state['watched'] = w
                st.session_state['ratings'] = r
                st.success("Profile imported.")
                st.experimental_rerun()
            else:
                st.error("Imported file invalid (need 'watched' and 'ratings' lists of same length).")
        except Exception as e:
            st.error("Failed to read JSON: " + str(e))

    if st.button("Clear profile"):
        st.session_state['watched'] = []; st.session_state['ratings'] = []
        st.success("Profile cleared.")
        st.experimental_rerun()

with right:
    st.header("2) Recommendation Settings & Generate")
    st.write("Adjust how similarity vs rating vs popularity influence the recommendations.")
    sim_w = st.slider("Similarity weight", 0.0, 1.0, 0.6, step=0.05)
    rating_w = st.slider("TMDb rating weight", 0.0, 1.0, 0.3, step=0.05)
    pop_w = st.slider("TMDb popularity weight", 0.0, 1.0, 0.1, step=0.05)
    total = sim_w + rating_w + pop_w
    if total > 0:
        # normalize for display explanation only
        pass
    genre_filter = st.selectbox("Optional genre filter", ["All"] + sorted({g.strip() for gg in df['listed_in'] for g in gg.split(',')}))
    n_recs = st.number_input("Number of recommendations to show", 1, 20, 8)

    st.markdown("---")
    if st.button("Generate recommendations"):
        if not st.session_state['watched']:
            st.warning("Add at least one title you watched first.")
        else:
            profile_vec, watched_idxs = compute_profile_vector(st.session_state['watched'], st.session_state['ratings'])
            if profile_vec is None:
                st.error("Couldn't compute profile vector.")
            else:
                exclude_idxs = set(watched_idxs)
                # also exclude hidden titles
                for h in st.session_state['hidden']:
                    idx = title_to_index.get(h.lower())
                    if idx is not None:
                        exclude_idxs.add(idx)
                cand_idxs, sim_vals = get_candidate_indices(profile_vec, exclude_idxs, top_k=300)
                if len(cand_idxs) == 0:
                    st.info("No candidates found.")
                else:
                    candidates = enrich_candidates(cand_idxs, sim_vals)
                    candidates = score_and_sort(candidates, sim_w, rating_w, pop_w)
                    # apply genre filter
                    if genre_filter != "All":
                        candidates = [c for c in candidates if genre_filter.lower() in c['genres'].lower()]
                    if not candidates:
                        st.info("No candidates after filters.")
                    else:
                        st.success(f"Showing top {min(n_recs, len(candidates))} recommendations")
                        for c in candidates[:n_recs]:
                            # skip if hidden (defensive)
                            if c['title'] in st.session_state['hidden']:
                                continue
                            cols = st.columns([1,4,2])
                            # poster
                            if c['poster']:
                                try:
                                    cols[0].image(c['poster'], width=140)
                                except Exception:
                                    cols[0].write("Poster\nN/A")
                            else:
                                cols[0].write("Poster\nN/A")
                            # main info
                            cols[1].markdown(f"### {c['title']}  ({c['type']})")
                            cols[1].write(f"**Genres:** {c['genres']}")
                            cols[1].write(f"**Cast:** {c['cast'] if c['cast'] else 'N/A'}")
                            cols[1].write(c['description'])
                            # actions and scores
                            cols[2].write(f"**Score:** {c['final_score']:.3f}")
                            if TMDB_ENABLED:
                                cols[2].write(f"⭐ TMDb: {c['rating']}  | 🔥 Pop: {int(c['popularity'])}")
                            cols[2].write(f"🔗 Sim: {c['sim_norm']:.3f}")

                            # Buttons for feedback
                            if cols[2].button("Not Interested", key=f"hide_{c['idx']}"):
                                st.session_state['hidden'].add(c['title'])
                                st.success(f"Hidden {c['title']}. It will be excluded from future recommendations.")
                                st.experimental_rerun()
                            if cols[2].button("Add to watched (Like)", key=f"like_{c['idx']}"):
                                if c['title'] not in st.session_state['watched']:
                                    st.session_state['watched'].append(c['title'])
                                    st.session_state['ratings'].append(5)
                                st.session_state['likes'].add(c['title'])
                                st.success(f"Added {c['title']} to watched (rating 5).")
                                st.experimental_rerun()

                            # explanation: which watched titles are closest contributors
                            watched_embs = embeddings[watched_idxs]
                            cand_emb = embeddings[c['idx']].reshape(1, -1)
                            contrib_sims = cosine_similarity(cand_emb, watched_embs)[0]
                            top_k = min(2, len(contrib_sims))
                            if top_k > 0:
                                top_order = np.argsort(contrib_sims)[-top_k:][::-1]
                                contrib = [f"{st.session_state['watched'][ti]} (sim {contrib_sims[ti]:.2f}, rated {st.session_state['ratings'][ti]})"
                                           for ti in top_order]
                                cols[2].write("Because: " + "; ".join(contrib))

                            # shared genre / cast overlap
                            watched_genres = set(g.strip().lower() for gg in df.loc[watched_idxs,'listed_in'] for g in gg.split(','))
                            cand_genres = set(g.strip().lower() for g in c['genres'].split(','))
                            overlap = watched_genres.intersection(cand_genres)
                            if overlap:
                                cols[2].write("Shared genres: " + ", ".join(sorted(overlap)))
                            st.markdown("---")

st.caption("Tip: start with 4–8 titles you liked. Use 'Not Interested' to remove noisy recommendations; 'Add to watched' helps the profile learn quickly.")
