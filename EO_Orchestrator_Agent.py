import streamlit as st
import google.generativeai as genai
import requests
import json
import uuid
import datetime
import os
import re
import subprocess
import sys
import atexit
import textwrap
import hashlib
import pandas as pd
import random
import zipfile
import io
import tempfile
import shutil
import gzip
import urllib3
import signal
import contextlib
import html
import logging
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD, RDFS, FOAF

# Disable insecure request warnings for self-signed or academic university SSL endpoints
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= SYSTEM CONSOLE =================
# There was previously no way to surface a silent failure to the user. log_error() is
# the central sink every silent `except` in this file now reports to (instead of just
# `pass`-ing), and AGENT_LOG_FILES are where each agent's own log output lives so it
# can be tailed from the Console tab (System Admin) instead of only appearing in a
# console the end user can't see.
ORCHESTRATOR_LOG_FILE = "EO_Orchestrator_Agent.log"
AGENT_LOG_FILES = {
    "Orchestrator": ORCHESTRATOR_LOG_FILE,
    "Recommender": "EO_Recommender_Agent.log",
    "Explainer": "EO_Explainer_Agent.log",
    "ProductReviews": "EO_ProductReviews_Agent.log",
}

@st.cache_resource
def _init_orchestrator_logger():
    """Sets up the Orchestrator's own file logger exactly once per server process
    (guarded by st.cache_resource, since Streamlit reruns this whole script on every
    interaction -- without the guard, handlers would be re-added and log lines would
    be duplicated on every rerun)."""
    _logger = logging.getLogger("OrchestratorLogger")
    _logger.setLevel(logging.INFO)
    if not _logger.handlers:
        fh = logging.FileHandler(ORCHESTRATOR_LOG_FILE, mode='a', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s | ORCHESTRATOR | %(levelname)s | %(message)s'))
        _logger.addHandler(fh)
    return _logger

orchestrator_logger = _init_orchestrator_logger()

def log_error(source, message):
    """Central error sink for the in-app System Console (System Admin > Console tab).
    Any failure that used to be swallowed by a bare `except: pass` now lands here
    instead, so it's visible on the UI rather than only in a terminal the user can't
    see -- and it's also written to EO_Orchestrator_Agent.log for the same reason
    "I don't see the orchestrator logs anywhere" was raised."""
    try:
        if "error_log" not in st.session_state:
            st.session_state.error_log = []
        st.session_state.error_log.append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "source": source,
            "message": str(message),
        })
        # Cap so a long session can't grow this unboundedly.
        st.session_state.error_log = st.session_state.error_log[-200:]
    except Exception:
        pass  # The console itself must never be able to crash the app.
    orchestrator_logger.error(f"[{source}] {message}")

def log_info(message):
    """Informational counterpart to log_error() -- writes normal progress lines (not
    errors) to EO_Orchestrator_Agent.log, so the Orchestrator's own log has the same
    kind of step-by-step trace the Recommender/Explainer/ProductReviews agents have."""
    orchestrator_logger.info(message)

def read_agent_log_tail(agent_key, max_lines=200):
    """Reads the last `max_lines` lines of an agent's log file, for display in the
    System Console (System Admin > Console tab)."""
    path = AGENT_LOG_FILES.get(agent_key)
    if not path or not os.path.exists(path):
        return "_No log output yet._"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:]) or "_Log file is empty._"
    except Exception as e:
        return f"_Could not read log file: {e}_"

# ================= CONTENT SAFETY FILTER (2026-07-23 addition) =================
# Keeps explicit/adult, sexual, racial, or gender-based slur terms out of what
# users see, in two different ways depending on where the item came from:
#
#   1. "Rate Random Samples" (fetch_product_reviews_samples): a flagged item is
#      REPLACED with a different, safer item from the same query -- see its
#      docstring. Nothing here changes the SPARQL query logic itself.
#   2. Recommendation / Explanation output: the Recommender/Explainer algorithm
#      logic is untouched. sanitize_item_title() is applied only at the display
#      layer, inside render_item_card_html() -- the single place every item name
#      is rendered (samples, search matches, recommendation gallery, explanation
#      gallery) -- so flagged terms are stripped from the title text shown to the
#      user without changing which items were recommended or how/why.
#
# Uses the community-maintained `better-profanity` word list (pip install
# better-profanity) as the blocklist rather than hardcoding slur/explicit terms
# directly in this file. Add any extra site-specific terms to always flag via
# CONTENT_FILTER_EXTRA_BLOCKED_TERMS below.
#
# Fails safe: if the dependency isn't installed, filtering is transparently
# disabled (original names pass through unchanged, nothing is ever swapped out)
# rather than crashing the app -- a missing optional dependency should never
# break the existing demo flow.
CONTENT_FILTER_EXTRA_BLOCKED_TERMS = [
    # Add any additional site-specific terms to always flag here, e.g. "term1", "term2"
]

try:
    from better_profanity import profanity as _content_profanity_filter
    _content_profanity_filter.load_censor_words()
    if CONTENT_FILTER_EXTRA_BLOCKED_TERMS:
        _content_profanity_filter.add_censor_words(CONTENT_FILTER_EXTRA_BLOCKED_TERMS)
    CONTENT_FILTER_AVAILABLE = True
except Exception as _content_filter_import_err:
    _content_profanity_filter = None
    CONTENT_FILTER_AVAILABLE = False
    print(
        "[content-safety-filter] 'better-profanity' not installed -- content "
        f"filtering is DISABLED until `pip install better-profanity` is run: {_content_filter_import_err}"
    )

def is_item_name_unsafe(name):
    """Returns True if `name` contains explicit/adult, sexual, racial, or gender-
    based slur terms per the content filter.

    Returns False (never blocks anything) if the filter dependency isn't
    installed -- see CONTENT_FILTER_AVAILABLE -- or if the check itself errors,
    so a filter problem can never take down the sample-fetching flow.
    """
    if not CONTENT_FILTER_AVAILABLE or not name:
        return False
    try:
        return _content_profanity_filter.contains_profanity(str(name))
    except Exception as e:
        log_error("Content Safety Filter", f"contains_profanity check failed: {e}")
        return False

def sanitize_item_title(name):
    """Removes flagged explicit/adult, sexual, racial, or gender-based terms from
    an item's DISPLAY title before it's shown to the user.

    This only cleans the text that gets rendered -- it never changes which item
    was recommended/selected, and it never touches the Recommender or Explainer
    algorithm logic. Called from render_item_card_html(), the single choke point
    every item name is rendered through, so this automatically covers samples,
    search matches, the recommendation gallery, and the explanation gallery.

    Fails safe: returns `name` unchanged if the filter is unavailable, if nothing
    is flagged, or if sanitization itself errors for any reason.
    """
    if not name:
        return name
    text = str(name)
    if not CONTENT_FILTER_AVAILABLE:
        return text
    try:
        if not _content_profanity_filter.contains_profanity(text):
            return text
        # censor_char="" removes the flagged word instead of masking it with
        # asterisks, per the requirement to REMOVE these terms from titles.
        cleaned = _content_profanity_filter.censor(text, censor_char="")
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_,.")
        return cleaned if cleaned else "Item"
    except Exception as e:
        log_error("Content Safety Filter", f"Title sanitization failed, showing original title: {e}")
        return text

def sanitize_recommendation_results(results):
    """Applies sanitize_item_title() to every item's display name in a
    Recommender result list, IN PLACE, right after image enrichment and BEFORE
    the results are used for anything else (the formatter LLM prompt, the
    stored session_state.recommendation_result, the image gallery, the
    Explainer Agent payload, etc.).

    CONTENT SAFETY (2026-07-23 fix): previously only the image gallery caption
    (render_item_card_html -> sanitize_item_title) was cleaned, while the
    formatter LLM was separately handed the raw, unsanitized JSON and told to
    "list the recommended items by name" in its own freeform narrative -- so a
    flagged item's raw name still reached the user in that narrative text even
    though the same item's gallery caption right below it was already clean
    (exactly the "anal plug" bypass reported from a live screenshot). Cleaning
    the name here, once, at the source, means the LLM is never even shown the
    raw flagged text in the first place -- nothing about which items were
    recommended changes, only the text label attached to each.
    """
    if not isinstance(results, list):
        return results
    for item in results:
        if not isinstance(item, dict):
            continue
        for key in ("name", "item", "title"):
            if item.get(key):
                item[key] = sanitize_item_title(item[key])
    return results

def sanitize_narrative_text(text):
    """Same protection as sanitize_item_title(), but safe to run over a whole
    multi-line/multi-paragraph narrative (the formatter's recommendation
    summary, or the Explainer's explanation text) instead of a single short
    title. Preserves newlines/paragraph structure -- only collapses extra
    spaces left behind on the same line after a flagged word is removed.

    Defense-in-depth alongside sanitize_recommendation_results(): even though
    the formatter/explainer LLMs should no longer be shown raw flagged names
    (see sanitize_recommendation_results), this catches anything explicit/
    adult, sexual, racial, or gender-based slur-related an LLM might still
    introduce on its own in freeform prose before it reaches the chat.

    Fails safe: returns `text` unchanged if the filter is unavailable, if
    nothing is flagged, or if sanitization itself errors for any reason.
    """
    if not text:
        return text
    text = str(text)
    if not CONTENT_FILTER_AVAILABLE:
        return text
    try:
        if not _content_profanity_filter.contains_profanity(text):
            return text
        cleaned_lines = []
        for line in text.split("\n"):
            if _content_profanity_filter.contains_profanity(line):
                line = _content_profanity_filter.censor(line, censor_char="")
                line = re.sub(r"[ \t]+", " ", line).strip()
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)
    except Exception as e:
        log_error("Content Safety Filter", f"Narrative sanitization failed, showing original text: {e}")
        return text

# ================= AUTO-START AGENTS =================
@st.cache_resource
def start_background_agents():
    """
    Starts the Recommender, Explainer, and ProductReviews Flask apps in the background,
    redirecting each one's stdout/stderr into its own log file (AGENT_LOG_FILES) so the
    System Console can tail it -- rather than that output only reaching a terminal the
    Streamlit user never sees.
    """
    print("🚀 Booting up background Agent APIs...")
    
    # Check if scripts exist before trying to run them
    scripts = ["EO_Recommender_Agent.py", "EO_Explainer_Agent.py", "EO_ProductReviews_Agent.py"]
    for script in scripts:
        if not os.path.exists(script):
            print(f"⚠️ Warning: {script} not found. Please ensure it exists in the current directory.")

    log_handles = []

    def _log_handle_for(agent_key):
        fh = open(AGENT_LOG_FILES[agent_key], "a", encoding="utf-8", errors="replace")
        log_handles.append(fh)
        return fh

    rec_proc = subprocess.Popen(
        [sys.executable, "EO_Recommender_Agent.py"],
        stdout=_log_handle_for("Recommender"), stderr=subprocess.STDOUT,
    )
    exp_proc = subprocess.Popen(
        [sys.executable, "EO_Explainer_Agent.py"],
        stdout=_log_handle_for("Explainer"), stderr=subprocess.STDOUT,
    )
    prod_proc = subprocess.Popen(
        [sys.executable, "EO_ProductReviews_Agent.py"],
        stdout=_log_handle_for("ProductReviews"), stderr=subprocess.STDOUT,
    )
    
    def cleanup():
        print("🛑 Shutting down background Agent APIs...")
        rec_proc.terminate()
        exp_proc.terminate()
        prod_proc.terminate()
        for fh in log_handles:
            try:
                fh.close()
            except Exception:
                pass
        
    atexit.register(cleanup)
    return True

# Trigger the background agents
start_background_agents()

# ================= PAGE SETUP & API KEY =================
st.set_page_config(page_title="Recommender Orchestrator", layout="wide")

# Default Fuseki Base URL, editable from Admin > Configuration (moved out of the
# sidebar per 2026-07-08 feedback -- nothing about it belongs in "API Configuration").
if "fuseki_base" not in st.session_state:
    st.session_state.fuseki_base = "https://linked.aub.edu.lb:8080/fuseki"

with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API Key to initialize.")
    selected_model = st.selectbox(
        "Gemini Model", 
        ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro"], 
        index=0, 
        help="Select the Gemini model to use for orchestration."
    )

if not api_key:
    st.title("KGenXAI Demo: Explainable Product Recommendations")
    st.subheader("🤖 Orchestrator AI Agent")
    st.warning("👋 Welcome! Please enter your Gemini API Key in the sidebar to initialize the platform.")
    st.stop()

# Set Key and Model for Orchestrator LLM calls
genai.configure(api_key=api_key)
st.session_state.api_key = api_key
st.session_state.selected_model = selected_model

# ================= CONFIGURATION & DATABASES =================
# FUSEKI_BASE is user-selectable (Admin > Configuration tab, defaults to the project's
# existing server). Streamlit reruns this whole script top-to-bottom on every
# interaction, so simply reading it from session_state here keeps every endpoint below
# in sync with whatever the user has chosen, with no extra plumbing needed.
FUSEKI_BASE = st.session_state.fuseki_base
# Namespaces. Every term below was resolved against its published ontology --
# see agents/kgenxai_config.py, which is the single source of truth, and
# scripts/validate_vocabulary.py, which enforces it against the live endpoints.
from kgenxai_config import (
    AUTH as CONFIG_AUTH, NS, NS_INSTANCE,
    MLS_TERMS, SCHEMA_TERMS, EX_TERMS,
    DEFAULT_FUSEKI_BASE as CONFIG_FUSEKI_BASE,
    DEFAULT_EXECUTIONS_DATASET, explanation_class_for, custom_explanation_uri,
    sparql_prefixes, endpoint as build_endpoint,
)

# Was the literal ("username", "aubfuseki"). Reading it from the environment
# is what makes this file publishable; the password should also be rotated,
# since the endpoints are open and it has been in the source.
FUSEKI_AUTH = CONFIG_AUTH 

# ================= PER-DATASET NAME OVERRIDES (2026-07-08 addition) =================
# Per 2026-07-08 feedback: the three datasets all live on the SAME Fuseki server
# (FUSEKI_BASE, above) -- what should be individually overridable is just each
# dataset's NAME on that server (e.g. if someone's Fuseki instance calls it something
# other than "XAI_ProductReviews"), not a whole separate base URL per dataset. Defaults
# match the project's existing dataset names, so nothing changes unless the user edits
# these in Admin > Configuration.
if "dataset_names" not in st.session_state:
    st.session_state.dataset_names = {
        "product_reviews": "XAI_ProductReviews",
        "algorithms": "XAI_RecommendationAlgorithms",
        "explanations": "XAI_InteractiveExplanations",
    }

def _dataset_name(key, default):
    """Effective dataset name for `key` (Admin > Configuration tab override, or the
    project's default if blank/unset)."""
    name = (st.session_state.dataset_names.get(key) or "").strip()
    return name or default

PRODUCT_REVIEWS_DATASET_NAME = _dataset_name("product_reviews", "XAI_ProductReviews")
ALGORITHMS_DATASET_NAME = _dataset_name("algorithms", "XAI_RecommendationAlgorithms")
EXPLANATIONS_DATASET_NAME = _dataset_name("explanations", "XAI_InteractiveExplanations")
# Execution records live in their own dataset. Runtime instances grow without
# bound, and keeping them out of the definitions graph makes it possible to
# inspect the procedures without wading through runs.
# Overridable from Admin > Configuration exactly like the other three.
EXECUTIONS_DATASET_NAME = _dataset_name("executions", "XAI_ExecutionLogs")

def get_product_reviews_fuseki_base():
    """Fuseki base URL sent to the ProductReviews Agent -- and, transitively, to the
    Recommender/Explainer agents, which forward it on for their own XAI_ProductReviews
    access. All three datasets share one Fuseki server, so this is just FUSEKI_BASE."""
    return FUSEKI_BASE

def get_product_reviews_dataset_name():
    """The (possibly renamed, via Admin > Configuration) dataset name for
    XAI_ProductReviews, sent alongside the Fuseki base to the ProductReviews Agent so
    it builds the right /query, /update, /data endpoints."""
    return PRODUCT_REVIEWS_DATASET_NAME

# NOTE: XAI_ProductReviews (the product/review dataset) is intentionally NOT given a
# direct Fuseki endpoint constant here. EO_ProductReviews_Agent.py is the sole owner
# of that dataset -- every read or write goes through its HTTP API below, never
# straight to Fuseki from this file. See PRODUCT_REVIEWS_*_API constants. Since that
# agent runs in a separate process, the Fuseki base AND dataset name are both passed to
# it on each request instead.

RECOMMENDATION_ALGORITHMS_ENDPOINT = f"{FUSEKI_BASE}/{ALGORITHMS_DATASET_NAME}/query"
RECOMMENDATION_ALGORITHMS_UPDATE = f"{FUSEKI_BASE}/{ALGORITHMS_DATASET_NAME}/update"

INTERACTIVE_EXPLANATIONS_ENDPOINT = f"{FUSEKI_BASE}/{EXPLANATIONS_DATASET_NAME}/query"
INTERACTIVE_EXPLANATIONS_UPDATE = f"{FUSEKI_BASE}/{EXPLANATIONS_DATASET_NAME}/update"

# Microservice Agent Endpoints
RECOMMENDER_API = "http://127.0.0.1:5003/api/autonomous-recommend"
EXPLAINER_API = "http://127.0.0.1:5001/api/explain" 
PRODUCT_REVIEWS_API = "http://127.0.0.1:5002/api/execute_safe_sparql"
PRODUCT_REVIEWS_UPLOAD_API = "http://127.0.0.1:5002/api/upload_graph"
PRODUCT_REVIEWS_CLEAR_API = "http://127.0.0.1:5002/api/clear_graph"

EX = Namespace(NS_INSTANCE["amazon"])
SCHEMA = Namespace(NS["schema"])
EO = Namespace(NS["eo"])
SIO = Namespace(NS["sio"])
EP = Namespace(NS["ep"])
PKO = Namespace(NS["pko"])
PPLAN = Namespace(NS["pplan"])
MLS = Namespace(NS["mls"])
PROV = Namespace(NS["prov"])
EXAI = Namespace(NS["ex"])
# The wd: namespace was declared here and never used anywhere in the codebase
# (defect 16). Removed: an unused namespace is one more thing an inspector has
# to rule out.

# SIO terms as rdflib references. SIO defines only six non-numeric URIs, so
# sio:hasDataItem, sio:Dataset, sio:SIO_000300 and sio:SIO_000008 -- all
# camel-cased rdfs:labels -- do not exist (defects 9-13). hasDataItem alone was
# on 9,998 triples in the deployed graph.
# SIO is no longer written. Each replacement is declared in
# ontology/kgenxai.ttl as a subclass or subproperty of the SIO term it stands
# in for, so alignment with the Explanation Ontology survives as a reasoning
# step rather than an unreadable URI in the data.
MLS_HAS_PART = URIRef(MLS_TERMS["has_part"])       # was SIO_001277
MLS_DATASET = URIRef(MLS_TERMS["dataset"])         # was SIO_000089
MLS_HAS_VALUE = URIRef(MLS_TERMS["has_value"])     # was SIO_000300

# ================= STATE INITIALIZATION =================
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "user_id": f"User_{uuid.uuid4().hex[:8]}",
        "history": [],
        "preferences": {"preferred_categories": [], "avoid_categories": []}
    }
if "step" not in st.session_state:
    st.session_state.step = "choose_method"
if "recommendation_result" not in st.session_state:
    st.session_state.recommendation_result = None
if "current_rec_uri" not in st.session_state:
    st.session_state.current_rec_uri = f"http://linked.aub.edu.lb/kgenxai/amazon/recommendation/{uuid.uuid4().hex[:8]}"
# WP3/WP4: identifies the recorded execution behind the current recommendation,
# so an explanation can be traced to the run that produced it rather than to
# whichever log file happened to carry the algorithm's name.
if "current_execution_id" not in st.session_state:
    st.session_state.current_execution_id = None
if "active_view" not in st.session_state:
    st.session_state.active_view = "chat"  # "chat" | "settings" | "console"
if "ingestion_status" not in st.session_state:
    st.session_state.ingestion_status = None
if "generated_custom_script" not in st.session_state:
    st.session_state.generated_custom_script = None
if "error_log" not in st.session_state:
    st.session_state.error_log = []
if "known_item_uris" not in st.session_state:
    # Maps a product display name -> its resolved KG URI, for every item ever shown
    # to the user via samples or search. Lets later free-text chat turns ("I like
    # item 1 and 3") attach a real URI to a history entry instead of a name only.
    st.session_state.known_item_uris = {}
if "shown_sample_uris" not in st.session_state:
    # Every item URI ever shown to the user via a random sample batch, across every
    # "show me more items" round this session -- passed to
    # fetch_product_reviews_samples() as an exclusion list so a fresh batch never
    # repeats something already shown (previously there was no exclusion at all, so
    # a small random OFFSET window could easily resurface the same items).
    st.session_state.shown_sample_uris = set()
if "last_search_terms" not in st.session_state:
    # Remembers the topic the user last searched for (e.g. "sports") so a follow-up
    # "give me other options" re-searches THAT topic (excluding what's already shown)
    # instead of losing the topic entirely and falling back to unrelated random
    # samples -- previously every "show me more" request routed to the same
    # blind-random-sample path regardless of whether the user had been browsing a
    # specific topic.
    st.session_state.last_search_terms = None

# ================= AMAZON 2023 DATASET CONFIGURATION (UCSD REPO) =================
# To avoid destabilizing live demos with multi-GB downloads of uncertain duration,
# only datasets under PROOF_OF_CONCEPT_SIZE_LIMIT_MB are enabled for one-click
# "Download and Ingest". Larger datasets remain visible (so the catalog/roadmap is
# clear) and previewable via "View Sample Data", but their ingest button is disabled
# and labeled experimental until they've been separately validated.
PROOF_OF_CONCEPT_SIZE_LIMIT_MB = 200

# Hard cap for the "Upload Custom Dataset" feature (separate axis from the Amazon
# dataset gate above). Enforced in the UI before any parsing or LLM script generation
# is attempted: large uploads risk the tool crashing or stalling on memory during
# in-browser/in-process parsing.
MAX_CUSTOM_UPLOAD_MB = 200

# Custom-dataset ingestion (the "AI writes a script for your file" path) is still being
# field-tested across file formats, so it is marked experimental and gated behind an
# explicit opt-in toggle rather than being available unconditionally in demos.
CUSTOM_INGESTION_ENABLED_BY_DEFAULT = False

# ================= DEMO MODE =================
# End users should be able to SEE that the platform/settings are configurable (the
# tabs, forms, and buttons all still render normally) but must not actually be able
# to change anything that could break the shared server for other users -- Fuseki
# base URL, dataset names, agent system prompts, algorithm/explanation-type
# definitions, or destructively clear the graph. Flip this to False for real
# development work; nothing else in the app depends on it.
DEMO_MODE = True
DEMO_MODE_MESSAGE = "This feature is disabled for this demo version."

AMAZON_DATASETS = {
    "Sports and Outdoors": {
        "url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/review_categories/Sports_and_Outdoors.jsonl.gz",
        "size": "~2.8GB (Compressed .gz)",
        "size_mb": 2867,
        "num_reviews": "~3.7M",
        "meta_url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/meta_categories/meta_Sports_and_Outdoors.jsonl.gz",
        "meta_size": "~185MB"
    },
    "Electronics": {
        "url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz",
        "size": "~3.5GB (Compressed .gz)",
        "size_mb": 3584,
        "num_reviews": "~4.5M",
        "meta_url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz",
        "meta_size": "~220MB"
    },
    "Clothing, Shoes and Jewelry": {
        "url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl.gz",
        "size": "~2.1GB (Compressed .gz)",
        "size_mb": 2150,
        "num_reviews": "~2.8M",
        "meta_url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl.gz",
        "meta_size": "~150MB"
    },
    "Home and Kitchen": {
        "url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/review_categories/Home_and_Kitchen.jsonl.gz",
        "size": "~3.2GB (Compressed .gz)",
        "size_mb": 3277,
        "num_reviews": "~4.2M",
        "meta_url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/meta_categories/meta_Home_and_Kitchen.jsonl.gz",
        "meta_size": "~200MB"
    },
    "Health and Household": {
        "url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/review_categories/Health_and_Household.jsonl.gz",
        "size": "~1.8GB (Compressed .gz)",
        "size_mb": 1843,
        "num_reviews": "~2.4M",
        "meta_url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/meta_categories/meta_Health_and_Household.jsonl.gz",
        "meta_size": "~120MB"
    },
    "Beauty and Personal Care": {
        "url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/review_categories/Beauty_and_Personal_Care.jsonl.gz",
        "size": "~1.2GB (Compressed .gz)",
        "size_mb": 1229,
        "num_reviews": "~1.6M",
        "meta_url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl.gz",
        "meta_size": "~80MB"
    },
    "Movies and TV": {
        "url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/review_categories/Movies_and_TV.jsonl.gz",
        "size": "~2.5GB (Compressed .gz)",
        "size_mb": 2560,
        "num_reviews": "~3.3M",
        "meta_url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/meta_categories/meta_Movies_and_TV.jsonl.gz",
        "meta_size": "~160MB"
    },
    "Books": {
        "url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/review_categories/Books.jsonl.gz",
        "size": "~4.0GB (Compressed .gz)",
        "size_mb": 4096,
        "num_reviews": "~5.3M",
        "meta_url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz",
        "meta_size": "~250MB"
    },
    "Gift Cards": {
        "url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz",
        "size": "~150MB (Compressed .gz)",
        "size_mb": 150,
        "num_reviews": "~200K",
        "meta_url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/meta_categories/meta_Gift_Cards.jsonl.gz",
        "meta_size": "~10MB"
    },
    "Digital Music": {
        "url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/review_categories/Digital_Music.jsonl.gz",
        "size": "~80MB (Compressed .gz)",
        "size_mb": 80,
        "num_reviews": "~100K",
        "meta_url": "https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/meta_categories/meta_Digital_Music.jsonl.gz",
        "meta_size": "~5MB"
    }
}

# Derive the proof-of-concept flag for each dataset from its declared size, rather than
# hardcoding which names are "safe" -- so adding a new dataset to the dict above
# automatically gets gated correctly without an extra manual step.
for _name, _info in AMAZON_DATASETS.items():
    _info["proof_of_concept"] = _info.get("size_mb", float("inf")) <= PROOF_OF_CONCEPT_SIZE_LIMIT_MB

# ================= ADMIN / DATA CONFIGURATION FUNCTIONS =================
def run_sparql_query(endpoint, query):
    try:
        res = requests.post(endpoint, data={'query': query}, auth=FUSEKI_AUTH, timeout=60)
        if res.status_code == 200:
            return res.json().get('results', {}).get('bindings', [])
        else:
            st.error(f"SPARQL Query Error (HTTP {res.status_code}): {res.text[:200]}")
            return []
    except requests.exceptions.Timeout:
        st.error("SPARQL query timed out. The endpoint may be slow or unreachable.")
        return []
    except Exception as e:
        st.error(f"SPARQL Query Error: {e}")
        return []

def run_sparql_update(endpoint, update_query):
    try:
        res = requests.post(endpoint, data={'update': update_query}, auth=FUSEKI_AUTH, timeout=120)
        if res.status_code in [200, 201, 204]:
            return True
        else:
            st.error(f"Fuseki Update Failed (HTTP {res.status_code}): {res.text[:200]}")
            return False
    except requests.exceptions.Timeout:
        st.error("SPARQL update timed out. The endpoint may be slow or unreachable.")
        return False
    except Exception as e:
        st.error(f"SPARQL Request Exception: {e}")
        return False

# ---------------------------------------------------------------------------
# XAI_ProductReviews access -- routed exclusively through EO_ProductReviews_Agent.py.
# This file holds no Fuseki endpoint URL or credential for that dataset; every read
# or write below is a plain HTTP call to the agent, which is the only thing that
# ever talks to Fuseki for XAI_ProductReviews.
# ---------------------------------------------------------------------------

def query_product_reviews_via_agent(query):
    """
    Read-only SPARQL SELECT against XAI_ProductReviews, via the ProductReviews
    Agent's /api/execute_safe_sparql. Mirrors run_sparql_query()'s return shape
    (a list of binding dicts, or [] on failure) so callers don't need to change.
    """
    try:
        res = requests.post(
            PRODUCT_REVIEWS_API,
            json={'query': query, 'fuseki_base': get_product_reviews_fuseki_base(), 'product_reviews_dataset': get_product_reviews_dataset_name()},
            timeout=30,
        )
        if res.status_code == 200:
            return res.json().get('results', {}).get('bindings', [])
        else:
            st.error(f"ProductReviews Agent query error (HTTP {res.status_code}): {res.text[:200]}")
            return []
    except requests.exceptions.Timeout:
        st.error("ProductReviews Agent query timed out. The agent may be slow or unreachable.")
        return []
    except Exception as e:
        st.error(f"ProductReviews Agent query error: {e}")
        return []

def upload_graph_to_product_reviews_via_agent(graph):
    """
    Uploads an rdflib Graph to XAI_ProductReviews via the ProductReviews Agent's
    /api/upload_graph, instead of POSTing Turtle straight to Fuseki. Mirrors
    upload_graph_to_fuseki()'s behavior (including the no-op-on-empty-graph case).
    """
    if len(graph) == 0:
        return True
    try:
        turtle_data = graph.serialize(format='turtle').encode('utf-8')
        res = requests.post(
            PRODUCT_REVIEWS_UPLOAD_API,
            data=turtle_data,
            headers={'Content-Type': 'text/turtle; charset=utf-8'},
            params={'fuseki_base': get_product_reviews_fuseki_base(), 'product_reviews_dataset': get_product_reviews_dataset_name()},
            timeout=120,
        )
        if res.status_code == 200:
            return True
        else:
            st.error(f"ProductReviews Agent upload failed (HTTP {res.status_code}): {res.text[:200]}")
            return False
    except requests.exceptions.Timeout:
        st.error("Graph upload via ProductReviews Agent timed out.")
        return False
    except Exception as e:
        st.error(f"Graph upload via ProductReviews Agent error: {e}")
        return False

def clear_product_reviews_via_agent():
    """
    Deletes ALL triples from XAI_ProductReviews via the ProductReviews Agent's
    /api/clear_graph (which does the batched delete loop server-side, rather than
    the Orchestrator looping over individual Fuseki update calls itself).
    Returns (success, info_dict_or_error_message).
    """
    try:
        res = requests.post(
            PRODUCT_REVIEWS_CLEAR_API,
            params={'fuseki_base': get_product_reviews_fuseki_base(), 'product_reviews_dataset': get_product_reviews_dataset_name()},
            timeout=300,
        )
        if res.status_code == 200:
            return True, res.json()
        else:
            return False, res.text[:300]
    except requests.exceptions.Timeout:
        return False, "Clear graph request timed out."
    except Exception as e:
        return False, str(e)

def upload_graph_to_fuseki(graph, endpoint):
    if len(graph) == 0:
        return True
    try:
        # Serialize to turtle
        turtle_data = graph.serialize(format='turtle').encode('utf-8')
        res = requests.post(
            endpoint, 
            data=turtle_data, 
            headers={'Content-Type': 'text/turtle; charset=utf-8'}, 
            auth=FUSEKI_AUTH,
            timeout=120
        )
        if res.status_code in [200, 201, 204]:
            return True
        else:
            st.error(f"Graph Upload Error (HTTP {res.status_code}): {res.text[:200]}")
            return False
    except requests.exceptions.Timeout:
        st.error("Graph upload timed out. The endpoint may be slow or unreachable.")
        return False
    except Exception as e:
        st.error(f"Graph Upload Exception: {e}")
        return False

def get_product_reviews_triple_count():
    """Get the current number of triples in the XAI_ProductReviews graph."""
    query = "SELECT (COUNT(?s) AS ?count) WHERE { ?s ?p ?o }"
    result = query_product_reviews_via_agent(query)
    if result and 'count' in result[0]:
        return int(result[0]['count']['value'])
    return 0

def get_product_reviews_sample_triples(limit=20):
    """Get a sample of triples from the XAI_ProductReviews graph for preview."""
    query = f"SELECT ?s ?p ?o WHERE {{ ?s ?p ?o }} LIMIT {limit}"
    return query_product_reviews_via_agent(query)

class ExecutionTimeout(Exception):
    """Raised when a guarded block of code exceeds its allotted wall-clock time."""
    pass


@contextlib.contextmanager
def time_limit(seconds):
    """
    Aborts the wrapped block if it runs longer than `seconds`.

    Used around exec() of LLM-generated ingestion scripts: a buggy or pathological
    generated script (e.g. an infinite loop, or a parser stuck on malformed input)
    would otherwise hang the entire Streamlit process with no way to recover short
    of restarting the server.

    Implementation note: signal.alarm is Unix-only and only works on the main
    thread. On platforms/contexts where it's unavailable, this degrades to a
    no-op (the block still runs, just without an enforced timeout) rather than
    raising, since the app must still function in those environments.
    """
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise ExecutionTimeout(f"Execution exceeded {seconds}s time limit.")

    previous_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def download_file(url, description):
    """Download a file with progress tracking."""
    try:
        # Use verify=False to bypass possible legacy or academic university SSL certificate issues
        response = requests.get(url, stream=True, timeout=30, verify=False)
        response.raise_for_status()
        
        # Get file size for progress
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB chunks
        
        # Determine temporary file suffix based on format
        file_suffix = '.jsonl.gz' if url.endswith('.gz') else '.jsonl'
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix)
        
        downloaded = 0
        with st.spinner(f"Downloading {description}..."):
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    temp_file.write(chunk)
                    downloaded += len(chunk)
                    # Update progress every 10MB
                    if downloaded % (10 * 1024 * 1024) < block_size:
                        progress = (downloaded / total_size * 100) if total_size > 0 else 0
                        st.caption(f"Downloaded: {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({progress:.1f}%)")
        
        temp_file.close()
        return temp_file.name
    except requests.exceptions.Timeout:
        st.error(f"Download timed out for {description}. The file may be too large or the server may be slow.")
        return None
    except Exception as e:
        st.error(f"Download failed for {description}: {e}")
        return None

def open_file_helper(filepath):
    """Dynamically open compressed (.gz) or plain text files."""
    if filepath.endswith('.gz'):
        return gzip.open(filepath, 'rt', encoding='utf-8')
    return open(filepath, 'r', encoding='utf-8')

def parse_amazon_meta_line(data):
    """Parse a single line of Amazon metadata JSONL and return RDF triples."""
    triples = []
    asin = data.get('parent_asin') or data.get('asin')
    if not asin:
        return []
    
    product_uri = EX[f"product/{asin}"]
    triples.append((product_uri, RDF.type, SCHEMA.Product))
    triples.append((product_uri, RDF.type, EO.object_record))
    
    title = data.get('title')
    if title:
        triples.append((product_uri, SCHEMA.name, Literal(title, datatype=XSD.string)))
    
    main_cat = data.get('main_category')
    if main_cat:
        triples.append((product_uri, SCHEMA.category, Literal(main_cat, datatype=XSD.string)))
    
    categories = data.get('categories', [])
    if isinstance(categories, list):
        for cat in categories:
            if cat:
                triples.append((product_uri, SCHEMA.category, Literal(str(cat), datatype=XSD.string)))
    
    avg_rating = data.get('average_rating')
    if avg_rating is not None:
        try:
            triples.append((product_uri, SCHEMA.aggregateRating, Literal(float(avg_rating), datatype=XSD.float)))
        except:
            pass
    
    store = data.get('store')
    if store:
        triples.append((product_uri, SCHEMA.brand, Literal(str(store), datatype=XSD.string)))
    
    # Image/Thumbnail Support
    images = data.get('images', [])
    if isinstance(images, list) and len(images) > 0:
        img_info = images[0]
        if isinstance(img_info, dict):
            img_val = img_info.get('hi_res') or img_info.get('large') or img_info.get('thumb')
            if img_val:
                img_url = img_val[0] if isinstance(img_val, list) else img_val
                if img_url and isinstance(img_url, str) and img_url.startswith('http'):
                    triples.append((product_uri, SCHEMA.image, URIRef(img_url)))
    
    return triples

def parse_amazon_review_line(data, dataset_uri):
    """Parse a single line of Amazon review JSONL and return RDF triples."""
    triples = []
    asin = data.get('parent_asin') or data.get('asin')
    user_id = data.get('user_id')
    
    if not asin or not user_id:
        return []
    
    review_uuid = str(uuid.uuid4())
    review_uri = EX[f"review/{review_uuid}"]
    product_uri = EX[f"product/{asin}"]
    user_uri = EX[f"user/{user_id}"]
    
    triples.append((review_uri, RDF.type, SCHEMA.Review))
    triples.append((review_uri, RDF.type, EO.object_record))
    triples.append((review_uri, SCHEMA.itemReviewed, product_uri))
    triples.append((dataset_uri, MLS_HAS_PART, review_uri))
    
    title = data.get('title')
    if title:
        triples.append((review_uri, SCHEMA.headline, Literal(str(title), datatype=XSD.string)))
    
    text = data.get('text')
    if text:
        triples.append((review_uri, SCHEMA.reviewBody, Literal(str(text), datatype=XSD.string)))
    
    rating = data.get('rating')
    if rating is not None:
        try:
            rating_node = EX[f"rating/{review_uuid}"]
            triples.append((review_uri, SCHEMA.reviewRating, rating_node))
            triples.append((rating_node, RDF.type, SCHEMA.Rating))
            triples.append((rating_node, SCHEMA.ratingValue, Literal(float(rating), datatype=XSD.float)))
        except:
            pass
    
    triples.append((review_uri, SCHEMA.author, user_uri))
    triples.append((user_uri, RDF.type, SCHEMA.Person))
    triples.append((user_uri, RDF.type, EO.user))
    triples.append((user_uri, RDF.type, EO.object_record))
    triples.append((user_uri, SCHEMA.identifier, Literal(str(user_id), datatype=XSD.string)))
    
    return triples

def ingest_amazon_dataset(dataset_name, dataset_info, progress_bar, status_text):
    """Download and ingest an Amazon 2023 dataset."""
    BATCH_SIZE = 1000
    DATASET_URI = EX["dataset/AmazonDomainData"]
    dataset_uri = DATASET_URI
    
    def create_bound_graph():
        g = Graph()
        g.bind("schema", SCHEMA)
        g.bind("ex", EX)
        g.bind("eo", EO)
        g.bind("sio", SIO)
        return g
    
    # Step 1: Download metadata
    status_text.text(f"Step 1/4: Downloading metadata for {dataset_name}...")
    meta_file = download_file(dataset_info["meta_url"], f"metadata for {dataset_name}")
    if not meta_file:
        return False, "Failed to download metadata file."
    
    # Step 2: Download reviews
    status_text.text(f"Step 2/4: Downloading reviews for {dataset_name}...")
    review_file = download_file(dataset_info["url"], f"reviews for {dataset_name}")
    if not review_file:
        os.unlink(meta_file)
        return False, "Failed to download reviews file."
    
    # Step 3: Process metadata
    status_text.text(f"Step 3/4: Processing metadata for {dataset_name}...")
    g = create_bound_graph()
    g.add((dataset_uri, RDF.type, MLS_DATASET))
    g.add((dataset_uri, RDF.type, EO.object_record))
    
    meta_count = 0
    try:
        with open_file_helper(meta_file) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    triples = parse_amazon_meta_line(data)
                    for s, p, o in triples:
                        g.add((s, p, o))
                    # Add dataset reference
                    asin = data.get('parent_asin') or data.get('asin')
                    if asin:
                        product_uri = EX[f"product/{asin}"]
                        g.add((dataset_uri, MLS_HAS_PART, product_uri))
                    meta_count += 1
                    
                    if meta_count % BATCH_SIZE == 0:
                        status_text.text(f"  Processed {meta_count} metadata items...")
                        if not upload_graph_to_product_reviews_via_agent(g):
                            os.unlink(meta_file)
                            os.unlink(review_file)
                            return False, f"Failed to upload metadata batch at {meta_count} items."
                        g = create_bound_graph()
                        g.add((dataset_uri, RDF.type, MLS_DATASET))
                        g.add((dataset_uri, RDF.type, EO.object_record))
                except json.JSONDecodeError as e:
                    st.warning(f"Skipped invalid JSON line in metadata: {e}")
                    continue
                
        # Upload remaining metadata
        if len(g) > 0:
            if not upload_graph_to_product_reviews_via_agent(g):
                os.unlink(meta_file)
                os.unlink(review_file)
                return False, "Failed to upload remaining metadata."
    
    except Exception as e:
        os.unlink(meta_file)
        os.unlink(review_file)
        return False, f"Error processing metadata: {e}"
    
    # Step 4: Process reviews
    status_text.text(f"Step 4/4: Processing reviews for {dataset_name}...")
    g = create_bound_graph()
    g.add((dataset_uri, RDF.type, MLS_DATASET))
    g.add((dataset_uri, RDF.type, EO.object_record))
    
    review_count = 0
    try:
        with open_file_helper(review_file) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    triples = parse_amazon_review_line(data, dataset_uri)
                    for s, p, o in triples:
                        g.add((s, p, o))
                    review_count += 1
                    
                    if review_count % BATCH_SIZE == 0:
                        status_text.text(f"  Processed {review_count} reviews...")
                        if not upload_graph_to_product_reviews_via_agent(g):
                            os.unlink(meta_file)
                            os.unlink(review_file)
                            return False, f"Failed to upload reviews batch at {review_count} items."
                        g = create_bound_graph()
                        g.add((dataset_uri, RDF.type, MLS_DATASET))
                        g.add((dataset_uri, RDF.type, EO.object_record))
                except json.JSONDecodeError as e:
                    st.warning(f"Skipped invalid JSON line in reviews: {e}")
                    continue
                
        # Upload remaining reviews
        if len(g) > 0:
            if not upload_graph_to_product_reviews_via_agent(g):
                os.unlink(meta_file)
                os.unlink(review_file)
                return False, "Failed to upload remaining reviews."
    
    except Exception as e:
        os.unlink(meta_file)
        os.unlink(review_file)
        return False, f"Error processing reviews: {e}"
    
    # Cleanup temp files
    try:
        os.unlink(meta_file)
        os.unlink(review_file)
    except:
        pass
    
    return True, f"✅ Successfully ingested {dataset_name} with {meta_count} products and {review_count} reviews."

def ingest_custom_dataset(uploaded_file, progress_bar, status_text):
    """Ingest a custom dataset uploaded by the user."""
    BATCH_SIZE = 1000
    DATASET_URI = EX[f"dataset/UserDataset_{uuid.uuid4().hex[:8]}"]

    # Defense in depth: enforce the size cap here too, not just in the calling UI code,
    # in case this function is ever invoked from another entry point.
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_CUSTOM_UPLOAD_MB:
        return False, (
            f"File is {file_size_mb:.1f}MB, which exceeds the {MAX_CUSTOM_UPLOAD_MB}MB "
            f"limit for custom uploads."
        )

    def create_bound_graph():
        g = Graph()
        g.bind("schema", SCHEMA)
        g.bind("ex", EX)
        g.bind("eo", EO)
        g.bind("sio", SIO)
        return g
    
    # Determine file type from extension
    file_ext = uploaded_file.name.split('.')[-1].lower()
    status_text.text(f"Processing {uploaded_file.name}...")
    
    # Generate custom ingestion script using LLM for non-JSONL formats
    if file_ext not in ['jsonl']:
        status_text.text("Generating custom ingestion script for your data format...")
        model = genai.GenerativeModel(
            st.session_state.get("selected_model", "gemini-3-flash-preview"),
            system_instruction=get_orchestrator_system_instruction("ingestion", INGESTION_TOOL_PROMPT_DEFAULT),
        )
        
        # Read a sample of the file to understand its structure
        sample_content = ""
        try:
            if file_ext in ['json']:
                content = uploaded_file.getvalue().decode('utf-8')
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        sample_content = json.dumps(data[:3], indent=2)
                    else:
                        sample_content = json.dumps(data, indent=2)[:2000]
                except:
                    sample_content = content[:2000]
            elif file_ext in ['csv', 'tsv']:
                sample_content = uploaded_file.getvalue().decode('utf-8')[:2000]
            elif file_ext in ['xlsx', 'xls']:
                try:
                    import openpyxl
                    from io import BytesIO
                    wb = openpyxl.load_workbook(BytesIO(uploaded_file.getvalue()))
                    sheet = wb.active
                    sample_rows = []
                    for i, row in enumerate(sheet.iter_rows(values=True)):
                        if i >= 5:
                            break
                        sample_rows.append(list(row))
                    sample_content = json.dumps(sample_rows, indent=2)
                except ImportError:
                    st.warning("openpyxl not installed. Please install it: pip install openpyxl")
                    return False, "openpyxl not installed. Please install it: pip install openpyxl"
                except Exception as e:
                    sample_content = f"Error reading Excel file: {e}"
            elif file_ext in ['pdf']:
                try:
                    import PyPDF2
                    from io import BytesIO
                    pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.getvalue()))
                    sample_text = ""
                    for i, page in enumerate(pdf_reader.pages[:3]):
                        sample_text += page.extract_text() or ""
                    sample_content = sample_text[:2000]
                except ImportError:
                    st.warning("PyPDF2 not installed. Please install it: pip install PyPDF2")
                    return False, "PyPDF2 not installed. Please install it: pip install PyPDF2"
                except Exception as e:
                    sample_content = f"Error reading PDF file: {e}"
            else:
                sample_content = uploaded_file.getvalue().decode('utf-8')[:2000]
        except Exception as e:
            sample_content = f"Could not read sample: {e}"
        
        # Generate custom ingestion script with highly descriptive prompt mirroring XAI_ProductReviews
        prompt = f"""
        Your task is to generate a robust, fully runnable Python script that reads a {file_ext.upper()} file and converts it into RDF triples.
        The generated RDF triples must perfectly mimic the data structures, ontologies, and class/property associations found in the AUB KGenXAI Amazon/ProductReviews dataset.

        ### Base URI & Vocabularies
        Base URI: http://linked.aub.edu.lb/kgenxai/
        - EX: Namespace("http://linked.aub.edu.lb/kgenxai/amazon/")
        - SCHEMA: Namespace("http://schema.org/")
        - EO: Namespace("https://purl.org/heals/eo#")
        - SIO: Namespace("http://semanticscience.org/resource/")
        - XSD: Namespace("http://www.w3.org/2001/XMLSchema#")

        ### Target Ontological Mapping Rules (Mimic XAI_ProductReviews Dataset Structure)
        Your script must map entities and generate triples using these exact patterns:
        
        1. **Dataset Level**:
           - Subject URI: dataset_uri (passed as a parameter)
           - Triples to generate:
             - (dataset_uri, RDF.type, MLS_DATASET)
             - (dataset_uri, RDF.type, EO.object_record)
             - (dataset_uri, MLS_HAS_PART, product_uri_or_review_uri) [Generate this for each product and review]

        2. **Product Level** (catalog/reviewed items):
           - Subject URI: EX[f"product/{{asin_or_id}}"] (Ensure values are converted to string and stripped of spaces/special characters)
           - Triples to generate:
             - (product_uri, RDF.type, SCHEMA.Product)
             - (product_uri, RDF.type, EO.object_record)
             - (product_uri, SCHEMA.name, Literal(title_str, datatype=XSD.string))
             - (product_uri, SCHEMA.category, Literal(category_str, datatype=XSD.string)) [Multiple category statements allowed]
             - (product_uri, SCHEMA.brand, Literal(brand_str, datatype=XSD.string))
             - (product_uri, SCHEMA.aggregateRating, Literal(float_rating, datatype=XSD.float))
             - (product_uri, SCHEMA.image, URIRef(image_url_str)) [Only if a valid URL starting with http]

        3. **Review Level** (ratings and user comments):
           - Subject URI: EX[f"review/{{review_id_or_uuid}}"] (If the source data doesn't have an ID, generate a unique ID, e.g. using uuid)
           - Triples to generate:
             - (review_uri, RDF.type, SCHEMA.Review)
             - (review_uri, RDF.type, EO.object_record)
             - (review_uri, SCHEMA.itemReviewed, product_uri)
             - (review_uri, SCHEMA.headline, Literal(headline_str, datatype=XSD.string))
             - (review_uri, SCHEMA.reviewBody, Literal(body_str, datatype=XSD.string))
             - (review_uri, SCHEMA.reviewRating, rating_uri)
             - (review_uri, SCHEMA.author, user_uri)

        4. **Rating Level** (score associated with a review):
           - Subject URI: EX[f"rating/{{review_id_or_uuid}}"]
           - Triples to generate:
             - (rating_uri, RDF.type, SCHEMA.Rating)
             - (rating_uri, SCHEMA.ratingValue, Literal(float_value, datatype=XSD.float))

        5. **User Level** (author of a review):
           - Subject URI: EX[f"user/{{user_id}}"]
           - Triples to generate:
             - (user_uri, RDF.type, SCHEMA.Person)
             - (user_uri, RDF.type, EO.user)
             - (user_uri, RDF.type, EO.object_record)
             - (user_uri, SCHEMA.identifier, Literal(user_id, datatype=XSD.string))

        ### Source Data Information ({file_ext.upper()})
        Analyze the column headers, keys, or contents in this sample of the uploaded file to extract products, categories, reviews, authors, or ratings dynamically:
        {sample_content[:3000]}

        ### Requirements of the Generated Python Script
        - It must define a function: `parse_data_to_triples(file_content_bytes, dataset_uri)`
        - It should accept `file_content_bytes` (bytes of the custom uploaded file).
        - It must parse this content dynamically. For CSV/TSV/Excel, use `pandas` (with `io.BytesIO`). For JSON, use `json.loads`. For PDF, use `PyPDF2`.
        - It must return a Python list of tuples `(subject_uri, predicate_uri, object_value)` where subject_uri and predicate_uri are `URIRef`s and object_value is either a `URIRef` or a `Literal`.
        - It must handle potential null or NaN values gracefully, ensuring `Literal` contains valid strings or numbers, and skipping empty/invalid values.
        - The classes/namespaces `URIRef`, `Literal`, `RDF`, `XSD`, `EX`, `SCHEMA`, `EO`, `SIO` are already pre-bound and injected in the global namespace of execution, but you can also define them locally.
        - Ensure all string literals are converted to safe representations.
        
        Return ONLY the raw executable Python code inside your response. Do not include markdown code block syntax (like ```python) or any conversational explanations.
        """
        
        response = model.generate_content(prompt)
        script_code = response.text.replace("```python", "").replace("```", "").strip()
        
        # Save custom ingestion script in streamlit session state for visibility on the UI
        st.session_state.generated_custom_script = script_code
        
        # Execute the generated script to parse the data
        temp_script = None
        try:
            # Create a temporary file to save the script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_code)
                temp_script = f.name
            
            # Execute the script in an environment rich with libraries
            exec_globals = {
                '__builtins__': __builtins__, 
                'URIRef': URIRef, 
                'Literal': Literal, 
                'RDF': RDF, 
                'SCHEMA': SCHEMA, 
                'EX': EX,
                'EO': EO,
                'SIO': SIO,
                'XSD': XSD,
                'Namespace': Namespace,
                'pd': pd,
                'io': io,
                'json': json,
                're': re,
                'uuid': uuid
            }
            # LLM-generated code is untrusted: bound its wall-clock time so a runaway
            # loop or pathological parse can't hang the whole Streamlit process.
            with time_limit(60):
                with open(temp_script, 'r') as f:
                    exec(f.read(), exec_globals)

                if 'parse_data_to_triples' not in exec_globals:
                    return False, "Generated script did not define parse_data_to_triples function."

                triples = exec_globals['parse_data_to_triples'](uploaded_file.getvalue(), DATASET_URI)

            # The LLM wrote this script -- validate its output shape before trusting it,
            # rather than assuming it returned exactly what was asked for.
            if not isinstance(triples, list):
                return False, (
                    f"Generated script's parse_data_to_triples() returned "
                    f"{type(triples).__name__}, expected a list of (subject, predicate, object) tuples."
                )
            valid_triples = []
            skipped = 0
            for item in triples:
                if (isinstance(item, (tuple, list)) and len(item) == 3
                        and isinstance(item[0], URIRef) and isinstance(item[1], URIRef)
                        and isinstance(item[2], (URIRef, Literal))):
                    valid_triples.append(tuple(item))
                else:
                    skipped += 1
            if skipped:
                logger_msg = f"⚠️ Skipped {skipped} malformed triple(s) returned by the generated script."
                st.warning(logger_msg)
            triples = valid_triples

        except ExecutionTimeout as e:
            return False, f"Custom ingestion script timed out: {e}"
        except Exception as e:
            return False, f"Error executing custom ingestion script: {e}"
        finally:
            if temp_script and os.path.exists(temp_script):
                try:
                    os.unlink(temp_script)
                except OSError:
                    pass
    else:
        # JSONL format - use existing parser
        triples = []

        # Parse JSONL file
        content = uploaded_file.getvalue().decode('utf-8')
        lines = content.split('\n')

        status_text.text("Parsing JSONL data...")
        for line in lines:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                # Try to detect if it's a review or product
                if 'parent_asin' in data or 'asin' in data:
                    if 'user_id' in data or 'rating' in data:
                        # It's a review
                        review_triples = parse_amazon_review_line(data, DATASET_URI)
                        triples.extend(review_triples)
                    else:
                        # It's a product
                        product_triples = parse_amazon_meta_line(data)
                        triples.extend(product_triples)
            except json.JSONDecodeError:
                continue

    # ---- Shared upload step for BOTH branches above ----
    # (Previously, only the JSONL branch actually built a Graph and uploaded it;
    # the LLM-generated-script branch built `triples` and then fell straight through
    # to a "success" return without ever calling upload_graph_to_fuseki, so non-JSONL
    # custom uploads silently did nothing while reporting success.)
    g = create_bound_graph()
    g.add((DATASET_URI, RDF.type, MLS_DATASET))
    g.add((DATASET_URI, RDF.type, EO.object_record))

    for s, p, o in triples:
        g.add((s, p, o))

    if len(g) > 0:
        status_text.text(f"Uploading {len(g)} triples to the knowledge graph...")
        if not upload_graph_to_product_reviews_via_agent(g):
            return False, "Failed to upload custom data to Fuseki."
    else:
        return False, "No valid triples were extracted from the uploaded file -- nothing was ingested."

    return True, f"✅ Successfully ingested custom dataset with {len(triples)} triples."

def display_triple_sample(sample_triples):
    """Display a sample of triples in a user-friendly format."""
    if not sample_triples:
        st.info("No triples found in the dataset.")
        return
    
    # Group triples by subject for better readability
    triples_by_subject = {}
    for triple in sample_triples:
        s = triple.get('s', {}).get('value', 'Unknown')
        if s not in triples_by_subject:
            triples_by_subject[s] = []
        p = triple.get('p', {}).get('value', '')
        o = triple.get('o', {}).get('value', '')
        o_type = triple.get('o', {}).get('type', 'literal')
        triples_by_subject[s].append({
            'predicate': p.split('/')[-1] if '/' in p else p,
            'object': o,
            'object_type': o_type
        })
    
    # Display as expandable cards
    for subject, triples_list in list(triples_by_subject.items())[:10]:
        with st.expander(f"📄 {subject}"):
            for t in triples_list[:10]:
                st.write(f"**{t['predicate']}**: {t['object']} ({t['object_type']})")
            if len(triples_list) > 10:
                st.write(f"... and {len(triples_list) - 10} more predicates")

# ================= RECOMMENDATION ALGORITHMS TAB (XAI_RecommendationAlgorithms) =================
def get_algorithm_functions():
    """Fetch all algorithm functions from the Functions Graph."""
    # Modified to fetch algorithms conforming to the updated recommendations.ttl (mls:Algorithm)
    # NOTE: ?workflow and ?specNode are now selected explicitly. Their real, existing URIs
    # are looked up from the graph (instead of being re-derived from a naming convention),
    # since some legacy workflows (e.g. Workflow/CF_Recommendation) don't follow the
    # "<Algorithm_URI>_Workflow" pattern. Editing/deleting must target the real URIs or
    # it silently creates a duplicate workflow instead of touching the original one.
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX mls: <http://www.w3.org/ns/mls#>

    SELECT DISTINCT ?uri ?name ?spec ?workflow ?specNode WHERE {
        ?uri a mls:Algorithm ;
             rdfs:label ?name .
        OPTIONAL {
            ?workflow mls:implements ?uri ;
                      mls:hasQuality ?specNode .
            ?specNode rdf:value ?spec .
        }
    }
    """
    return run_sparql_query(RECOMMENDATION_ALGORITHMS_ENDPOINT, query)

def _step_payload(step_uri, info, order):
    """Shape one step for the Admin form.

    The legacy keys ("comment", "order", "uri") are kept exactly as they were so
    any existing reader keeps working; the structured fields are additions.
    """
    return {
        "comment": {"value": info.get('comment', '')},
        "order": {"value": order},
        "uri": step_uri,
        "description": info.get('description', ''),
        "constraint": info.get('constraint', ''),
        "function": info.get('function', ''),
        "inputs": info.get('inputs', []),
        "outputs": info.get('outputs', []),
        "parameters": info.get('parameters', []),
    }


def get_algorithm_steps(algorithm_uri):
    """
    Fetch the steps for a specific algorithm by traversing the step linked list 
    defined in recommendations.ttl (pko:hasFirstStep and pko:nextStep).
    """
    # pko:isStepOf is not a PKO term (defect 2b); the relationship is
    # pko:hasStep, Procedure -> Step.
    query = f"""
    {sparql_prefixes('rdf', 'rdfs', 'mls', 'pko', 'pplan', 'dcterms', 'ex')}

    SELECT ?step ?comment ?description ?next ?first ?stepNumber ?constraint ?function
           (GROUP_CONCAT(DISTINCT ?inLabel;  separator="|") AS ?inputs)
           (GROUP_CONCAT(DISTINCT ?outLabel; separator="|") AS ?outputs)
           (GROUP_CONCAT(DISTINCT ?paramText; separator="|") AS ?parameters)
    WHERE {{
        ?workflow mls:implements <{algorithm_uri}> .
        ?workflow pko:hasStep ?step .
        ?step rdfs:comment ?comment .
        OPTIONAL {{ ?step dcterms:description ?description }}
        OPTIONAL {{ ?step pko:stepNumber ?stepNumber }}
        OPTIONAL {{ ?step ex:generationConstraint ?constraint }}
        OPTIONAL {{ ?step pko:requiresFunction ?fn . ?fn rdfs:label ?function }}
        OPTIONAL {{ ?step pplan:hasInputVar  ?inVar  . ?inVar  rdfs:label ?inLabel  }}
        OPTIONAL {{ ?step pplan:hasOutputVar ?outVar . ?outVar rdfs:label ?outLabel }}
        OPTIONAL {{
            ?step mls:hasHyperParameter ?param .
            ?param rdfs:label ?paramName .
            OPTIONAL {{ ?setting mls:specifiedBy ?param ; mls:hasValue ?paramValue }}
            BIND(CONCAT(?paramName, IF(BOUND(?paramValue),
                        CONCAT("=", ?paramValue), "")) AS ?paramText)
        }}
        OPTIONAL {{ ?step pko:nextStep ?next }}
        OPTIONAL {{
            ?workflow pko:hasFirstStep ?first .
            FILTER(?step = ?first)
        }}
    }}
    GROUP BY ?step ?comment ?description ?next ?first ?stepNumber ?constraint ?function
    """
    bindings = run_sparql_query(RECOMMENDATION_ALGORITHMS_ENDPOINT, query)
    if not bindings:
        return []
    
    steps_dict = {}
    first_step_uri = None

    for b in bindings:
        step_uri = b['step']['value']
        comment = b['comment']['value']
        next_uri = b.get('next', {}).get('value')
        is_first = b.get('first', {}).get('value') is not None
        
        def _split(key):
            raw = b.get(key, {}).get('value', '')
            return [p for p in raw.split('|') if p.strip()] if raw else []

        steps_dict[step_uri] = {
            'comment': comment,
            'next': next_uri,
            'description': b.get('description', {}).get('value', ''),
            'constraint': b.get('constraint', {}).get('value', ''),
            'function': b.get('function', {}).get('value', ''),
            'inputs': _split('inputs'),
            'outputs': _split('outputs'),
            'parameters': _split('parameters'),
        }
        if is_first:
            first_step_uri = step_uri

    # Fallback if first step wasn't explicitly matched
    if not first_step_uri and steps_dict:
        pointed_to = set(s['next'] for s in steps_dict.values() if s['next'])
        for step_uri in steps_dict:
            if step_uri not in pointed_to:
                first_step_uri = step_uri
                break

    if not first_step_uri and steps_dict:
        first_step_uri = list(steps_dict.keys())[0]

    ordered_steps = []
    current_uri = first_step_uri
    visited = set()
    step_num = 1

    while current_uri and current_uri in steps_dict and current_uri not in visited:
        visited.add(current_uri)
        step_info = steps_dict[current_uri]
        ordered_steps.append(_step_payload(current_uri, step_info, str(step_num)))
        current_uri = step_info['next']
        step_num += 1

    # Append orphans
    for step_uri, step_info in steps_dict.items():
        if step_uri not in visited:
            ordered_steps.append(
                _step_payload(step_uri, step_info, f"{step_num} (orphan)"))
            step_num += 1

    return ordered_steps

def _slugify(text):
    """A short, stable, URI-safe fragment derived from a label."""
    out = "".join(c.lower() if c.isalnum() else "_" for c in str(text)).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "value"


def _normalise_steps(steps):
    """Accept either the legacy list-of-strings or the structured list-of-dicts.

    The old form passed five plain strings. Keeping that shape working means a
    caller that has not been updated still saves a valid, if minimal, step.
    """
    normalised = []
    for entry in steps or []:
        if isinstance(entry, str):
            text = entry.strip()
            if text:
                normalised.append({"comment": text})
        elif isinstance(entry, dict):
            # Lists have to be tested as lists. str([]) is "[]", which is
            # truthy, so a blanket str().strip() test would treat an entirely
            # empty step as having content -- and setting the step count to
            # five while filling in three would have saved two empty steps.
            def _has(value):
                if isinstance(value, (list, tuple, set)):
                    return any(str(v).strip() for v in value)
                return bool(str(value or "").strip())

            if any(_has(entry.get(k)) for k in
                   ("comment", "description", "inputs", "outputs",
                    "parameters", "function", "constraint")):
                normalised.append(entry)
    return normalised


def build_step_triples(workflow_uri, steps, existing_step_uris=None,
                       existing_variable_uris=None):
    """Build the SPARQL triples for a workflow's steps.

    Separated from the update call so it can be tested without a triple store.

    Two rules keep an edit from being destructive:

    1. Existing step URIs are reused positionally. The hand-authored workflows
       name their steps /Trace/cf_step1 while a newly minted one would be
       <workflow>/step1. Minting fresh URIs on every save relocated every step
       and orphaned the variables they pointed at, so the URI already in the
       graph wins whenever there is one.

    2. A variable whose label has not changed keeps its URI, so the declared
       variables from the WP2 migration survive an edit of the step that uses
       them. Only genuinely new variables are minted, and those are nested under
       the step that owns them so they can be cleaned up later.

    Returns (triples, step_uris).
    """
    existing_step_uris = list(existing_step_uris or [])
    existing_variable_uris = dict(existing_variable_uris or {})
    steps = _normalise_steps(steps)
    if not steps:
        return [], []

    step_uris = []
    for i in range(len(steps)):
        if i < len(existing_step_uris) and existing_step_uris[i]:
            step_uris.append(existing_step_uris[i])
        else:
            step_uris.append(f"{workflow_uri}/step{i + 1}")

    def lit(value):
        return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")

    def variable_uri(step_uri, label):
        known = existing_variable_uris.get(label.strip().lower())
        return known or f"{step_uri}/var/{_slugify(label)}"

    triples = [f"<{workflow_uri}> pko:hasFirstStep <{step_uris[0]}> ."]

    for i, step in enumerate(steps):
        s = step_uris[i]
        comment = str(step.get("comment") or step.get("description") or "").strip()
        description = str(step.get("description") or "").strip()

        triples.append(f"<{s}> a pplan:Step ;")
        triples.append(f'    rdfs:comment "{lit(comment)}" ;')
        triples.append(f'    pko:stepNumber "{i + 1}" ;')
        triples.append(f"    pplan:isStepOfPlan <{workflow_uri}> .")
        triples.append(f"<{workflow_uri}> pko:hasStep <{s}> .")

        if description:
            triples.append(f'<{s}> dcterms:description "{lit(description)}" .')

        constraint = str(step.get("constraint") or "").strip()
        if constraint:
            triples.append(f'<{s}> ex:generationConstraint "{lit(constraint)}" .')

        function = str(step.get("function") or "").strip()
        if function:
            fn_uri = f"{workflow_uri}/function/{_slugify(function)}"
            triples.append(f"<{s}> pko:requiresFunction <{fn_uri}> .")
            triples.append(f'<{fn_uri}> a pko:Function ; rdfs:label "{lit(function)}" .')

        for prop, key in (("pplan:hasInputVar", "inputs"),
                          ("pplan:hasOutputVar", "outputs")):
            for label in step.get(key) or []:
                label = str(label).strip()
                if not label:
                    continue
                v = variable_uri(s, label)
                triples.append(f"<{s}> {prop} <{v}> .")
                triples.append(f'<{v}> a pplan:Variable ; rdfs:label "{lit(label)}" .')

        for entry in step.get("parameters") or []:
            entry = str(entry).strip()
            if not entry:
                continue
            if "=" in entry:
                pname, pvalue = entry.split("=", 1)
            else:
                pname, pvalue = entry, ""
            pname, pvalue = pname.strip(), pvalue.strip()
            if not pname:
                continue
            hp = f"{s}/param/{_slugify(pname)}"
            triples.append(f"<{s}> mls:hasHyperParameter <{hp}> .")
            triples.append(f'<{hp}> a mls:HyperParameter ; rdfs:label "{lit(pname)}" .')
            if pvalue:
                hps = f"{hp}/setting"
                triples.append(
                    f'<{hps}> a mls:HyperParameterSetting ; '
                    f'mls:specifiedBy <{hp}> ; mls:hasValue "{lit(pvalue)}" .')

        if i < len(steps) - 1:
            triples.append(f"<{s}> pko:nextStep <{step_uris[i + 1]}> .")

    return triples, step_uris


def save_algorithm_function(uri, name, spec, steps=None, workflow_uri=None, spec_node_uri=None):
    """Save or update an algorithm function conforming to recommendations.ttl structure.

    IMPORTANT (bug fix): workflow_uri / spec_node_uri should be the REAL URIs already
    linked to this algorithm in the graph (fetched via get_algorithm_functions), and
    passed in here explicitly when editing an existing algorithm. Some legacy workflows
    (e.g. <.../Workflow/CF_Recommendation>) do not follow the "<Algorithm>_Workflow"
    naming convention. Re-deriving the URI mechanically meant the delete step below
    never matched the real workflow/spec triples -- it deleted nothing (since the
    derived URI didn't exist yet) and the insert step then created a brand-new,
    parallel workflow alongside the original one. Only fall back to the mechanical
    naming convention for genuinely new algorithms, where there is nothing to look up.
    """
    if not workflow_uri:
        if "Algorithm/" in uri:
            workflow_uri = uri.replace("Algorithm/", "Workflow/") + "_Workflow"
        elif "method/" in uri:
            workflow_uri = uri.replace("method/", "workflow/") + "_Workflow"
        else:
            workflow_uri = uri + "_Workflow"
    if not spec_node_uri:
        if "Algorithm/" in uri:
            spec_node_uri = uri.replace("Algorithm/", "Characteristic/") + "_UsageSpec"
        elif "method/" in uri:
            spec_node_uri = uri.replace("method/", "characteristic/") + "_UsageSpec"
        else:
            spec_node_uri = uri + "_UsageSpec"

    # Read what is already there BEFORE deleting, so an edit reuses the step
    # URIs and variable URIs already in the graph instead of minting new ones
    # and orphaning everything the old steps pointed at.
    existing_step_uris = []
    existing_variable_uris = {}
    try:
        for step in get_algorithm_steps(uri):
            existing_step_uris.append(step.get("uri"))
        lookup = f"""
        {sparql_prefixes('rdfs', 'mls', 'pko', 'pplan')}
        SELECT ?var ?label WHERE {{
            <{workflow_uri}> pko:hasStep ?step .
            {{ ?step pplan:hasInputVar ?var }} UNION {{ ?step pplan:hasOutputVar ?var }}
            ?var rdfs:label ?label .
        }}
        """
        for b in run_sparql_query(RECOMMENDATION_ALGORITHMS_ENDPOINT, lookup) or []:
            label = b.get("label", {}).get("value", "").strip().lower()
            if label:
                existing_variable_uris[label] = b["var"]["value"]
    except Exception as e:
        log_error("Save Algorithm", f"Could not read existing step structure: {e}")

    # 1. Delete Existing Data
    del_query = f"""
    {sparql_prefixes('rdf', 'rdfs', 'mls', 'pko', 'pplan')}

    DELETE WHERE {{ <{uri}> ?p ?o . }};
    DELETE WHERE {{ <{workflow_uri}> ?p ?o . }};
    DELETE WHERE {{ <{spec_node_uri}> ?p ?o . }};
    DELETE WHERE {{
        ?step_uri pplan:isStepOfPlan <{workflow_uri}> ;
                  ?p ?o .
    }};
    """
    # Steps are cleared via p-plan:isStepOfPlan; the workflow's own pko:hasStep
    # triples go with the workflow node above.
    if not run_sparql_update(RECOMMENDATION_ALGORITHMS_UPDATE, del_query):
        return False, "Failed to clear existing algorithm data."

    # 2. Insert Core Algorithm, Workflow, and Specification Triples
    clean_spec = spec.replace('"', '\\"').replace('\n', ' ')
    core_insert = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX mls: <http://www.w3.org/ns/mls#>
    PREFIX pko: <https://w3id.org/pko#>

    INSERT DATA {{
        <{uri}> a mls:Algorithm ;
                 rdfs:label "{name}" .

        <{workflow_uri}> a mls:Implementation, pko:Procedure ;
                          rdfs:label "{name} Workflow" ;
                          mls:implements <{uri}> ;
                          mls:hasQuality <{spec_node_uri}> .

        <{spec_node_uri}> a mls:ImplementationCharacteristic ;
                           rdfs:label "Agent Usage Specification" ;
                           rdf:value "{clean_spec}" .
    }}
    """
    if not run_sparql_update(RECOMMENDATION_ALGORITHMS_UPDATE, core_insert):
        return False, "Failed to insert core algorithm metadata."

    # 3. Create and Link Steps dynamically
    #
    # The triple building lives in build_step_triples() so it can be tested
    # without a triple store. Existing step and variable URIs are passed in so
    # an edit reuses them: minting fresh ones relocated /Trace/cf_step1 to
    # <workflow>/step1 and orphaned every variable the step pointed at.
    if steps:
        normalised_steps = _normalise_steps(steps)
        if normalised_steps:
            step_triples, _step_uris = build_step_triples(
                workflow_uri, normalised_steps,
                existing_step_uris=existing_step_uris,
                existing_variable_uris=existing_variable_uris)

            steps_insert = f"""
            {sparql_prefixes('rdf', 'rdfs', 'pko', 'pplan', 'dcterms', 'mls', 'ex')}

            INSERT DATA {{
                {"   ".join(step_triples)}
            }}
            """
            if not run_sparql_update(RECOMMENDATION_ALGORITHMS_UPDATE, steps_insert):
                return False, "Failed to insert sequential algorithm steps."

    # 4. Clean up nodes no step references any more.
    #
    # Variables, hyperparameters and functions are reachable only from a step.
    # When a step is edited to drop one, the node itself would otherwise linger
    # in the graph forever. Each clause is scoped by NOT EXISTS, so a node still
    # referenced by any other step -- including one in a different workflow --
    # is left alone. Settings go before parameters: a setting points at its
    # parameter, so removing the parameter first would strand it.
    gc_query = f"""
    {sparql_prefixes('rdf', 'rdfs', 'pko', 'pplan', 'mls')}

    DELETE {{ ?hps ?p ?o }}
    WHERE {{
        ?hps a mls:HyperParameterSetting ; mls:specifiedBy ?hp ; ?p ?o .
        FILTER NOT EXISTS {{ ?anyStep mls:hasHyperParameter ?hp }}
    }};
    DELETE {{ ?hp ?p ?o }}
    WHERE {{
        ?hp a mls:HyperParameter ; ?p ?o .
        FILTER NOT EXISTS {{ ?anyStep mls:hasHyperParameter ?hp }}
    }};
    DELETE {{ ?v ?p ?o }}
    WHERE {{
        ?v a pplan:Variable ; ?p ?o .
        FILTER NOT EXISTS {{ ?anyStep pplan:hasInputVar ?v }}
        FILTER NOT EXISTS {{ ?anyStep pplan:hasOutputVar ?v }}
    }};
    DELETE {{ ?f ?p ?o }}
    WHERE {{
        ?f a pko:Function ; ?p ?o .
        FILTER NOT EXISTS {{ ?anyStep pko:requiresFunction ?f }}
    }}
    """
    # A failure here leaves unreferenced nodes behind, which is untidy but
    # harmless, so it must not fail the save.
    if not run_sparql_update(RECOMMENDATION_ALGORITHMS_UPDATE, gc_query):
        log_error("Save Algorithm",
                  "Saved, but could not clean up unreferenced variables.")
                
    return True, "Successfully saved algorithm function and reconstructed step sequence."

def delete_algorithm_function(uri, workflow_uri=None, spec_node_uri=None):
    """Delete an algorithm function, its workflow, its specification node, and all step structures.

    Same fix as save_algorithm_function: prefer the REAL workflow_uri/spec_node_uri
    looked up from the graph. Falling back to the naming convention here would mean
    deleting a URI that may never have existed, leaving the actual workflow (and its
    orphaned steps) behind in the store.
    """
    if not workflow_uri:
        if "Algorithm/" in uri:
            workflow_uri = uri.replace("Algorithm/", "Workflow/") + "_Workflow"
        elif "method/" in uri:
            workflow_uri = uri.replace("method/", "workflow/") + "_Workflow"
        else:
            workflow_uri = uri + "_Workflow"
    if not spec_node_uri:
        if "Algorithm/" in uri:
            spec_node_uri = uri.replace("Algorithm/", "Characteristic/") + "_UsageSpec"
        elif "method/" in uri:
            spec_node_uri = uri.replace("method/", "characteristic/") + "_UsageSpec"
        else:
            spec_node_uri = uri + "_UsageSpec"

    del_query = f"""
    {sparql_prefixes('rdf', 'rdfs', 'mls', 'pko', 'pplan')}

    DELETE WHERE {{ <{uri}> ?p ?o . }};
    DELETE WHERE {{ <{workflow_uri}> ?p ?o . }};
    DELETE WHERE {{ <{spec_node_uri}> ?p ?o . }};
    DELETE WHERE {{
        ?step_uri pplan:isStepOfPlan <{workflow_uri}> ;
                  ?p ?o .
    }};
    """
    # Steps are cleared via p-plan:isStepOfPlan, the inverse PKO declares for
    # pko:hasStep. Without this, deleting an algorithm orphans all its steps.
    return run_sparql_update(RECOMMENDATION_ALGORITHMS_UPDATE, del_query)

# ================= INTERACTIVE EXPLANATIONS TAB =================
def get_explanation_types():
    """Fetch all explanation types from the Interactive Explanations Graph."""
    # EO defines no Explanation class; the root is the dedalo ep:Explanation it
    # imports (defect 4), and the graph is migrated to match.
    query = f"""
    {sparql_prefixes('rdfs', 'eo', 'ep', 'ex')}
    SELECT ?uri ?name ?desc ?questions ?action WHERE {{
        ?uri a ep:Explanation ;
             rdfs:label ?name .
        OPTIONAL {{ ?uri rdfs:comment ?desc . }}
        OPTIONAL {{ ?uri ex:exampleQuestions ?questions . }}
        OPTIONAL {{ ?uri ex:llmAction ?action . }}
    }}
    """
    return run_sparql_query(INTERACTIVE_EXPLANATIONS_ENDPOINT, query)

def save_explanation_type(uri, name, desc, questions, action):
    """Save or update an explanation type."""
    if uri:
        del_query = f"""
        PREFIX eo: <https://purl.org/heals/eo#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX ex: <http://linked.aub.edu.lb/kgenxai#>
        DELETE WHERE {{
            <{uri}> ?p ?o .
        }};
        """
        if not run_sparql_update(INTERACTIVE_EXPLANATIONS_UPDATE, del_query):
            return False, "Failed to delete existing explanation type."
    
    clean_desc = desc.replace('"', '\\"').replace('\n', ' ')
    clean_q = questions.replace('"', '\\"').replace('\n', ' ')
    clean_a = action.replace('"', '\\"').replace('\n', ' ')
    
    # Explanation types are typed with the dedalo root class EO actually
    # imports. eo:Explanation was never an EO term (defect 4), and this is the
    # write path that kept reintroducing it every time an admin saved a type.
    ins_query = f"""
    {sparql_prefixes('rdfs', 'eo', 'ep', 'ex')}
    INSERT DATA {{
        <{uri}> a ep:Explanation ;
                rdfs:label "{name}" ;
                rdfs:comment "{clean_desc}" ;
                ex:exampleQuestions "{clean_q}" ;
                ex:llmAction "{clean_a}" .
    }}
    """
    return run_sparql_update(INTERACTIVE_EXPLANATIONS_UPDATE, ins_query), "Successfully saved explanation type."

def delete_explanation_type(uri):
    """Delete an explanation type."""
    del_query = f"""
    PREFIX eo: <https://purl.org/heals/eo#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX ex: <http://linked.aub.edu.lb/kgenxai#>
    DELETE WHERE {{
        <{uri}> ?p ?o .
    }};
    """
    return run_sparql_update(INTERACTIVE_EXPLANATIONS_UPDATE, del_query)

# ================= AGENT SETUP (SYSTEM PROMPTS / IDENTITY) =================
# Each agent's ROLE ("who it is") is defined ONCE, in a fixed, KG-editable place,
# rather than being re-composed inline into every one-off task prompt.
# AGENT_ROLE_DEFAULTS are the fallback identities (used until/unless someone edits
# them in the Admin > Agent Setup tab); AGENT_ROLE_LABELS gives each key a friendly
# name for that tab.
#
# "interviewer"/"classifier"/"ingestion"/"formatter" are used directly by this file's own
# Gemini calls. "selector"/"composer" are sent to the Recommender Agent, "explainer" to
# the Explainer Agent, and "nl_to_sparql" to the ProductReviews Agent -- each of those
# agents falls back to its own built-in default if nothing is sent (so this file being
# unreachable, or the KG having no entries yet, can never break them).
#
# ORCHESTRATOR IDENTITY: the four prompts below (interviewer/classifier/ingestion/
# formatter) are all TASKS performed by one and the same agent -- the Orchestrator --
# not four separate agents. Each Gemini call for those four tasks now gets a
# system_instruction that is this high-level identity PREFIXED to the task-specific
# prompt (see get_orchestrator_system_instruction() below), so the model is always
# told "you are the Orchestrator, and right now your specific job is <task>" rather
# than the task prompt standing in as the model's whole identity. This is purely a
# prompt-composition change -- it does not add a new agent, and every call site's
# behavior is unchanged apart from the richer system instruction.
ORCHESTRATOR_SYSTEM_PROMPT_DEFAULT = (
    "You are the Orchestrator Agent for the KGenXAI Recommender System. Your overall "
    "responsibility is to manage the conversation with the user end-to-end: build up "
    "their preference profile, classify their intent at each turn, optionally ingest a "
    "custom dataset they upload, and channel the right task to the right downstream "
    "agent -- the Recommender Agent (generates recommendations), the Explainer Agent "
    "(answers 'why' questions about a recommendation), and the ProductReviews Agent "
    "(reads and searches the product/review Knowledge Graph). You never perform those "
    "downstream agents' jobs yourself. Below is the specific task you are performing "
    "right now."
)
INTERVIEWER_TOOL_PROMPT_DEFAULT = (
    "You are a Profile Builder for a Recommender System. Your sole responsibility is "
    "to have a natural conversation with the user to build up their preference profile "
    "-- you never generate recommendations yourself."
)
CLASSIFIER_TOOL_PROMPT_DEFAULT = (
    "You are an Intent Classification Agent for a Recommender System's profile-building "
    "flow. You strictly classify user input into one of a fixed set of intents and "
    "extract any items mentioned -- you never generate conversational replies."
)
# POST_REC_CLASSIFIER (2026-07-14 addition): the post-recommendation chat phase used
# to send EVERY message straight to the Explainer Agent, with no intent check at all --
# so "I'd like to add a few more items to my profile" was misrouted to the Explainer
# instead of actually updating the profile (raised directly by the user). This task
# prompt lets the Orchestrator tell those two cases apart before deciding where to send
# the message; see classify_post_recommendation_intent() below.
POST_REC_CLASSIFIER_TOOL_PROMPT_DEFAULT = (
    "You are an Intent Classification Agent for the post-recommendation chat phase of a "
    "Recommender System. You strictly classify whether the user wants to add a specific "
    "new item to their profile, wants to rate more items without naming one specifically, "
    "or is asking a question about the recommendation they already received -- you never "
    "generate conversational replies yourself."
)
# INTERVIEW_CLASSIFIER (2026-07-14 addition): during the interview phase, a request
# like "none of them are good, show me more items" used to go straight to the
# free-form interviewer LLM, which has no real product data or images to work with --
# it would improvise plausible-sounding item names with broken/empty image markup
# (the "no images displaying for show me more items" bug reported from a screenshot).
# This classifier runs first so a genuine "show me a different set of items" request
# is answered via show_fresh_sample_items() (real KG data, guaranteed images) instead.
INTERVIEW_CLASSIFIER_TOOL_PROMPT_DEFAULT = (
    "You are an Intent Classification Agent for the interview phase of a Recommender "
    "System's profile-building flow. You strictly classify whether the user is asking "
    "to see a different/fresh batch of sample items, mentioning a brand new topic or "
    "category they're interested in, or doing anything else (rating shown items, "
    "saying they're ready) -- you never generate conversational replies yourself."
)
INGESTION_TOOL_PROMPT_DEFAULT = (
    "You are a semantic web developer and data ingestion specialist. Your task is to "
    "generate a robust, fully runnable Python script that converts an uploaded data file "
    "into RDF triples that mimic an existing target ontology exactly."
)
FORMATTER_TOOL_PROMPT_DEFAULT = (
    "You are a Recommendation Presentation Agent. Your sole responsibility is to turn a "
    "raw recommender-system result into a short, friendly, accurate narrative for the "
    "end user -- you never invent items or scores that are not in the data you are given."
)
# Mirrors EO_Recommender_Agent.py's own defaults, so the Admin tab can show/edit a single
# consistent value even though these are actually sent over HTTP to that agent.
#
# RECOMMENDER / EXPLAINER / PRODUCT REVIEWS IDENTITY: same pattern
# as ORCHESTRATOR_SYSTEM_PROMPT_DEFAULT above -- one high-level identity prompt
# per agent, composed ahead of that agent's own tool prompt(s) via
# get_agent_system_instruction() below before being sent over HTTP as that agent's
# system_instruction override. The Recommender has two tool prompts (selector,
# composer); the Explainer and ProductReviews Agents currently have one each
# (explainer, nl_to_sparql), but keep the same identity-plus-tool-prompts shape so the
# structure is consistent and ready for more tool prompts later.
RECOMMENDER_SYSTEM_PROMPT_DEFAULT = (
    "You are the Recommender Agent for the KGenXAI Recommender System. Your overall "
    "responsibility is to turn a user's preference profile into a concrete list of "
    "recommended product items, by (1) choosing the most appropriate recommendation "
    "algorithm for that profile, then (2) writing and running the Python script that "
    "executes it against the Knowledge Graph. Below is the specific tool prompt for "
    "the task you are performing right now."
)
SELECTOR_TOOL_PROMPT_DEFAULT = (
    "You are an Expert Recommender System Architect. Your sole responsibility is to "
    "choose the single most appropriate recommendation algorithm for a given user "
    "profile, strictly based on the provided algorithm manuals. You never write code "
    "and you never fabricate an algorithm name that wasn't provided to you."
)
COMPOSER_TOOL_PROMPT_DEFAULT = (
    "You are a Semantic Data Engineer responsible for writing robust, fully runnable "
    "Python scripts that implement a given recommendation algorithm workflow against "
    "a SPARQL-backed knowledge graph. You always follow the mandatory logging setup "
    "and output contract you are given exactly, and you never invent data you were not "
    "given access to."
)
EXPLAINER_SYSTEM_PROMPT_DEFAULT = (
    "You are the Explainer Agent for the KGenXAI Recommender System. Your overall "
    "responsibility is to answer the user's 'why' questions about a recommendation, "
    "grounded strictly in the Explanation Ontology (EO) and the actual data/trace of "
    "the recommendation being explained. Below is the specific tool prompt for the "
    "task you are performing right now."
)
EXPLAINER_TOOL_PROMPT_DEFAULT = (
    "You are an advanced Explainable AI (XAI) Agent natively aligned with the "
    "Explanation Ontology (EO). Your goal is to provide fluid, data-driven, and highly "
    "structured explanations based on a user's conversational query."
)
PRODUCT_REVIEWS_SYSTEM_PROMPT_DEFAULT = (
    "You are the ProductReviews Agent for the KGenXAI Recommender System. Your overall "
    "responsibility is to be the sole point of read access to the product/review "
    "Knowledge Graph for every other agent -- you never write, delete, or otherwise "
    "mutate the graph on behalf of a query. Below is the specific tool prompt for the "
    "task you are performing right now."
)
NL_TO_SPARQL_TOOL_PROMPT_DEFAULT = (
    "You are a Data Engineer building SPARQL queries for a Knowledge Graph. You only "
    "ever write strictly safe, read-only SELECT queries."
)

AGENT_ROLE_DEFAULTS = {
    "orchestrator_system_prompt": ORCHESTRATOR_SYSTEM_PROMPT_DEFAULT,
    "interviewer": INTERVIEWER_TOOL_PROMPT_DEFAULT,
    "classifier": CLASSIFIER_TOOL_PROMPT_DEFAULT,
    "post_rec_classifier": POST_REC_CLASSIFIER_TOOL_PROMPT_DEFAULT,
    "interview_classifier": INTERVIEW_CLASSIFIER_TOOL_PROMPT_DEFAULT,
    "ingestion": INGESTION_TOOL_PROMPT_DEFAULT,
    "formatter": FORMATTER_TOOL_PROMPT_DEFAULT,
    "recommender_system_prompt": RECOMMENDER_SYSTEM_PROMPT_DEFAULT,
    "selector": SELECTOR_TOOL_PROMPT_DEFAULT,
    "composer": COMPOSER_TOOL_PROMPT_DEFAULT,
    "explainer_system_prompt": EXPLAINER_SYSTEM_PROMPT_DEFAULT,
    "explainer": EXPLAINER_TOOL_PROMPT_DEFAULT,
    "product_reviews_system_prompt": PRODUCT_REVIEWS_SYSTEM_PROMPT_DEFAULT,
    "nl_to_sparql": NL_TO_SPARQL_TOOL_PROMPT_DEFAULT,
}
# Naming convention (2026-07-13 addition, per feedback): every agent's dict of prompts
# has exactly the same shape -- one "System Prompt" (the parent -- who the agent is,
# overall) and one or more "Tool Prompts" (the children -- what it does for a specific
# task, using that System Prompt). This is deliberately the SAME terminology for all
# four agents so the Admin tab reads as one consistent pattern rather than four
# different ad hoc naming schemes.
AGENT_ROLE_LABELS = {
    "orchestrator_system_prompt": "🧩 Orchestrator -- System Prompt",
    "interviewer": "🗣️ Orchestrator Tool Prompt -- Profile Builder (interview chat)",
    "classifier": "🧭 Orchestrator Tool Prompt -- Intent Classifier (choose_method / search)",
    "post_rec_classifier": "🧭 Orchestrator Tool Prompt -- Intent Classifier (post-recommendation chat)",
    "interview_classifier": "🧭 Orchestrator Tool Prompt -- Intent Classifier (interview phase)",
    "ingestion": "📥 Orchestrator Tool Prompt -- Custom Dataset Ingestion Specialist",
    "formatter": "📝 Orchestrator Tool Prompt -- Recommendation Presentation Formatter",
    "recommender_system_prompt": "🧩 Recommender Agent -- System Prompt",
    "selector": "🎯 Recommender Tool Prompt -- Algorithm Selector",
    "composer": "🛠️ Recommender Tool Prompt -- Script Composer",
    "explainer_system_prompt": "🧩 Explainer Agent -- System Prompt",
    "explainer": "💬 Explainer Tool Prompt -- XAI Explanation Generator",
    "product_reviews_system_prompt": "🧩 ProductReviews Agent -- System Prompt",
    "nl_to_sparql": "🔎 ProductReviews Tool Prompt -- NL-to-SPARQL Builder",
}
# Drives both the Admin > Agent Setup tree view and the system-prompt+tool-prompt
# composition helper below: one entry per agent, each with exactly one
# system_prompt_key (the parent) and a list of tool_keys (the children) in the
# order they should be displayed.
AGENT_PROMPT_TREE = [
    {"agent": "Orchestrator", "icon": "🧭", "system_prompt_key": "orchestrator_system_prompt",
     "tool_keys": ["interviewer", "classifier", "post_rec_classifier", "interview_classifier", "ingestion", "formatter"]},
    {"agent": "Recommender Agent", "icon": "🎯", "system_prompt_key": "recommender_system_prompt",
     "tool_keys": ["selector", "composer"]},
    {"agent": "Explainer Agent", "icon": "💬", "system_prompt_key": "explainer_system_prompt",
     "tool_keys": ["explainer"]},
    {"agent": "ProductReviews Agent", "icon": "🔎", "system_prompt_key": "product_reviews_system_prompt",
     "tool_keys": ["nl_to_sparql"]},
]
AGENT_DEFINITION_NAMESPACE = "http://linked.aub.edu.lb/kgenxai/AgentDefinition/"

def get_agent_system_instruction(system_prompt_key, system_prompt_default, tool_key, tool_default):
    """Compose one agent's System Prompt (parent -- who the agent is, overall) with
    one of its Tool Prompts (child -- what it does for a specific task) into a single
    system_instruction string. Every Gemini call still gets exactly ONE
    system_instruction -- this just guarantees that string always states the agent's
    overall system prompt before narrowing to the specific tool, instead of the tool
    prompt alone standing in as the model's whole identity."""
    system_prompt = get_agent_role(system_prompt_key, system_prompt_default)
    tool_prompt = get_agent_role(tool_key, tool_default)
    return f"{system_prompt}\n\n### YOUR CURRENT TOOL PROMPT\n{tool_prompt}"

def get_orchestrator_system_instruction(tool_key, tool_default):
    """Thin wrapper over get_agent_system_instruction() for the Orchestrator's own
    Tool Prompts (interviewer/classifier/post_rec_classifier/ingestion/formatter)."""
    return get_agent_system_instruction(
        "orchestrator_system_prompt", ORCHESTRATOR_SYSTEM_PROMPT_DEFAULT, tool_key, tool_default
    )

def get_agent_definitions():
    """Fetch all saved agent role definitions from the Functions Graph
    (XAI_RecommendationAlgorithms), stored as kgenxai:AgentDefinition entities."""
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX kgenxai: <http://linked.aub.edu.lb/kgenxai/>
    SELECT ?uri ?key ?value WHERE {
        ?uri a kgenxai:AgentDefinition ;
             rdfs:label ?key ;
             rdf:value ?value .
    }
    """
    return run_sparql_query(RECOMMENDATION_ALGORITHMS_ENDPOINT, query)

def save_agent_definition(key, role_text):
    """Save or update one agent's role/system-prompt definition in the KG."""
    uri = AGENT_DEFINITION_NAMESPACE + key
    clean_value = role_text.replace('"', '\\"').replace('\n', ' ')
    del_query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    DELETE WHERE {{ <{uri}> ?p ?o . }};
    """
    if not run_sparql_update(RECOMMENDATION_ALGORITHMS_UPDATE, del_query):
        return False, "Failed to clear existing agent definition."
    ins_query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX kgenxai: <http://linked.aub.edu.lb/kgenxai/>
    INSERT DATA {{
        <{uri}> a kgenxai:AgentDefinition ;
                 rdfs:label "{key}" ;
                 rdf:value "{clean_value}" .
    }}
    """
    if not run_sparql_update(RECOMMENDATION_ALGORITHMS_UPDATE, ins_query):
        return False, "Failed to insert agent definition."
    return True, "Successfully saved agent role definition."

def load_agent_roles_into_session():
    """Fetches all agent role definitions from the KG once and caches them in
    session_state, so per-message chat calls don't each re-query Fuseki. Falls back
    silently to AGENT_ROLE_DEFAULTS for any key not (yet) present in the KG."""
    roles = dict(AGENT_ROLE_DEFAULTS)
    try:
        for b in get_agent_definitions():
            key = b.get("key", {}).get("value")
            value = b.get("value", {}).get("value")
            if key and value:
                roles[key] = value
    except Exception as e:
        log_error("Agent Setup", f"Could not load agent role definitions from KG, using defaults: {e}")
    st.session_state.agent_roles = roles

def get_agent_role(key, default=None):
    """Returns the (possibly KG-overridden) role/system-prompt text for `key`."""
    roles = st.session_state.get("agent_roles")
    if roles is None:
        load_agent_roles_into_session()
        roles = st.session_state.agent_roles
    return roles.get(key, default if default is not None else AGENT_ROLE_DEFAULTS.get(key, ""))

# ================= ORIGINAL CHAT FUNCTIONS =================

def fetch_product_reviews_topic():
    """Queries the ProductReviews Agent to find the dominant category."""
    query = """PREFIX schema: <http://schema.org/> SELECT ?category (COUNT(?s) as ?count) WHERE { { SELECT ?s ?category WHERE { ?s schema:category ?category } LIMIT 10000 } } GROUP BY ?category ORDER BY DESC(?count) LIMIT 1"""
    try:
        res = requests.post(PRODUCT_REVIEWS_API, json={'query': query, 'fuseki_base': get_product_reviews_fuseki_base(), 'product_reviews_dataset': get_product_reviews_dataset_name()}, timeout=10)
        if res.status_code == 200:
            return res.json().get('results', {}).get('bindings', [])[0]['category']['value']
        log_error("ProductReviews Agent", f"Topic lookup failed (HTTP {res.status_code}).")
    except Exception as e:
        log_error("ProductReviews Agent", f"Topic lookup failed: {e}")
    return "Products/Items"

# ================= DISCOVERY LAYER =================
# Additive, read-only "capabilities discovery" step: instead of the orchestrator only
# implicitly/hardcodedly knowing what the Recommender, Explainer, and ProductReviews
# agents can do, this asks each agent's own Knowledge Graph what it's actually capable
# of right now, and turns that into a short human-readable briefing.
#
# IMPORTANT: this does NOT replace or touch any of the existing routing/decision logic
# elsewhere in this file (classify_initial_intent, chat_with_interviewer's profile-update
# logic, select_best_algorithm calls, etc.) -- none of that is modified. This briefing is
# purely extra context: it's injected into the interviewer's system prompt as additional
# background, and shown in the sidebar for transparency, so both the LLM and the user can
# see what's driving the system's capabilities from the KG. If discovery fails for any
# reason, it degrades to a safe fallback string rather than raising -- it can never break
# the existing chat flow.
def build_agent_capabilities_briefing():
    """Read-only discovery: asks each agent's KG what it's capable of, for display/context only."""
    sections = []

    # --- Recommender Agent: which algorithms exist (XAI_RecommendationAlgorithms) ---
    try:
        algo_bindings = get_algorithm_functions()
        algo_names = sorted({b['name']['value'] for b in algo_bindings if 'name' in b})
        if algo_names:
            sections.append(
                "**Recommender Agent** can generate recommendations using these algorithms "
                f"(discovered from `XAI_RecommendationAlgorithms`): {', '.join(algo_names)}."
            )
    except Exception as e:
        logger_msg = f"Discovery: could not enumerate recommender algorithms ({e})"
        print(logger_msg)

    # --- Explainer Agent: which explanation styles exist (XAI_InteractiveExplanations) ---
    try:
        expl_bindings = get_explanation_types()
        expl_summaries = []
        for b in expl_bindings:
            name = b.get('name', {}).get('value')
            if name:
                expl_summaries.append(name)
        if expl_summaries:
            sections.append(
                "**Explainer Agent** can answer questions using these explanation styles "
                f"(discovered from `XAI_InteractiveExplanations`): {', '.join(expl_summaries)}."
            )
    except Exception as e:
        print(f"Discovery: could not enumerate explanation types ({e})")

    # --- ProductReviews Agent: dominant domain/topic ---
    try:
        topic = fetch_product_reviews_topic()
        if topic:
            sections.append(
                f"**ProductReviews Agent** holds review/rating data primarily about: {topic}."
            )
    except Exception as e:
        print(f"Discovery: could not fetch product reviews topic ({e})")

    if not sections:
        return "_No agent capabilities could be discovered from the Knowledge Graph right now (check Fuseki connectivity)._"

    return "\n".join(f"- {s}" for s in sections)

def fetch_product_reviews_samples(exclude_uris=None):
    """Queries the ProductReviews Agent to fetch random sample items, WITH images AND
    their system URIs, instantly.

    `exclude_uris` (2026-07-14 addition): an optional collection of item URIs to
    exclude from the results -- used so repeated "show me more items" requests never
    resurface something the user already saw and rejected. Without this, the small
    random OFFSET window below could easily overlap across calls.

    Returns a list of {"item": name, "image": img_url, "uri": item_uri} dicts, or an
    empty list if nothing could be fetched. On failure this now logs a real error to
    the System Console (see log_error) instead of silently substituting fake
    placeholder items -- previously "Sample Item A" / "Sample Item B", which caused
    users to see irrelevant placeholder items with no indication anything had failed.
    The caller is responsible for telling the user when this returns empty.

    CONTENT SAFETY (2026-07-23): any item whose name is flagged by
    is_item_name_unsafe() (explicit/adult, sexual, racial, or gender-based slur
    terms) is REPLACED with a different item rather than shown -- it's added to
    the exclusion set and a follow-up query backfills its slot, up to a small
    retry budget. The base SPARQL query/shape is unchanged; this only widens the
    same NOT IN(...) exclusion filter that already existed for "show me more".
    """
    import random
    target = 5
    max_attempts = 4  # 1 normal query + up to 3 backfill retries if items get filtered
    seen_uris = set(u for u in (exclude_uris or []) if u)
    collected = []
    had_real_failure = False

    for attempt in range(max_attempts):
        rand_offset = random.randint(0, 100)
        exclude_filter = ""
        if seen_uris:
            uri_list = ", ".join(f"<{u}>" for u in seen_uris if u)
            if uri_list:
                exclude_filter = f"FILTER(?item NOT IN ({uri_list}))"
        query = f"""
        PREFIX schema: <http://schema.org/>
        SELECT DISTINCT ?item ?name ?img WHERE {{
            ?item a schema:Product .
            ?item schema:name ?name .
            OPTIONAL {{ ?item schema:image ?img }}
            {exclude_filter}
        }} LIMIT {target} OFFSET {rand_offset}
        """
        try:
            res = requests.post(PRODUCT_REVIEWS_API, json={'query': query, 'fuseki_base': get_product_reviews_fuseki_base(), 'product_reviews_dataset': get_product_reviews_dataset_name()}, timeout=10)
            if res.status_code == 200:
                bindings = res.json().get('results', {}).get('bindings', [])
                for b in bindings:
                    if 'name' not in b:
                        continue
                    name = b['name']['value']
                    uri = b.get('item', {}).get('value', '')
                    if uri:
                        seen_uris.add(uri)
                    if is_item_name_unsafe(name):
                        # Skip it -- it's already excluded via seen_uris above, so the
                        # next attempt's query will backfill with a different item.
                        log_info("Content Safety Filter: excluded a flagged random sample item and will fetch a replacement.")
                        continue
                    collected.append({
                        "item": name,
                        "image": b.get('img', {}).get('value', ''),
                        "uri": uri,
                    })
                    if len(collected) >= target:
                        break
                if len(collected) >= target or not bindings:
                    # Either we have a full batch, or the KG has nothing further to
                    # offer at all -- either way, more attempts won't help.
                    break
            else:
                log_error("ProductReviews Agent", f"Sample fetch failed (HTTP {res.status_code}): {res.text[:200]}")
                had_real_failure = True
                break
        except Exception as e:
            log_error("ProductReviews Agent", f"Sample fetch failed: {e}")
            had_real_failure = True
            break

    if collected:
        return collected
    if not had_real_failure:
        log_error("ProductReviews Agent", "Sample query returned zero items (the dataset may be empty, or all candidates were filtered).")
    return []

def search_product_reviews_items(item_names, exclude_uris=None):
    """Queries the ProductReviews Agent to perform fuzzy searches for items, WITH
    images AND their system URIs, instantly.

    `exclude_uris` (2026-07-16 addition): an optional collection of item URIs to
    exclude -- without this, re-running the same topic search (e.g. re-searching
    "sports" after "give me other options") returned the exact same top matches
    every time, since this is a deterministic CONTAINS query with no randomization.
    """
    matches = []
    exclude_uris = exclude_uris or set()
    for name in item_names:
        safe_name = name.replace('"', '').replace('\\', '')
        exclude_filter = ""
        if exclude_uris:
            uri_list = ", ".join(f"<{u}>" for u in exclude_uris if u)
            if uri_list:
                exclude_filter = f"FILTER(?item NOT IN ({uri_list}))"
        query = f"""
        PREFIX schema: <http://schema.org/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?item ?name ?img WHERE {{
            ?item a schema:Product .
            ?item schema:name ?name .
            FILTER(CONTAINS(LCASE(?name), LCASE("{safe_name}")))
            {exclude_filter}
            OPTIONAL {{ ?item schema:image ?img }}
        }} LIMIT 5
        """
        try:
            res = requests.post(PRODUCT_REVIEWS_API, json={'query': query, 'fuseki_base': get_product_reviews_fuseki_base(), 'product_reviews_dataset': get_product_reviews_dataset_name()}, timeout=10)
            if res.status_code == 200:
                for b in res.json().get('results', {}).get('bindings', []):
                    if 'name' in b:
                        matches.append({
                            "item": b['name']['value'],
                            "image": b.get('img', {}).get('value', ''),
                            "uri": b.get('item', {}).get('value', ''),
                        })
            else:
                log_error("ProductReviews Agent", f"Search for '{name}' failed (HTTP {res.status_code}): {res.text[:200]}")
        except Exception as e:
            log_error("ProductReviews Agent", f"Search for '{name}' failed: {e}")
        
    unique_matches = []
    seen = set()
    for m in matches:
        if m['item'] not in seen:
            seen.add(m['item'])
            unique_matches.append(m)
    return unique_matches

# ================= IMAGE HARDENING =================
# Guarantee: every item ever shown to the user (recommendation results, sample lists,
# fuzzy-search matches) carries a renderable image -- either a real schema:image URL
# from XAI_ProductReviews, or this local placeholder. Nothing downstream should ever
# have to branch on "does this item have an image".
NO_IMAGE_PLACEHOLDER = (
    "data:image/svg+xml;charset=UTF-8,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E"
    "%3Crect width='100%25' height='100%25' fill='%23e0e0e0'/%3E"
    "%3Ctext x='50%25' y='50%25' font-size='11' text-anchor='middle' fill='%23888888' "
    "dominant-baseline='middle' font-family='sans-serif'%3ENo Image%3C/text%3E%3C/svg%3E"
)

def render_item_card_html(name, img_url):
    """Single source of truth for how one item + its image is rendered. Always
    produces an <img>, using NO_IMAGE_PLACEHOLDER when the KG has no schema:image
    for this item, so the caller never needs an if/else around image presence.

    CONTENT SAFETY (2026-07-23): the display name is passed through
    sanitize_item_title() before rendering, which strips any explicit/adult,
    sexual, racial, or gender-based slur terms from the title text. This is the
    single place every item name reaches the UI from (samples, search matches,
    the recommendation gallery, and the explanation gallery), so every one of
    those flows is covered here without needing to touch each caller -- and
    without touching the Recommender/Explainer algorithm logic itself, which
    still decides/receives the *same* items as before; only the label shown to
    the user is cleaned up.
    """
    safe_img = img_url if img_url else NO_IMAGE_PLACEHOLDER
    display_name = sanitize_item_title(name) if name else name
    safe_name = str(display_name).replace('"', '&quot;') if display_name else "Recommended Item"
    return (
        f'<a href="{safe_img}" target="_blank" title="Click to view full-size image">'
        f'<img src="{safe_img}" style="width: 80px; height: 80px; object-fit: cover; '
        f'border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 8px 0; '
        f'cursor: pointer; transition: transform 0.2s;" '
        f'onmouseover="this.style.transform=\'scale(1.05)\'" '
        f'onmouseout="this.style.transform=\'scale(1)\'"/></a>'
        f'<div style="font-size:12px;">{safe_name}</div>'
    )

def build_image_gallery_html(items):
    """Deterministically renders every item in `items` with its image, side by side.
    This is called AFTER any LLM-generated narrative text so that the presence of
    images next to every returned item is guaranteed by code, not by the LLM
    remembering to follow an instruction."""
    if not items or not isinstance(items, list):
        return ""
    cards = []
    for item in items:
        if isinstance(item, dict):
            name = (item.get('name') or item.get('title') or item.get('item_id')
                    or item.get('item') or item.get('s') or item.get('uri') or "Recommended Item")
            img_url = item.get('image')
        else:
            name = str(item)
            img_url = None
        card_inner = render_item_card_html(name, img_url)
        cards.append(
            f'<div style="display:inline-block; width:110px; margin:6px; '
            f'text-align:center; vertical-align:top;">{card_inner}</div>'
        )
    return (
        '<div style="display:flex; flex-wrap:wrap; margin-top:8px;">'
        + "".join(cards) + "</div>"
    )

def enrich_recommendations_with_images(recommendations):
    """Queries XAI_ProductReviews (via the ProductReviews Agent) to find image URLs
    for recommended items that don't already have one, and guarantees every returned
    item ends up with a usable `image` value -- falling back through three passes
    before settling on the shared NO_IMAGE_PLACEHOLDER, so no item is ever returned
    without one:
      0. Trust an `image` the generated Recommender script already resolved (its
         OUTPUT CONTRACT requires this) -- skip re-querying for that item.
      1. Exact match on item URI (VALUES + schema:image).
      2. Exact case-insensitive match on item name/label.
      3. Fuzzy CONTAINS match on name, for whatever is still unmatched after 1-2.
    """
    enriched = []
    uris_to_check = []
    names_to_check = []

    for item in recommendations:
        if isinstance(item, str):
            if item.startswith("http"): uris_to_check.append(item)
            else: names_to_check.append(item)
        elif isinstance(item, dict):
            # HARDENING: the generated Recommender script's OUTPUT CONTRACT now requires
            # it to resolve schema:image itself. If it already did, trust that value and
            # skip re-querying the KG for this item -- only fall back to a fresh lookup
            # when the script didn't supply a usable image.
            existing_image = item.get('image')
            if existing_image and str(existing_image).strip() and str(existing_image) != NO_IMAGE_PLACEHOLDER:
                continue

            uri = item.get('s') or item.get('uri')
            name = item.get('name') or item.get('title') or item.get('item_id') or item.get('item')

            if uri and str(uri).startswith("http"): uris_to_check.append(str(uri))
            elif name: names_to_check.append(str(name))

    fuseki_base = get_product_reviews_fuseki_base()

    # ---- Pass 1: exact URI match ----
    uri_img_map = {}
    if uris_to_check:
        uri_values = " ".join([f"<{u}>" for u in set(uris_to_check)])
        q_uri = f"PREFIX schema: <http://schema.org/> SELECT ?item ?img WHERE {{ VALUES ?item {{ {uri_values} }} ?item schema:image ?img . }}"
        try:
            res = requests.post(PRODUCT_REVIEWS_API, json={'query': q_uri, 'fuseki_base': fuseki_base}, timeout=10)
            if res.status_code == 200:
                for b in res.json().get('results', {}).get('bindings', []):
                    uri_img_map[b['item']['value']] = b['img']['value']
        except Exception as e:
            log_error("ProductReviews Agent", f"Image lookup (URI pass) failed: {e}")

    # ---- Pass 2: exact case-insensitive name/label match ----
    name_img_map = {}
    if names_to_check:
        name_values = " ".join(['"{}"'.format(str(n).replace('"', '').replace('\\', '')) for n in set(names_to_check)])
        q_name = f"""
        PREFIX schema: <http://schema.org/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?search_name (SAMPLE(?img) as ?image) WHERE {{
            VALUES ?search_name {{ {name_values} }}
            ?item a schema:Product .
            {{ ?item schema:name ?name }} UNION {{ ?item rdfs:label ?name }}
            FILTER(LCASE(STR(?name)) = LCASE(STR(?search_name)))
            ?item schema:image ?img .
        }} GROUP BY ?search_name
        """
        try:
            res = requests.post(PRODUCT_REVIEWS_API, json={'query': q_name, 'fuseki_base': fuseki_base}, timeout=10)
            if res.status_code == 200:
                for b in res.json().get('results', {}).get('bindings', []):
                    name_img_map[b['search_name']['value'].lower()] = b['image']['value']
        except Exception as e:
            log_error("ProductReviews Agent", f"Image lookup (exact-name pass) failed: {e}")

    # ---- Pass 3: fuzzy CONTAINS fallback, only for names still unmatched ----
    unmatched_names = [n for n in set(names_to_check) if n.lower() not in name_img_map]
    for n in unmatched_names:
        safe_n = str(n).replace('"', '').replace('\\', '')
        q_fuzzy = f"""
        PREFIX schema: <http://schema.org/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?img WHERE {{
            ?item a schema:Product .
            {{ ?item schema:name ?name }} UNION {{ ?item rdfs:label ?name }}
            FILTER(CONTAINS(LCASE(STR(?name)), LCASE("{safe_n}")))
            ?item schema:image ?img .
        }} LIMIT 1
        """
        try:
            res = requests.post(PRODUCT_REVIEWS_API, json={'query': q_fuzzy, 'fuseki_base': fuseki_base}, timeout=10)
            if res.status_code == 200:
                bindings = res.json().get('results', {}).get('bindings', [])
                if bindings:
                    name_img_map[n.lower()] = bindings[0]['img']['value']
        except Exception as e:
            log_error("ProductReviews Agent", f"Image lookup (fuzzy pass) failed for '{n}': {e}")

    # ---- Assemble, guaranteeing a non-empty `image` on every item ----
    for item in recommendations:
        img_url = None
        if isinstance(item, str):
            img_url = uri_img_map.get(item) if item.startswith("http") else name_img_map.get(item.lower())
            enriched.append({"item": item, "image": img_url or NO_IMAGE_PLACEHOLDER})
        elif isinstance(item, dict):
            # Trust an image the generated script already resolved (per the OUTPUT
            # CONTRACT); only fall back to our own lookup maps if it didn't supply one.
            existing_image = item.get('image')
            if existing_image and str(existing_image).strip() and str(existing_image) != NO_IMAGE_PLACEHOLDER:
                enriched.append(item)
                continue

            uri = item.get('s') or item.get('uri')
            name = item.get('name') or item.get('title') or item.get('item_id') or item.get('item')

            if uri and str(uri).startswith("http"):
                img_url = uri_img_map.get(str(uri))
            elif name:
                img_url = name_img_map.get(str(name).lower())

            item['image'] = img_url or NO_IMAGE_PLACEHOLDER
            enriched.append(item)
        else:
            enriched.append({"item": str(item), "image": NO_IMAGE_PLACEHOLDER})

    return enriched

def classify_initial_intent(user_input):
    model_name = st.session_state.get("selected_model", "gemini-3-flash-preview")
    role_prompt = get_orchestrator_system_instruction("classifier", CLASSIFIER_TOOL_PROMPT_DEFAULT)
    model = genai.GenerativeModel(model_name, system_instruction=role_prompt)
    prompt = f"""
    The user is interacting with a Recommender System profile builder.
    Determine their intent:
    - Choose 1 or ask for samples -> "CHOOSE_SAMPLES".
    - Choose 2 but no items -> "PROVIDE_ITEMS".
    - Mention specific items -> "SEARCH_ITEMS" and extract items.
    Respond STRICTLY in JSON:
    {{"intent": "CHOOSE_SAMPLES" | "PROVIDE_ITEMS" | "SEARCH_ITEMS", "items": ["item1"]}}
    User input: "{user_input}"
    """
    response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    return json.loads(response.text)

def classify_post_recommendation_intent(user_input):
    """Distinguishes three things in the post_recommendation_chat phase: "add a
    specific item to my profile", "show me fresh items to rate" (no specific item
    named), and "ask a question about the recommendation I already got". Added
    2026-07-14 (ADD_PREFERENCE vs ASK_EXPLANATION only); extended 2026-07-16 to add
    MORE_SAMPLES after "give me more products to rate" -- which names no specific
    item -- fell through to ASK_EXPLANATION and got answered by the Explainer
    Agent instead of actually pulling fresh items, since the only two categories
    that existed couldn't represent it."""
    model_name = st.session_state.get("selected_model", "gemini-3-flash-preview")
    role_prompt = get_orchestrator_system_instruction("post_rec_classifier", POST_REC_CLASSIFIER_TOOL_PROMPT_DEFAULT)
    model = genai.GenerativeModel(model_name, system_instruction=role_prompt)
    prompt = f"""
    The user already received a recommendation and is now chatting further. Determine
    their intent:
    - They mention specific new item(s) they like/dislike, or ask to add/update their
      profile/preferences with a NAMED item -> "ADD_PREFERENCE" and extract the item names.
    - They ask to rate more items, see more products, or build their profile further,
      WITHOUT naming any specific item (e.g. "give me more products to rate", "show me
      more items") -> "MORE_SAMPLES".
    - Anything else -- asking why something was/wasn't recommended, how the algorithm
      works, general commentary, thanks, or a follow-up question about the explanation
      -- -> "ASK_EXPLANATION".
    Respond STRICTLY in JSON:
    {{"intent": "ADD_PREFERENCE" | "MORE_SAMPLES" | "ASK_EXPLANATION", "items": ["item1"]}}
    User input: "{user_input}"
    """
    response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    return json.loads(response.text)

def classify_interview_intent(user_input):
    """Distinguishes three things during the interview phase: "show me a different/
    fresh batch of items" (no new topic named), "I'm now interested in a NEW topic/
    category I haven't mentioned before" (search it for real), and normal
    profile-building chat (rating/commenting on items already shown, readiness,
    etc.). Added 2026-07-14 (MORE_SAMPLES vs PROFILE_CHAT only); extended
    2026-07-17 to add SEARCH_NEW_TOPIC after "I like sports" -- said right after
    rating a batch of "creams" items -- got routed to PROFILE_CHAT and answered by
    the free-form interviewer LLM, which has no real search capability of its own
    and can only log it as a soft preference (`preferred_categories`), instead of
    actually searching the Knowledge Graph for real sports-related items the way
    "I like creams" did a moment earlier in the search_specific phase. The
    interviewer LLM narrating "Got it, I've noted that..." is not the same as an
    item ever actually being searched, shown, or added -- this intent exists so a
    new topic gets the exact same real search treatment no matter which phase the
    user happens to be in when they mention it.

    2026-07-17 fix (same day, second bug): that first fix immediately regressed
    referencing an ALREADY-SHOWN item -- e.g. "I like big dragonfly" right after
    item #1 on screen was literally named "Big Dragonfly Adjustable Neoprene
    Sports...". With no visibility into what was actually on screen, the
    classifier had no way to tell "this references a shown item" apart from "this
    is a brand new topic", and guessed SEARCH_NEW_TOPIC -- triggering a fresh,
    unrelated catalog search for the literal string "big dragonfly", which (being
    only a fragment/rephrasing of the full product name) found no exact CONTAINS
    match and surfaced the "I couldn't find exact matches" message instead of
    just rating the item the user was clearly pointing at. Now passes the actual
    list of items shown this session as context so the model can check against it
    first, instead of guessing blind.

    2026-07-17 fix #2 (same day, third bug): that second fix over-corrected the
    other way. After being shown cycling items including "ALKARMI Compression Knee
    Brace... Supports Crossfit, Hiking, Running, Cycling, Basketball, Weightlifting,
    Gym, Sports, Workout", saying "I like sports" got misread as referencing THAT
    item (since "sports" is a literal substring of its long description) and
    classified as PROFILE_CHAT -- right back to the original bug of being logged as
    a vague preference with no real search. The fix must distinguish a genuine
    reference to an item's DISTINCTIVE name (e.g. "big dragonfly" matching a
    product literally branded/named "Big Dragonfly...") from a broad category/
    activity word that merely happens to appear as one of many generic descriptor
    tags in a long compound product title (e.g. "sports" appearing amid "Crossfit,
    Hiking, Running, Cycling, Basketball, Weightlifting, Gym, Sports, Workout" is
    incidental, not what that product distinctively IS) -- only the former counts
    as a match.
    """
    model_name = st.session_state.get("selected_model", "gemini-3-flash-preview")
    role_prompt = get_orchestrator_system_instruction("interview_classifier", INTERVIEW_CLASSIFIER_TOOL_PROMPT_DEFAULT)
    model = genai.GenerativeModel(model_name, system_instruction=role_prompt)
    shown_item_names = list(st.session_state.get("known_item_uris", {}).keys())
    prompt = f"""
    The user is in the middle of building their preference profile by rating sample
    items. Determine their intent:
    - They say the items shown weren't good/relevant, or explicitly ask for more/
      different/fresh items to rate, WITHOUT naming a new topic/category -> "MORE_SAMPLES".
    - They mention a NEW topic, category, or item type that is NOT a distinctive-name
      match to anything in the "Items already shown this session" list below ->
      "SEARCH_NEW_TOPIC" and extract the topic/item name(s).
    - Anything else -- including rating or commenting on an item whose DISTINCTIVE
      NAME/BRAND (not just any incidental word in its full title) reasonably matches
      something in the "Items already shown this session" list below, saying they're
      ready for a recommendation, general chat -> "PROFILE_CHAT".

      CRITICAL -- how to judge a "match" against the shown-items list: only count it
      as PROFILE_CHAT if the user's wording matches what a shown item DISTINCTIVELY
      IS (its brand/product name, e.g. "big dragonfly" matching a product literally
      named "Big Dragonfly Adjustable Neoprene Sports..."). Do NOT count it as a
      match just because the word happens to appear somewhere inside a long,
      multi-purpose product title/description -- e.g. if a shown item's title lists
      many generic use-cases like "...Supports Crossfit, Hiking, Running, Cycling,
      Basketball, Weightlifting, Gym, Sports, Workout", the word "sports" appearing
      there is incidental, not that item's distinctive identity, so a user saying "I
      like sports" is naming a broad NEW category (SEARCH_NEW_TOPIC), not referring
      to that specific item (PROFILE_CHAT) -- even though the substring is present.
      When in doubt: a specific/unusual product name or brand -> likely a real match;
      a broad, generic category or activity word -> likely a new topic, not a match.

    Items already shown this session (name -> already has a resolved KG URI):
    {json.dumps(shown_item_names)}

    Respond STRICTLY in JSON:
    {{"intent": "MORE_SAMPLES" | "SEARCH_NEW_TOPIC" | "PROFILE_CHAT", "items": ["item1"]}}
    User input: "{user_input}"
    """
    response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    return json.loads(response.text)

def show_fresh_sample_items(intro_text):
    """Fetches a fresh batch of real, KG-backed sample items -- via
    fetch_product_reviews_samples(), guaranteed a real image or the shared
    placeholder via render_item_card_html() -- and shows them to the user.

    Used both by the initial "Rate Random Samples" choice AND (2026-07-14 fix) when
    the user asks for a different/fresh set of items mid-interview (e.g. "none of
    them are good, show me more" -- see classify_interview_intent() below). Before
    this fix, that mid-interview request went to the free-form interviewer LLM,
    which had no real item data or images to work with and would improvise plausible
    -sounding product names with broken/empty image markup -- exactly the "no images
    displaying for show me more" bug reported from the screenshot.
    """
    st.session_state.step = "interview"
    with st.spinner("🛍️ Consulting Product Reviews Agent..."):
        samples = fetch_product_reviews_samples(exclude_uris=st.session_state.shown_sample_uris)
    if samples:
        # Remember exactly what was shown (name -> uri) so later free-text turns in
        # the interview ("I like item 1 and 3") can be tied back to a real KG URI
        # instead of only a name string -- this is what fixes items later showing up
        # as "Could not find URI for item".
        for s in samples:
            if s.get("item") and s.get("uri"):
                st.session_state.known_item_uris[s["item"]] = s["uri"]
                st.session_state.shown_sample_uris.add(s["uri"])

        formatted_samples = []
        for i, sample_obj in enumerate(samples):
            s_name = sample_obj.get("item", "")
            s_img = sample_obj.get("image", "")
            # HARDENING: always render an image (falls back to the shared
            # placeholder), never a name-only line.
            img_html = render_item_card_html(s_name, s_img)
            formatted_samples.append(f"{i+1}. {img_html}")

        bullet_list = "\n\n".join(formatted_samples)
        msg = f"{intro_text}\n\n{bullet_list}"
    else:
        # HARDENING: previously this silently substituted fake "Sample Item A/B"
        # placeholders on failure, leaving users looking at irrelevant items with no
        # indication anything had gone wrong. Now show a real, visible error instead.
        st.error("⚠️ Couldn't fetch sample items from the Knowledge Graph right now -- see System Admin > Console for details.")
        msg = f"I couldn't fetch samples from the database right now. Can you tell me what {st.session_state.product_reviews_topic} you like instead?"
    st.session_state.messages.append({"role": "assistant", "content": msg, "phase": "interview"})
    st.rerun()

def show_more_items(random_intro_text):
    """Shared 'show me more items' handler for every follow-up entry point
    (interview MORE_SAMPLES, ready_to_fetch, feedback, post_recommendation_chat).

    HARDENING (2026-07-16): if the user has an active search topic from earlier
    (e.g. they said "I like sports"), re-searches THAT topic -- excluding what's
    already been shown, via search_product_reviews_items()'s exclude_uris -- so
    "give me other options" stays on-topic. Previously every "show me more"
    request went straight to fetch_product_reviews_samples() (blind random,
    topic-agnostic), so a sports search would show sports items once, then lose
    the topic entirely on the next request and switch to unrelated generic items.
    Falls back to genuinely random KG samples only when there's no active topic
    (e.g. the user originally chose "Rate Random Samples").
    """
    if st.session_state.get("last_search_terms"):
        process_search_items(st.session_state.last_search_terms)
    else:
        show_fresh_sample_items(random_intro_text)

def consult_explainer_agent(user_input):
    """Shared Explainer-Agent consultation, used by BOTH the feedback phase
    and post_recommendation_chat -- extracted 2026-07-17 so asking for an
    explanation (or adding an item / requesting more samples, via the same
    3-way classifier) works whether or not the user has clicked 'Submit
    Rating' yet. Previously the feedback phase's chat input could only
    detect named items to add (via classify_initial_intent), with no path to
    the Explainer at all -- asking 'why was X recommended?' before rating
    just returned "I couldn't detect a specific item to add there."
    """
    with st.status("🧠 Consulting Explainer Agent...", expanded=True) as status:
        log_info(f"consult_explainer_agent: user_query='{user_input}'")
        try:
            payload = {
                "user_profile": st.session_state.user_profile,
                "recommendation_result": st.session_state.recommendation_result,
                "user_query": user_input,
                "api_key": st.session_state.api_key,
                "selected_model": st.session_state.selected_model,
                "fuseki_base": get_product_reviews_fuseki_base(),
                "product_reviews_dataset": get_product_reviews_dataset_name(),
                "algorithms_dataset": ALGORITHMS_DATASET_NAME,
                "explanations_dataset": EXPLANATIONS_DATASET_NAME,
                "executions_dataset": EXECUTIONS_DATASET_NAME,
                "execution_id": st.session_state.get("current_execution_id"),
                # Agent Setup override (Admin tab) -- the Explainer Agent falls back
                # to its own built-in default if this is None/absent.
                "explainer_system_prompt": get_agent_system_instruction(
                    "explainer_system_prompt", EXPLAINER_SYSTEM_PROMPT_DEFAULT, "explainer", EXPLAINER_TOOL_PROMPT_DEFAULT
                ),
            }

            response = requests.post(EXPLAINER_API, json=payload, timeout=60)
            if response.status_code == 200:
                explanation_data = response.json()
                deduced_style = explanation_data.get('selected_style', 'General')
                reason_for_style = explanation_data.get('reason_for_style', 'N/A')
                status.update(label="✅ Response Generated", state="complete", expanded=False)
                expl_text = explanation_data.get('explanation_text', 'No response provided.')
                # CONTENT SAFETY (2026-07-23): defense-in-depth -- the items discussed
                # here come from st.session_state.recommendation_result, whose names are
                # already sanitized (see sanitize_recommendation_results), but this
                # catches anything explicit/adult, sexual, racial, or gender-based
                # slur-related the Explainer LLM might still introduce on its own in
                # freeform prose.
                expl_text = sanitize_narrative_text(expl_text)

                # HARDENING: the initial recommendation step already guarantees an
                # image gallery for every returned item via build_image_gallery_html();
                # this follow-up explainer step did not, so images could silently be
                # missing from follow-up item displays. Deterministically append the
                # same gallery for the items actually being discussed here, rather
                # than relying on the Explainer's narrative text to carry images.
                rec_result = st.session_state.recommendation_result or {}
                gallery_html = build_image_gallery_html(rec_result.get('results', []))
                if gallery_html:
                    expl_text = f"{expl_text}\n\n{gallery_html}"

                explainer_reasoning = {
                    "intent_detected": "Explanation Generation",
                    "kg_nodes_accessed": [
                        "ep:Explanation (XAI_InteractiveExplanations)",
                        "schema:Product (XAI_ProductReviews, via ProductReviews Agent)",
                        "mls:Algorithm + p-plan:Step chain (XAI_RecommendationAlgorithms)",
                        "pko:ProcedureExecution (XAI_ExecutionLogs)",
                    ],
                    # Was f"eo:{deduced_style}Explanation", which displayed
                    # undefined terms such as eo:ScientificExplanation to the
                    # user. Resolved through the same verified table used when
                    # persisting, so what is shown matches what is written.
                    "ontology_mapping": (
                        explanation_class_for(deduced_style)
                        or "ep:Explanation (no subtype claimed)"),
                    "narrative": f"Mapped user natural language query to EO XAI Style. {reason_for_style}",
                    "dynamic_trace": {
                        "action": "explanation_generation",
                        "style": deduced_style,
                        "reason": reason_for_style,
                        "user_query": user_input,
                        "strategy_used": st.session_state.recommendation_result.get('strategy', 'Unknown')
                    }
                }

                st.session_state.messages.append({
                    "role": "assistant", "content": expl_text, "reasoning": explainer_reasoning, "phase": "explanation"
                })
                save_explanation_to_fuseki(user_input, explanation_data)
                st.rerun()
            else:
                status.update(label="❌ Explainer Error", state="error")
                # HARDENING (2026-07-08 fix): previously this reported a generic
                # "unavailable" message with no status code or response body, so the
                # System Console couldn't show the actual reason the call failed.
                # Try to surface the Explainer Agent's own reported error first (it
                # returns {"error": str(e), ...} on a 500), falling back to raw text.
                # The technical detail goes to the log (via `exception=`) only --
                # the chat bubble itself stays short and user-friendly.
                try:
                    detail = response.json().get("error", response.text[:300])
                except Exception:
                    detail = response.text[:300]
                report_chat_error(
                    "Explainer Agent",
                    "The Explainer Agent returned an error",
                    f"HTTP {response.status_code}: {detail}",
                    phase="explanation",
                )
                st.rerun()
        except requests.exceptions.Timeout:
            status.update(label="❌ Request Timeout", state="error")
            report_chat_error("Explainer Agent", "The explanation request timed out (the server may be busy)", phase="explanation")
            st.rerun()
        except Exception as e:
            status.update(label="❌ Connection Error", state="error")
            report_chat_error("Explainer Agent", "I couldn't connect to the Explainer Agent", e, phase="explanation")
            st.rerun()

def process_search_items(items):
    # HARDENING (2026-07-16): remember the topic so a later "give me other
    # options" (see MORE_SAMPLES routing below) can re-search THIS topic instead
    # of losing it and falling back to unrelated random samples.
    st.session_state.last_search_terms = items
    with st.spinner("🛍️ Consulting Product Reviews Agent..."):
        matches = search_product_reviews_items(items, exclude_uris=st.session_state.shown_sample_uris)
        
        reasoning_data = {
            "intent_detected": "Fuzzy Search Entity Match",
            "kg_nodes_accessed": ["schema:Product", "schema:name", "schema:image"],
            "ontology_mapping": "eo:UserPreference",
            "narrative": f"Executing SPARQL CONTAINS query to map natural language entities to formal Knowledge Graph instances. Found: {[m['item'] for m in matches] if matches else 'None'}",
            "dynamic_trace": {
                "action": "fuzzy_search",
                "entities_extracted": ", ".join(items),
                "results_returned": ", ".join([m['item'] for m in matches]) if matches else "None Found"
            }
        }

        if matches:
            # HARDENING (2026-07-14): this used to auto-add every match straight to
            # the profile at a hardcoded 5-star rating, with no chance for the user
            # to actually rate them -- "I like hair products" would silently commit
            # 3 specific items as 5/5 before the user ever saw or agreed to them.
            # Now this just PRESENTS the matches (same rendering as
            # show_fresh_sample_items()) and lets the normal interview flow -- which
            # already understands "I like item 1 and 3" / per-item ratings against a
            # just-shown numbered list, see chat_with_interviewer()'s KNOWN ITEM
            # URIS section -- handle the actual rating and profile update.
            formatted_matches = []
            for i, match_obj in enumerate(matches):
                m_name = match_obj.get("item", "")
                m_img = match_obj.get("image", "")
                m_uri = match_obj.get("uri", "")
                if m_uri:
                    st.session_state.known_item_uris[m_name] = m_uri
                    st.session_state.shown_sample_uris.add(m_uri)
                img_html = render_item_card_html(m_name, m_img)
                formatted_matches.append(f"{i+1}. {img_html}")

            bullet_list = "\n\n".join(formatted_matches)
            msg = f"Here's what I found matching what you're looking for:\n\n{bullet_list}\n\nHow would you rate these (1-5 stars), or just tell me which ones you like?"
            st.session_state.step = "interview"
        else:
            msg = "I couldn't find exact matches for those items in my database. Could you try different names, or would you like to rate some random samples instead (type 'samples')?"
            st.session_state.step = "choose_method"

        st.session_state.messages.append({"role": "assistant", "content": msg, "reasoning": reasoning_data, "phase": "interview"})
        st.rerun()

def chat_with_interviewer(user_input, chat_history):
    model_name = st.session_state.get("selected_model", "gemini-3-flash-preview")
    role_prompt = get_orchestrator_system_instruction("interviewer", INTERVIEWER_TOOL_PROMPT_DEFAULT)
    model = genai.GenerativeModel(model_name, system_instruction=role_prompt)

    # Known items with an already-resolved KG URI (from samples shown or a prior
    # search), so the LLM can attach that URI to a history entry instead of only a
    # name -- this is what fixes items downstream showing up as "Could not find URI
    # for item" when the user refers to something shown earlier in the conversation.
    known_uris = st.session_state.get("known_item_uris", {})
    known_uris_block = (
        "\n".join(f'- "{name}" -> {uri}' for name, uri in known_uris.items())
        if known_uris else "(none known yet this session)"
    )

    task_prompt = f"""
    ### SYSTEM CONTEXT: Discovered Agent Capabilities (auto-discovered from the Knowledge Graph, informational only)
    {st.session_state.get("agent_capabilities_briefing", "Not yet discovered.")}
    This section is background only, so you understand what the wider system can do -- it
    does not change your task below or your output format in any way.

    ### CURRENT PROFILE STATE (JSON)
    {json.dumps(st.session_state.user_profile)}

    ### KNOWN ITEM URIS (items already shown to the user this session, with their real KG URI)
    {known_uris_block}
    If the user refers to one of these items (by name or by its list position, e.g. "item 1"),
    include its "uri" field in the matching add_history entry below. Only include a "uri" when
    you are confident it matches one of the items listed above -- never invent one.

    ### IMPORTANT INSTRUCTION: THINKING BLOCK
    You MUST always prefix your response with a technical reasoning block enclosed in <thinking> and </thinking> tags.
    Inside these tags, provide STRICTLY a VALID JSON OBJECT matching this exact schema:
    {{
        "intent_detected": "string (e.g., 'Extracting Preferences', 'Trigger Recommendation')",
        "narrative": "string (1-2 sentences explaining the logical step)"
    }}
    Note: this phase only updates the in-memory profile, it does not query the knowledge graph,
    so do not invent ontology nodes or SPARQL targets here.

    ### YOUR GOAL
    1. Extract ratings or sentiments from user text.
    2. Update the JSON profile.
    3. After each item you add, naturally ask a simple follow-up: does the user want
       to rate/add more items, or are they ready for their recommendation now?
       Ask this regardless of how many items are in 'history' -- even just one is
       enough to proceed if that's what the user wants. Never imply they need to
       rate a minimum number of items first, and never repeat this nudge to keep
       rating if the user has already indicated they don't want to.
    4. If the user says anything indicating they're satisfied, done, or don't want to
       rate more -- e.g. "that's enough", "just go with what I have", "recommend now",
       "I'm ready", "no more" -- you MUST treat that as ready, regardless of how few
       items are in 'history' (even just one). Do not push back or ask them to rate
       additional items first.
    5. IF the user confirms they are ready (per rule 4, or by answering "yes" to the
       follow-up in rule 3), UNDER NO CIRCUMSTANCES should you provide actual
       recommendations yourself. You do not have access to the dataset.
       Instead, you MUST set "intent_detected": "Trigger Recommendation" in the JSON and reply simply with "Great! I am sending your profile to the recommendation engine now."
    
    ### OUTPUT FORMAT
    Start with the <thinking> block, then provide your user-facing message.
    If updating profile, END your response with this JSON block:
    ```json_update
    {{
        "add_history": [{{"item_id": "Item_Name_Here", "rating": 5, "note": "User reason", "uri": "<only if known from KNOWN ITEM URIS above>"}}],
        "add_preference": ["CategoryName"],
        "add_avoid": ["CategoryName"]
    }}
    ```
    """
    gemini_history = [{"role": ("user" if msg["role"] == "user" else "model"), "parts": [msg.get("content", "")]} for msg in chat_history if msg.get("content")]
    chat = model.start_chat(history=gemini_history)
    response = chat.send_message(f"{user_input}\n\n[SYSTEM INSTRUCTION: {task_prompt}]")
    return response.text

def extract_thinking(text):
    reasoning_data = {}
    clean_text = text
    if "<thinking>" in text and "</thinking>" in text:
        parts = text.split("</thinking>")
        raw_thinking = parts[0].replace("<thinking>", "").strip()
        clean_text = parts[1].strip()

        json_match = re.search(r'{.*}', raw_thinking, re.DOTALL)
        if json_match:
            try:
                reasoning_data = json.loads(json_match.group(0))
            except:
                reasoning_data = {"intent_detected": "Processing", "narrative": raw_thinking}
        else:
            reasoning_data = {"intent_detected": "Processing", "narrative": raw_thinking}

    return reasoning_data, clean_text

def generate_flowchart(reasoning_dict, phase="interview"):
    """Generates a Graphviz DOT script for the 4-layer architecture.

    HARDENING (2026-07-08 fix): previously every zone was drawn in full every time --
    all three Zone 3 agents and all three Zone 4 databases were always rendered, even
    when only one or two were actually involved in this action. Beyond being cluttered
    and misleading ("why is the Explainer Agent shown when this was a plain profile
    update?"), it also caused a genuine Graphviz layout bug: an agent connected by an
    edge (forcing it to a specific rank) sitting in the same cluster as unconnected,
    unrelated sibling agents (with no rank constraint) made dot's cluster bounding box
    and the node's actual position disagree, so the node visually overflowed outside
    its zone's border.

    Fix: build the node/cluster set PER ACTION -- only the nodes actually referenced by
    this action's edges are ever declared, and a Zone cluster is only emitted at all if
    it ends up with at least one active node in it. A cluster with exactly the nodes
    that are actually rank-constrained by its own edges cannot suffer that overflow.
    """
    if not isinstance(reasoning_dict, dict):
        reasoning_dict = {}

    def safe_str(s, length=35):
        if not s: return "None"
        s = str(s).replace('"', '\"').replace('\n', ' ').replace('\r', '')
        return "\n".join(textwrap.wrap(s, width=length))

    intent = safe_str(reasoning_dict.get("intent_detected", "Processing"), 25)
    trace = reasoning_dict.get("dynamic_trace", {})
    action = trace.get("action", "profile_update")

    CANVAS_COLOR = "#12151c"
    PANEL_BORDER = "#5b6577"

    # ---- Node catalog: (zone_id, zone_label, dot declaration line) ----
    # Nothing here is drawn yet -- this is just the menu of nodes an action MAY use.
    NODE_CATALOG = {
        "User": ("Z1", "Zone 1: The User Layer",
                 ' User [label="User", shape=ellipse, fillcolor="#56b6c2", fontcolor="black"];'),
        "Orchestrator": ("Z2", "Zone 2: The Interface / Orchestration Layer",
                         f' Orchestrator [label="Orchestrator Agent\n(Intent: {intent})", fillcolor="#c678dd"];'),
        "ProductReviewsAgent": ("Z3", "Zone 3: The Multi-Agent System",
                                ' ProductReviewsAgent [label="Product Reviews Agent", fillcolor="#98c379", fontcolor="black"];'),
        "RecommenderAgent": ("Z3", "Zone 3: The Multi-Agent System",
                             ' RecommenderAgent [label="Recommender Agent", fillcolor="#98c379", fontcolor="black"];'),
        "ExplainerAgent": ("Z3", "Zone 3: The Multi-Agent System",
                           ' ExplainerAgent [label="Explainer Agent", fillcolor="#98c379", fontcolor="black"];'),
        "DB_ProductReviews": ("Z4", "Zone 4: The Knowledge Graph",
                              ' DB_ProductReviews [label="XAI_ProductReviews\n(Product & Review Data)", shape=cylinder, fillcolor="#e5c07b", fontcolor="black"];'),
        "DB_Algorithms": ("Z4", "Zone 4: The Knowledge Graph",
                          ' DB_Algorithms [label="XAI_RecommendationAlgorithms\n(Algorithm Workflows)", shape=cylinder, fillcolor="#e5c07b", fontcolor="black"];'),
        "DB_Explanations": ("Z4", "Zone 4: The Knowledge Graph",
                            ' DB_Explanations [label="XAI_InteractiveExplanations\n(Interaction History)", shape=cylinder, fillcolor="#e5c07b", fontcolor="black"];'),
        # The execution log is a knowledge graph now, not a file on disk, so it is
        # drawn as a fourth dataset. The Recommender writes a pko:ProcedureExecution
        # to it on every run and the Explainer reads that run back, instead of
        # parsing prose out of a .log file.
        "DB_Executions": ("Z4", "Zone 4: The Knowledge Graph",
                          ' DB_Executions [label="XAI_ExecutionLogs\n(Recorded Runs)", shape=cylinder, fillcolor="#e5c07b", fontcolor="black"];'),
        "RecLLM": (None, None,
                   ' RecLLM [label="LLM Reasoning\n(Gemini calls)", shape=box, style="rounded,filled,dashed", fillcolor="#7a5ca8"];'),
    }
    ZONE_ORDER = ["Z1", "Z2", "Z3", "Z4"]  # top-to-bottom draw order when present

    # ---- Per-action: which nodes are ACTUALLY involved, the edges between them, and
    # whether the dashed-vs-solid Legend is worth showing (only when a dashed/LLM-only
    # edge actually appears in this action). ----
    active_nodes, edges, show_legend = [], [], False

    if action == "fuzzy_search":
        items = safe_str(trace.get("entities_extracted"))
        results = safe_str(trace.get("results_returned"))
        active_nodes = ["User", "Orchestrator", "ProductReviewsAgent", "DB_ProductReviews", "DB_Explanations"]
        edges = [
            f'User -> Orchestrator [label=" 1. Search Request\n(Input: {items})"];',
            f'Orchestrator -> ProductReviewsAgent [label=" 2. Looks up matching products,\nreturns match list", dir=both];',
            f'ProductReviewsAgent -> DB_ProductReviews [label=" 3. SPARQL CONTAINS search\n(Found: {results})"];',
            f'Orchestrator -> DB_Explanations [label=" 4. Writes Interaction\n(Appends to Profile)"];',
        ]

    elif action == "profile_update":
        # NOTE: building the profile from chat (extracting ratings/preferences) only
        # updates st.session_state.user_profile in memory -- it does NOT write to
        # Fuseki. The only real writes happen later, at the feedback step
        # (save_session_to_fuseki) and after an explanation (save_explanation_to_fuseki).
        active_nodes = ["User", "Orchestrator"]
        edges = [f'User -> Orchestrator [label=" 1. Chat message\n(builds profile in memory)"];']
        if "Trigger Recommendation" in intent:
            active_nodes.append("RecommenderAgent")
            edges.append('Orchestrator -> RecommenderAgent [label=" 2. Profile complete -\nhands off for a recommendation"];')

    elif action == "recommendation_loop":
        strategy = safe_str(trace.get("strategy", "Unknown Strategy"))
        attempts = trace.get("attempts", 1)
        results_cnt = len(trace.get("results", []))
        # This sequence matches EO_Recommender_Agent.py's /api/autonomous-recommend route:
        # get_available_recipes() -> select_best_algorithm() -> get_algorithm_source()
        # -> generate_dynamic_script() -> execute_code() (retried up to 3x on failure).
        # The two LLM calls (select a strategy, then write the script) are drawn as a
        # dedicated reasoning node rather than a self-loop, to keep the flow readable.
        # The two separate reads from DB_Algorithms (algorithm menu, then that algorithm's
        # step chain) are combined into one edge since they go to the same database.
        active_nodes = ["User", "Orchestrator", "RecommenderAgent", "DB_Algorithms",
                        "RecLLM", "DB_ProductReviews", "DB_Executions"]
        edges = [
            f'User -> Orchestrator [label=" 1. Sends Profile\n(JSON payload)"];',
            f'Orchestrator -> RecommenderAgent [label=" 2. POST /autonomous-recommend"];',
            f'RecommenderAgent -> DB_Algorithms [label=" 3. Reads Algorithm Menu, then the\ndeclared steps of the chosen workflow\n(inputs, outputs, parameters)"];',
            f'RecommenderAgent -> RecLLM [label=" 4. Picks a strategy, writes +\nruns a Python script\n(Chosen: {strategy}; {attempts} attempt(s))"];',
            f'RecommenderAgent -> DB_ProductReviews [label=" 5. (inside the script)\nfetches user/item/rating data"];',
            f'RecommenderAgent -> DB_Executions [label=" 6. Records the run\n(pko:ProcedureExecution +\none pko:StepExecution per step)"];',
            f'RecommenderAgent -> Orchestrator [label=" 7. Returns result\n({results_cnt} items + execution id)"];',
            f'Orchestrator -> User [label=" 8. Shows recommendation\n(history written later,\nwhen you rate it)", style=dashed];',
        ]
        show_legend = True

    elif action == "explanation_generation":
        style = safe_str(trace.get("style", "General"))
        user_query = safe_str(trace.get("user_query", "Ask Why/How"))
        strategy = safe_str(trace.get("strategy_used", "Recommender Logs"))
        # Matches EO_Explainer_Agent.py's /api/explain route: fetch_item_context() (via
        # the ProductReviews Agent), fetch_function_context(), then
        # fetch_execution_trace() -- which queries XAI_ExecutionLogs by execution id --
        # and fetch_explanation_types(), before the final Gemini call.
        #
        # Step 5 used to be drawn as a self-loop reading execution_trace_*.log. That is
        # no longer where the evidence comes from: the trace is a set of
        # pko:StepExecution instances, and the log file is only the fallback when the
        # graph holds no record of the run.
        active_nodes = ["User", "Orchestrator", "ExplainerAgent", "DB_ProductReviews",
                        "DB_Algorithms", "DB_Executions", "DB_Explanations"]
        edges = [
            f'User -> Orchestrator [label=" 1. Asks a Question\n({user_query})"];',
            f'Orchestrator -> ExplainerAgent [label=" 2. POST /explain\n(with the execution id)"];',
            f'ExplainerAgent -> DB_ProductReviews [label=" 3. Reads Item Details\n(via ProductReviews Agent)"];',
            f'ExplainerAgent -> DB_Algorithms [label=" 4. Reads the declared steps\n(p-plan:Step chain for\n{strategy})"];',
            f'ExplainerAgent -> DB_Executions [label=" 5. Reads what actually ran\n(pko:StepExecution, prov:used,\np-plan:correspondsToVariable)"];',
            f'ExplainerAgent -> DB_Explanations [label=" 6. Reads Explanation Styles\n(ep:Explanation types)"];',
            f'ExplainerAgent -> Orchestrator [label=" 7. Returns Explanation\n(Style:\n{style})"];',
            f'Orchestrator -> DB_Explanations [label=" 8. Writes Question + Explanation"];',
        ]
        show_legend = True

    else:
        active_nodes = ["Orchestrator", "DB_Explanations"]
        edges = [f'Orchestrator -> DB_Explanations [label=" Processing Intent\n({intent})"];']

    # ---- Emit the DOT script: only zones/nodes that ended up active are ever declared ----
    dot = [
        'digraph G {',
        'rankdir=TB;',
        'splines=polyline;',
        'nodesep=0.9;',
        'ranksep=1.3;',
        'pad=0.4;',
        'margin=0.2;',
        'bgcolor="#12151c";',
        'compound=true;',
        'newrank=true;',
        'node [fontname="Helvetica,Arial,sans-serif", fontsize=10, shape=box, style="rounded,filled", fontcolor="#ffffff", margin="0.3,0.2", width=0, height=0];',
        'edge [fontname="Helvetica,Arial,sans-serif", fontsize=9, color="#9aa5b8", fontcolor="#c7ceda", penwidth=1.2];',
    ]

    zones = {}
    for node_id in active_nodes:
        zone_id, zone_label, decl = NODE_CATALOG[node_id]
        if zone_id is None:
            continue  # standalone node (e.g. RecLLM), drawn outside any zone cluster
        zones.setdefault(zone_id, {"label": zone_label, "decls": []})["decls"].append(decl)

    for zone_id in ZONE_ORDER:
        if zone_id not in zones:
            continue  # zone has no active nodes this turn -- omit it entirely
        zone = zones[zone_id]
        dot.append(f'subgraph cluster_{zone_id} {{')
        dot.append(
            f' label="{zone["label"]}"; style=filled; fillcolor="{CANVAS_COLOR}"; '
            f'color="{PANEL_BORDER}"; penwidth=1.6; fontcolor="#ffffff"; margin=20;'
        )
        dot.extend(zone["decls"])
        dot.append('}')

    # Standalone nodes (not inside any zone cluster), e.g. the LLM reasoning box.
    for node_id in active_nodes:
        zone_id, _, decl = NODE_CATALOG[node_id]
        if zone_id is None:
            dot.append(decl)

    dot.extend(edges)

    if show_legend:
        dot.extend([
            'subgraph cluster_Legend {',
            f' label="Legend"; style=filled; fillcolor="{CANVAS_COLOR}"; color="{PANEL_BORDER}"; penwidth=1.6; fontcolor="#ffffff"; fontsize=9; margin=20;',
            ' Legend_KG [label="Real KG read/write", shape=box, style="rounded,filled", fillcolor="#3b4252", fontsize=8, width=1.8, height=0.35];',
            ' Legend_LLM [label="LLM reasoning step\n(no graph access)", shape=box, style="rounded,filled,dashed", fillcolor="#7a5ca8", fontsize=8, width=1.8, height=0.35];',
            '}',
        ])

    dot.append("}")
    return "\n".join(dot)



def save_session_to_fuseki(rating):
    timestamp = datetime.datetime.now().isoformat()
    profile_json = json.dumps(st.session_state.user_profile).replace('"', '\"').replace('\n', ' ')
    rec_json = json.dumps(st.session_state.recommendation_result).replace('"', '\"').replace('\n', ' ')
    user_uri = f"http://linked.aub.edu.lb/kgenxai/amazon/user/{st.session_state.user_profile['user_id']}"
    rec_uri = st.session_state.current_rec_uri
    # eo:hasCharacteristic resolves to https://purl.org/heals/eo#hasCharacteristic,
    # which is undefined (defect 8). EO is published across TWO namespaces and
    # hasCharacteristic exists only under the http variant. Rather than bind a
    # second prefix and depend on an inconsistency in someone else's ontology,
    # this uses EO's own canonical pattern -- the restriction EO places on
    # eo:user models a characteristic as SIO_000008 ("has attribute") pointing
    # at a typed eo:UserCharacteristic that carries SIO_000300 ("has value").
    # That also lifts the profile out of a bare literal into a typed node that
    # can carry structure later without another migration.
    #
    # eo:isConsumerOf is not an EO term either (defect 5). EO restricts
    # eo:consumes to (user -> ep:Explanation), and this triple links a user to
    # a SystemRecommendation, so substituting the term alone would trade an
    # undefined property for a range violation. eo:isUsedBy says what is meant
    # and has no declared range to violate.
    # ex:hasCharacteristic replaces SIO_000008 and is declared a subproperty of
    # it in ontology/kgenxai.ttl, so EO's axiom on eo:user is still satisfied
    # under reasoning while the endpoint stays readable. The rating uses
    # ex:userRating rather than a second mls:hasValue, because two hasValue
    # triples on one node with different meanings would be ambiguous.
    char_uri = f"{user_uri}/characteristic"
    prologue = sparql_prefixes('rdfs', 'xsd', 'eo', 'ep', 'mls', 'prov', 'ex')
    sparql_update = f"""{prologue}
INSERT DATA {{
    <{user_uri}> a eo:user ;
        ex:hasCharacteristic <{char_uri}> .
    <{char_uri}> a eo:UserCharacteristic ;
        rdfs:label "User profile" ;
        mls:hasValue "{profile_json}" .
    <{rec_uri}> a eo:SystemRecommendation ;
        mls:hasValue "{rec_json}" ;
        ex:userRating "{rating}"^^xsd:integer ;
        prov:generatedAtTime "{timestamp}"^^xsd:dateTime .
    <{rec_uri}> eo:isUsedBy <{user_uri}> .
}}"""
    try:
        requests.post(INTERACTIVE_EXPLANATIONS_UPDATE, data={'update': sparql_update}, auth=FUSEKI_AUTH, timeout=30)
        return True
    except Exception as e:
        log_error("XAI_InteractiveExplanations", f"Failed to save rating/session: {e}")
        return False

def save_explanation_to_fuseki(user_query, explanation_data):
    timestamp = datetime.datetime.now().isoformat()
    user_uri = f"http://linked.aub.edu.lb/kgenxai/amazon/user/{st.session_state.user_profile['user_id']}"
    rec_uri = st.session_state.current_rec_uri
    question_uri = f"http://linked.aub.edu.lb/kgenxai/amazon/question/{uuid.uuid4().hex[:8]}"
    expl_uri = f"http://linked.aub.edu.lb/kgenxai/amazon/explanation/{uuid.uuid4().hex[:8]}"
    expl_text = str(explanation_data.get('explanation_text', '')).replace('"', '\"').replace('\n', ' ')
    # The class used to be derived by stripping non-letters from the display
    # label and appending "Explanation". That happened to work for "Trace-Based"
    # and silently produced undefined terms for others: "Scientific" became
    # eo:ScientificExplanation (defect 3c -- EO spells it scientificExplanation,
    # the one lower-case subclass) and "Safety and Performance" became
    # eo:SafetyandPerformanceExplanation. Guessing a URI from a label is the
    # root cause of most of the sixteen defects, so the mapping is now an
    # explicit table in kgenxai_config, every entry checked against EO.
    #
    # An unrecognised style yields None and the resource is typed ep:Explanation
    # alone -- correct, and honest: it IS an explanation, we simply do not claim
    # a subtype we cannot substantiate.
    style_class = explanation_class_for(explanation_data.get('selected_style'))
    type_clause = "ep:Explanation" + (f", <{style_class}>" if style_class else "")
    escaped_query = user_query.replace('"', '\"')
    # eo:consumes is added here rather than in save_session_to_fuseki because
    # EO restricts it to (user -> ep:Explanation): this is the point at which an
    # explanation actually exists to be consumed.
    prologue = sparql_prefixes('rdfs', 'xsd', 'eo', 'ep', 'mls', 'prov', 'schema')
    execution_clause = ""
    if st.session_state.get("current_execution_id"):
        execution_clause = (
            f'    <{expl_uri}> prov:wasDerivedFrom '
            f'<{NS_INSTANCE["execution"]}'
            f'{st.session_state.current_execution_id}> .\n')
    sparql_update = f"""{prologue}
INSERT DATA {{
    <{user_uri}> eo:asks <{question_uri}> .
    <{question_uri}> a schema:Question ;
        mls:hasValue "{escaped_query}" ;
        prov:generatedAtTime "{timestamp}"^^xsd:dateTime .
    <{expl_uri}> a {type_clause} ;
        eo:addresses <{question_uri}> ;
        ep:isBasedOn <{rec_uri}> ;
        mls:hasValue "{expl_text}" ;
        prov:generatedAtTime "{timestamp}"^^xsd:dateTime .
{execution_clause}    <{user_uri}> eo:consumes <{expl_uri}> .
}}"""
    try:
        requests.post(INTERACTIVE_EXPLANATIONS_UPDATE, data={'update': sparql_update}, auth=FUSEKI_AUTH, timeout=30)
        return True
    except Exception as e:
        log_error("XAI_InteractiveExplanations", f"Failed to save explanation: {e}")
        return False

def report_chat_error(source, user_message, exception=None, phase="interview"):
    """Logs an error to the System Console AND posts a visible assistant message in
    the main chat. Per 2026-07-08 feedback: a caught exception must never just vanish
    behind a transient st.error() banner (which disappears on the next rerun) while
    the conversation itself sits frozen -- that is exactly what looked like "the
    chatbot isn't replying." Every step that talks to Gemini or an agent now calls
    this on failure instead, so the user always gets a chat message plus a rerun."""
    detail = f"{user_message}: {exception}" if exception is not None else user_message
    log_error(source, detail)
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            f"⚠️ {user_message}. You can try again -- if it keeps happening, check "
            f"System Admin > Console for technical details."
        ),
        "phase": phase,
    })

def reset_session_state():
    st.session_state.user_profile = {
        "user_id": f"User_{uuid.uuid4().hex[:8]}",
        "history": [], "preferences": {"preferred_categories": [], "avoid_categories": []}
    }
    st.session_state.messages = []
    st.session_state.step = "choose_method"
    st.session_state.current_rec_uri = f"http://linked.aub.edu.lb/kgenxai/amazon/recommendation/{uuid.uuid4().hex[:8]}"
    st.session_state.current_execution_id = None
    st.session_state.generated_custom_script = None
    st.session_state.known_item_uris = {}
    st.session_state.shown_sample_uris = set()
    st.session_state.last_search_terms = None
    st.rerun()

def render_settings_view():
    """Full-page Admin > Settings view (Agent Setup / Configuration / Product
    Reviews Data / Recommendation Algorithms / Interactive Explanations). Rendered
    INSTEAD OF the chat (see the view router below) rather than appended below it,
    so switching into Settings no longer also re-renders the entire chat history,
    sidebar capability discovery, etc. on every interaction -- that double-render
    was both the "opens within the chat" confusion and the main slowness cause.
    Console now lives in its own render_console_view() instead of a 6th tab here.
    """
    st.header("\u2699\ufe0f Semantic Infrastructure Settings")
    st.caption("Manage the XAI_ProductReviews, XAI_RecommendationAlgorithms, and XAI_InteractiveExplanations datasets residing in your Fuseki Triple Store.")

    if st.button("\u2b05\ufe0f Return to Chat Interaction", key="settings_return_to_chat"):
        st.session_state.active_view = "chat"
        st.rerun()

    st.markdown("---")
    # HARDENING (2026-07-14): st.tabs() looks like only the visible tab's content is
    # "loaded", but Streamlit actually executes EVERY tab's Python body on every
    # single rerun regardless of which one is visually selected -- tabs are a
    # purely visual/CSS construct, not lazy. That's what caused "it sometimes loads
    # the whole settings in ALL tabs" (all 5 sections' queries/forms were always
    # running together) and could occasionally show a tab-label/content mismatch
    # during a rerun. Using a radio button as the section picker instead means only
    # the SELECTED section's code path executes at all -- genuinely lazy.
    settings_section_options = [
        "\U0001f9e9 Agent Setup",
        "\u2699\ufe0f Configuration",
        "\U0001f4ca Product Reviews Data (XAI_ProductReviews)",
        "\U0001f9e0 Recommendation Algorithms (XAI_RecommendationAlgorithms)",
        "\U0001f5e3\ufe0f Interactive Explanations (XAI_InteractiveExplanations)",
    ]
    selected_settings_section = st.radio(
        "Settings section", settings_section_options, horizontal=True,
        label_visibility="collapsed", key="settings_section_radio",
    )
    st.markdown("---")

    # ---------------- TAB: CONFIGURATION (Fuseki base URL + per-dataset names) ----------------
    if selected_settings_section == "\u2699\ufe0f Configuration":
        if DEMO_MODE:
            st.warning(f"⚠️ {DEMO_MODE_MESSAGE}")

        st.subheader("Fuseki Server")
        new_fuseki_base = st.text_input(
            "Fuseki Base URL",
            value=st.session_state.fuseki_base,
            help="Base URL of the Apache Jena Fuseki server hosting all three datasets below.",
            disabled=DEMO_MODE,
        )

        st.subheader("Dataset Names")
        st.caption(
            "All three datasets live on the Fuseki server above. Each name defaults to the "
            "project's convention below, but can be changed if your Fuseki instance names "
            "them differently -- the code builds each dataset's endpoints from the base URL "
            "and the name you set here."
        )

        names = st.session_state.dataset_names
        with st.form("dataset_config_form"):
            pr_name = st.text_input(
                "XAI_ProductReviews -- dataset name",
                value=names.get("product_reviews", "XAI_ProductReviews"),
                help="Sent to the ProductReviews Agent (and, via it, the Recommender/Explainer agents) on every request.",
                disabled=DEMO_MODE,
            )
            algo_name = st.text_input(
                "XAI_RecommendationAlgorithms -- dataset name",
                value=names.get("algorithms", "XAI_RecommendationAlgorithms"),
                help="Used to build this dataset's /query and /update endpoints.",
                disabled=DEMO_MODE,
            )
            expl_name = st.text_input(
                "XAI_InteractiveExplanations -- dataset name",
                value=names.get("explanations", "XAI_InteractiveExplanations"),
                help="Used to build this dataset's /query and /update endpoints.",
                disabled=DEMO_MODE,
            )

            col1, col2 = st.columns([2, 1])
            with col1:
                if st.form_submit_button("💾 Save Configuration", type="primary", disabled=DEMO_MODE):
                    if DEMO_MODE:
                        st.warning(f"⚠️ {DEMO_MODE_MESSAGE}")
                    else:
                        st.session_state.fuseki_base = (new_fuseki_base.strip().rstrip("/")
                                                         or "https://linked.aub.edu.lb:8080/fuseki")
                        st.session_state.dataset_names = {
                            "product_reviews": pr_name.strip() or "XAI_ProductReviews",
                            "algorithms": algo_name.strip() or "XAI_RecommendationAlgorithms",
                            "explanations": expl_name.strip() or "XAI_InteractiveExplanations",
                        }
                        st.success("Saved. Endpoints below now reflect your configuration.")
                        st.rerun()
            with col2:
                if st.form_submit_button("↩️ Reset to Defaults", disabled=DEMO_MODE):
                    if DEMO_MODE:
                        st.warning(f"⚠️ {DEMO_MODE_MESSAGE}")
                    else:
                        st.session_state.dataset_names = {
                            "product_reviews": "XAI_ProductReviews",
                            "algorithms": "XAI_RecommendationAlgorithms",
                            "explanations": "XAI_InteractiveExplanations",
                        }
                        st.rerun()

        st.markdown("###### Resolved endpoints currently in effect")
        st.code(
            f"Fuseki base:                           {FUSEKI_BASE}\n"
            f"XAI_ProductReviews dataset name:        {PRODUCT_REVIEWS_DATASET_NAME}\n"
            f"XAI_RecommendationAlgorithms /query:    {RECOMMENDATION_ALGORITHMS_ENDPOINT}\n"
            f"XAI_RecommendationAlgorithms /update:   {RECOMMENDATION_ALGORITHMS_UPDATE}\n"
            f"XAI_InteractiveExplanations /query:     {INTERACTIVE_EXPLANATIONS_ENDPOINT}\n"
            f"XAI_InteractiveExplanations /update:    {INTERACTIVE_EXPLANATIONS_UPDATE}",
            language="text",
        )


    # ---------------- TAB 1: PRODUCT REVIEWS DATA MANAGEMENT (XAI_ProductReviews) ----------------
    elif selected_settings_section == "\U0001f4ca Product Reviews Data (XAI_ProductReviews)":
        st.subheader("Product Reviews Data Store Control (XAI_ProductReviews)")

        if DEMO_MODE:
            st.warning(f"⚠️ {DEMO_MODE_MESSAGE}")

        # Show current triple count
        triple_count = get_product_reviews_triple_count()
        st.metric("Current Triples in Graph", f"{triple_count:,}")

        # Clear XAI_ProductReviews data button
        # NOTE: the batched delete-all logic used to live here, looping over individual
        # Fuseki update calls directly. That batching now happens server-side inside
        # EO_ProductReviews_Agent.py's /api/clear_graph -- this is the only thing in the
        # Orchestrator function that touches the XAI_ProductReviews graph's deletion, and it does so over
        # HTTP to the agent, never straight to Fuseki.
        if st.button("🗑️ Clear XAI_ProductReviews Graph", type="primary", disabled=DEMO_MODE):
            if triple_count == 0:
                st.info("XAI_ProductReviews graph is already empty!")
            else:
                with st.spinner(f"Deleting {triple_count:,} triples via the ProductReviews Agent (this may take a moment for large graphs)..."):
                    success, result = clear_product_reviews_via_agent()

                if success:
                    deleted = result.get("deleted", 0) if isinstance(result, dict) else 0
                    st.success(f"XAI_ProductReviews graph cleared successfully! ({deleted:,} triples deleted)")
                    st.rerun()
                else:
                    st.error(f"Failed to clear XAI_ProductReviews graph: {result}")

        st.markdown("---")
        st.subheader("📥 Ingest Amazon 2023 Dataset")
        st.info("Select a dataset from the Amazon Reviews 2023 collection to download and ingest automatically.")

        # Dataset selection dropdown with details
        dataset_options = list(AMAZON_DATASETS.keys())
        selected_dataset = st.selectbox(
            "Select Amazon Dataset to Ingest:",
            dataset_options,
            format_func=lambda x: f"{x} ({AMAZON_DATASETS[x]['size']}, {AMAZON_DATASETS[x]['num_reviews']} reviews)",
            disabled=DEMO_MODE,
        )

        if selected_dataset:
            dataset_info = AMAZON_DATASETS[selected_dataset]
            is_poc_dataset = dataset_info.get("proof_of_concept", False)

            st.info(f"""
            Dataset Details:
            Category: {selected_dataset}
            Reviews File: {dataset_info['size']} ({dataset_info['num_reviews']} reviews)
            Metadata File: {dataset_info['meta_size']}
            Reviews URL: {dataset_info['url']}
            Metadata URL: {dataset_info['meta_url']}
            """)

            if not is_poc_dataset:
                st.warning(
                    f"⚠️ **Experimental / Under Development.** This dataset is "
                    f"{dataset_info['size']} and has not been validated for live, one-click "
                    f"ingestion -- download time and server load are unpredictable. "
                    f"Live ingestion is currently limited to datasets under "
                    f"{PROOF_OF_CONCEPT_SIZE_LIMIT_MB}MB as a proof of concept. "
                    f"You can still preview a sample of this dataset below."
                )

            col1, col2 = st.columns([3, 1])
            with col1:
                ingest_clicked = st.button(
                    f"📥 Download and Ingest {selected_dataset}",
                    type="primary",
                    disabled=DEMO_MODE or not is_poc_dataset,
                    help=(
                        "Disabled: this feature is disabled for this demo version." if DEMO_MODE
                        else (None if is_poc_dataset else "Disabled: dataset exceeds the proof-of-concept size limit for live ingestion.")
                    ),
                )

                # Track multi-step confirmation state across reruns using session_state,
                # keyed per-dataset. A bare nested `st.button(...)` inside this `if` block
                # would never fire: Streamlit reruns the whole script on every click, so a
                # second button defined and checked within the same run as the first click
                # always evaluates False on that run, instantly tripping st.stop().
                confirm_key = f"confirm_ingest_{selected_dataset}"

                if ingest_clicked:
                    if get_product_reviews_triple_count() > 0:
                        st.session_state[confirm_key] = "pending"
                    else:
                        st.session_state[confirm_key] = "confirmed"

                if st.session_state.get(confirm_key) == "pending":
                    st.warning("⚠️ The XAI_ProductReviews graph already contains data. Continuing will append to existing data.")
                    confirm_col, cancel_col = st.columns(2)
                    with confirm_col:
                        if st.button("⚠️ Confirm Ingest Anyway", key=f"{confirm_key}_yes", disabled=DEMO_MODE):
                            st.session_state[confirm_key] = "confirmed"
                            st.rerun()
                    with cancel_col:
                        if st.button("✖️ Cancel", key=f"{confirm_key}_no", disabled=DEMO_MODE):
                            st.session_state[confirm_key] = None
                            st.rerun()

                if st.session_state.get(confirm_key) == "confirmed":
                    st.session_state[confirm_key] = None  # consume the confirmation so a later rerun doesn't re-trigger it

                    status_text = st.empty()
                    progress_bar = st.progress(0)

                    with st.spinner(f"Downloading and ingesting {selected_dataset}..."):
                        status_text.text("Starting ingestion process...")
                        success, message = ingest_amazon_dataset(selected_dataset, dataset_info, progress_bar, status_text)

                        if success:
                            st.success(message)
                            st.info(f"Total triples in graph: {get_product_reviews_triple_count():,}")

                            # Show sample of ingested triples
                            with st.expander("📋 Sample Ingested Triples", expanded=True):
                                sample_triples = get_product_reviews_sample_triples(20)
                                display_triple_sample(sample_triples)

                            st.rerun()
                        else:
                            st.error(f"❌ Ingestion failed: {message}")

            with col2:
                if st.button("📋 View Sample Data", help="Preview first 5 records from the selected dataset", disabled=DEMO_MODE):
                    try:
                        sample_response = requests.get(dataset_info['url'], stream=True, timeout=30, verify=False)
                        sample_response.raise_for_status()

                        if dataset_info['url'].endswith('.gz'):
                            gz_file = gzip.GzipFile(fileobj=sample_response.raw)
                            text_file = io.TextIOWrapper(gz_file, encoding='utf-8')
                            sample_lines = []
                            for i in range(5):
                                line = text_file.readline()
                                if not line:
                                    break
                                try:
                                    sample_lines.append(json.loads(line.strip()))
                                except:
                                    sample_lines.append({"raw": line[:200] + "..."})
                        else:
                            sample_lines = []
                            for i, line in enumerate(sample_response.iter_lines()):
                                if i >= 5:
                                    break
                                if line:
                                    try:
                                        sample_lines.append(json.loads(line.decode('utf-8')))
                                    except:
                                        sample_lines.append({"raw": line[:200] + "..."})

                        if sample_lines:
                            st.json(sample_lines)
                        else:
                            st.info("No sample data retrieved.")
                    except Exception as e:
                        st.error(f"Failed to fetch sample: {e}")

        st.markdown("---")
        st.subheader("📤 Upload Custom Dataset")
        st.warning(
            "⚠️ **Experimental / Under Development.** This feature relies on an LLM to "
            "write a custom ingestion script for whatever file you provide. It has not "
            "been validated across all supported formats and may fail unpredictably. "
            "Enable it explicitly below if you want to try it."
        )
        custom_ingestion_enabled = st.checkbox(
            "Enable custom dataset upload (experimental)",
            value=CUSTOM_INGESTION_ENABLED_BY_DEFAULT,
            key="custom_ingestion_enabled_toggle",
            disabled=DEMO_MODE,
        )

        if custom_ingestion_enabled:
            st.info(
                f"Upload your own dataset in any format (max {MAX_CUSTOM_UPLOAD_MB}MB). "
                f"The system will use AI to generate a custom ingestion script."
            )

            uploaded_file = st.file_uploader(
                "Choose a dataset file",
                type=['jsonl', 'json', 'csv', 'tsv', 'xlsx', 'xls', 'pdf'],
                help=f"Supported formats: JSONL, JSON, CSV, TSV, Excel, PDF. Max size: {MAX_CUSTOM_UPLOAD_MB}MB.",
                disabled=DEMO_MODE,
            )

            if uploaded_file is not None:
                file_size_mb = uploaded_file.size / (1024 * 1024)
                st.info(f"📎 File: {uploaded_file.name} ({file_size_mb:.2f} MB)")

                if file_size_mb > MAX_CUSTOM_UPLOAD_MB:
                    st.error(
                        f"❌ This file is {file_size_mb:.1f}MB, which exceeds the "
                        f"{MAX_CUSTOM_UPLOAD_MB}MB limit for custom uploads. Please upload "
                        f"a smaller file, or split it into smaller chunks."
                    )
                else:
                    if st.button("🔍 Preview Custom Data", help="Show a sample of the uploaded data", disabled=DEMO_MODE):
                        try:
                            if uploaded_file.name.endswith('.jsonl'):
                                content = uploaded_file.getvalue().decode('utf-8')
                                lines = content.split('\n')[:5]
                                preview_data = []
                                for line in lines:
                                    if line.strip():
                                        try:
                                            preview_data.append(json.loads(line))
                                        except:
                                            preview_data.append({"raw": line[:200] + "..."})
                                st.json(preview_data)
                            elif uploaded_file.name.endswith('.json'):
                                content = json.loads(uploaded_file.getvalue().decode('utf-8'))
                                if isinstance(content, list):
                                    st.json(content[:5])
                                else:
                                    st.json(content)
                            elif uploaded_file.name.endswith('.csv'):
                                df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
                                st.dataframe(df.head())
                            elif uploaded_file.name.endswith('.xlsx'):
                                df = pd.read_excel(io.BytesIO(uploaded_file.getvalue()))
                                st.dataframe(df.head())
                            else:
                                st.info("Preview not available for this file type.")
                        except Exception as e:
                            st.error(f"Failed to preview file: {e}")

                    ingest_custom_clicked = st.button("📥 Ingest Custom Dataset", type="primary", disabled=DEMO_MODE)
                    custom_confirm_key = "confirm_ingest_custom_dataset"

                    if ingest_custom_clicked:
                        if get_product_reviews_triple_count() > 0:
                            st.session_state[custom_confirm_key] = "pending"
                        else:
                            st.session_state[custom_confirm_key] = "confirmed"

                    if st.session_state.get(custom_confirm_key) == "pending":
                        st.warning("⚠️ The XAI_ProductReviews graph already contains data. Continuing will append to existing data.")
                        confirm_col, cancel_col = st.columns(2)
                        with confirm_col:
                            if st.button("⚠️ Confirm Ingest Anyway", key=f"{custom_confirm_key}_yes", disabled=DEMO_MODE):
                                st.session_state[custom_confirm_key] = "confirmed"
                                st.rerun()
                        with cancel_col:
                            if st.button("✖️ Cancel", key=f"{custom_confirm_key}_no", disabled=DEMO_MODE):
                                st.session_state[custom_confirm_key] = None
                                st.rerun()

                    if st.session_state.get(custom_confirm_key) == "confirmed":
                        st.session_state[custom_confirm_key] = None

                        status_text = st.empty()
                        progress_bar = st.progress(0)

                        with st.spinner("Processing custom dataset..."):
                            success, message = ingest_custom_dataset(uploaded_file, progress_bar, status_text)

                            if success:
                                st.success(message)
                                st.info(f"Total triples in graph: {get_product_reviews_triple_count():,}")

                                with st.expander("📋 Sample Ingested Triples", expanded=True):
                                    sample_triples = get_product_reviews_sample_triples(20)
                                    display_triple_sample(sample_triples)

                                st.rerun()
                            else:
                                st.error(f"❌ Custom ingestion failed: {message}")

                    # Display dynamically generated custom ingestion script if available in session state
                    if "generated_custom_script" in st.session_state and st.session_state.generated_custom_script:
                        st.markdown("### 📄 Generated Custom Ingestion Script")
                        st.caption("This script was generated dynamically by the Gemini orchestrator to map your custom schema to the AUB KGenXAI product review ontology.")
                        st.code(st.session_state.generated_custom_script, language="python")

    # ---------------- TAB 2: RECOMMENDATION ALGORITHMS (XAI_RecommendationAlgorithms) ----------------

    # ---------------- TAB 2: RECOMMENDATION ALGORITHMS (XAI_RecommendationAlgorithms) ----------------
    elif selected_settings_section == "\U0001f9e0 Recommendation Algorithms (XAI_RecommendationAlgorithms)":
        if DEMO_MODE:
            st.warning(f"⚠️ {DEMO_MODE_MESSAGE}")

        st.subheader("Algorithm Workflows & Manuals (XAI_RecommendationAlgorithms)")
        if st.button("🔄 Refresh Algorithm List", key="refresh_algo_list"):
            # The list below is cached for 15s for performance -- if you just ran an
            # update script against this dataset (e.g. reduce_to_3_algorithms.py,
            # fix_association_rule_mining_steps.py), this forces an immediate
            # re-fetch instead of waiting out the cache window.
            get_algorithm_functions.clear()
            st.rerun()

        # Fetch existing functions
        funcs = get_algorithm_functions()
        func_list = [{
            "uri": f["uri"]["value"],
            "name": f["name"]["value"],
            "spec": f.get("spec", {}).get("value", ""),
            "workflow_uri": f.get("workflow", {}).get("value"),
            "spec_node_uri": f.get("specNode", {}).get("value"),
        } for f in funcs]

        if func_list:
            st.info(f"Found {len(func_list)} algorithm functions in the graph.")

        func_options = ["-- Create New Algorithm --"] + [f["name"] for f in func_list]
        # Browsing which algorithm to VIEW is read-only and harmless -- only editing
        # (below) is blocked in demo mode. Disabling this selectbox too would trap
        # everyone on index 0 ("-- Create New Algorithm --", i.e. blank fields),
        # making it impossible to even look at an existing algorithm's steps/spec.
        selected_func_name = st.selectbox("Select Algorithm to Edit:", func_options)

        if selected_func_name == "-- Create New Algorithm --":
            f_uri = f"http://linked.aub.edu.lb/kgenxai/Algorithm/new_{uuid.uuid4().hex[:6]}"
            f_name = ""
            f_spec = ""
            f_workflow_uri = None
            f_spec_node_uri = None
        else:
            selected_f = next(f for f in func_list if f["name"] == selected_func_name)
            f_uri = selected_f["uri"]
            f_name = selected_f["name"]
            f_spec = selected_f["spec"]
            f_workflow_uri = selected_f["workflow_uri"]
            f_spec_node_uri = selected_f["spec_node_uri"]

        # Steps are read once, here, and used both to size the editor and to
        # prefill it. The count control has to live OUTSIDE st.form: widgets
        # inside a form do not trigger a rerun until the form is submitted, so
        # a number_input in there would appear to do nothing when changed.
        step_records = []
        if selected_func_name != "-- Create New Algorithm --":
            step_records = get_algorithm_steps(f_uri)

        st.markdown("### Algorithm Steps")
        st.caption(
            "Each step mirrors how the procedure is represented in the knowledge "
            "graph: a description, the variables it consumes and produces, its "
            "parameters, and any constraint on the generated code. Editing a step "
            "keeps its existing URI and its declared variables, so nothing is "
            "relocated or orphaned."
        )
        step_count = st.number_input(
            "Number of steps",
            min_value=1, max_value=12,
            value=max(1, min(12, len(step_records) if step_records else 3)),
            step=1, disabled=DEMO_MODE,
            key=f"step_count_{selected_func_name}",
            help="Reducing this drops the trailing steps when you save.",
        )

        with st.form("function_form"):
            new_f_name = st.text_input("Algorithm Name", value=f_name, disabled=DEMO_MODE)
            new_f_spec = st.text_area(
                "Agent Usage Spec (LLM Instructions & Context)",
                value=f_spec,
                height=150,
                help="This is the manual that guides the LLM on when and how to use this algorithm.",
                disabled=DEMO_MODE,
            )

            def _record(index):
                return step_records[index] if index < len(step_records) else {}

            # Widget keys are scoped to the selected algorithm and to the step
            # itself. Streamlit gives session_state precedence over value= once
            # a keyed widget exists, so a key shared across algorithms keeps
            # whatever was in the box the first time it rendered. The page opens
            # on "-- Create New Algorithm --", which stored empty strings under
            # step_desc_0 and friends -- and every algorithm selected afterwards
            # then showed those empty strings instead of its own data.
            def _key(field, index, step_uri):
                scope = step_uri or f"{selected_func_name}#{index}"
                # A deterministic digest rather than hash(): Python randomises
                # string hashing per process, so a widget key built from hash()
                # would change on every restart.
                digest = hashlib.md5(scope.encode("utf-8")).hexdigest()[:10]
                return f"algstep_{field}_{digest}"

            steps = []
            for idx in range(int(step_count)):
                rec = _record(idx)
                existing_uri = rec.get("uri")
                label = f"Step {idx + 1}"
                if rec.get("description") or rec.get("comment", {}).get("value"):
                    preview = (rec.get("description")
                               or rec.get("comment", {}).get("value", ""))
                    label = f"Step {idx + 1} — {preview[:70]}"

                with st.expander(label, expanded=(idx == 0)):
                    if existing_uri:
                        st.caption(f"URI: {existing_uri}")

                    s_desc = st.text_area(
                        "Description (dcterms:description)",
                        value=rec.get("description", ""),
                        height=80, key=_key("desc", idx, existing_uri), disabled=DEMO_MODE,
                        help="What the step does, in prose meant for a reader.",
                    )
                    s_comment = st.text_area(
                        "Instruction to the code generator (rdfs:comment)",
                        value=rec.get("comment", {}).get("value", ""),
                        height=80, key=_key("comment", idx, existing_uri), disabled=DEMO_MODE,
                        help="Retained for backward compatibility. If the "
                             "description is empty this is what the composer sees.",
                    )
                    col_in, col_out = st.columns(2)
                    with col_in:
                        s_inputs = st.text_input(
                            "Inputs (p-plan:hasInputVar)",
                            value=", ".join(rec.get("inputs", [])),
                            key=_key("in", idx, existing_uri), disabled=DEMO_MODE,
                            help="Comma-separated variable labels.",
                        )
                    with col_out:
                        s_outputs = st.text_input(
                            "Outputs (p-plan:hasOutputVar)",
                            value=", ".join(rec.get("outputs", [])),
                            key=_key("out", idx, existing_uri), disabled=DEMO_MODE,
                            help="Comma-separated variable labels.",
                        )
                    col_p, col_f = st.columns(2)
                    with col_p:
                        s_params = st.text_input(
                            "Parameters (mls:hasHyperParameter)",
                            value=", ".join(rec.get("parameters", [])),
                            key=_key("param", idx, existing_uri), disabled=DEMO_MODE,
                            help="Comma-separated, as name=value. "
                                 "Example: N=10, similarity floor=low",
                        )
                    with col_f:
                        s_function = st.text_input(
                            "Operation (pko:requiresFunction)",
                            value=rec.get("function", ""),
                            key=_key("fn", idx, existing_uri), disabled=DEMO_MODE,
                            help="Example: cosine similarity",
                        )
                    s_constraint = st.text_area(
                        "Constraint on generated code (ex:generationConstraint)",
                        value=rec.get("constraint", ""),
                        height=68, key=_key("constraint", idx, existing_uri), disabled=DEMO_MODE,
                        help="Instructions addressed to the code generator, kept "
                             "out of rdfs:comment and out of the reused ontologies.",
                    )

                steps.append({
                    "uri": existing_uri,
                    "description": s_desc,
                    "comment": s_comment,
                    "inputs": [x.strip() for x in s_inputs.split(",") if x.strip()],
                    "outputs": [x.strip() for x in s_outputs.split(",") if x.strip()],
                    "parameters": [x.strip() for x in s_params.split(",") if x.strip()],
                    "function": s_function,
                    "constraint": s_constraint,
                })

            col1, col2 = st.columns([2, 1])
            with col1:
                if st.form_submit_button("💾 Save Algorithm Function", type="primary", disabled=DEMO_MODE):
                    if DEMO_MODE:
                        st.warning(f"⚠️ {DEMO_MODE_MESSAGE}")
                    elif not new_f_name.strip():
                        st.error("Algorithm name is required.")
                    else:
                        success, message = save_algorithm_function(
                            f_uri, new_f_name, new_f_spec, steps,
                            workflow_uri=f_workflow_uri, spec_node_uri=f_spec_node_uri
                        )
                        if success:
                            get_algorithm_functions.clear()
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(f"Failed to save: {message}")

            with col2:
                if selected_func_name != "-- Create New Algorithm --":
                    if st.form_submit_button("🗑️ Delete Algorithm", type="secondary", disabled=DEMO_MODE):
                        if DEMO_MODE:
                            st.warning(f"⚠️ {DEMO_MODE_MESSAGE}")
                        elif delete_algorithm_function(f_uri, workflow_uri=f_workflow_uri, spec_node_uri=f_spec_node_uri):
                            get_algorithm_functions.clear()
                            st.success(f"Deleted algorithm: {selected_func_name}")
                            st.rerun()
                        else:
                            st.error("Failed to delete algorithm.")

    # ---------------- TAB 3: INTERACTIVE EXPLANATIONS ----------------
    elif selected_settings_section == "\U0001f5e3\ufe0f Interactive Explanations (XAI_InteractiveExplanations)":
        if DEMO_MODE:
            st.warning(f"⚠️ {DEMO_MODE_MESSAGE}")

        st.subheader("Ontology Explanation Styles (XAI_InteractiveExplanations)")
        if st.button("🔄 Refresh Explanation Types", key="refresh_expl_types"):
            get_explanation_types.clear()
            st.rerun()

        expls = get_explanation_types()
        expl_list = [{"uri": e["uri"]["value"], "name": e["name"]["value"], "desc": e.get("desc", {}).get("value", ""), "questions": e.get("questions", {}).get("value", ""), "action": e.get("action", {}).get("value", "")} for e in expls]

        if expl_list:
            st.info(f"Found {len(expl_list)} explanation types in the graph.")

        expl_options = ["-- Create New Explanation Type --"] + [e["name"] for e in expl_list]
        # Same reasoning as the algorithm selectbox above: browsing is read-only.
        selected_expl_name = st.selectbox("Select Explanation Type to Edit:", expl_options)

        if selected_expl_name == "-- Create New Explanation Type --":
            # Custom styles used to be minted inside EO's namespace as
            # eo:Custom<hex>Explanation (defect 14). Coining terms in a third
            # party's namespace is not permissible regardless of intent, so
            # these now live in our own instance space. Migration 02 rewrites
            # any that already exist. The style NAME shown in the UI is
            # unchanged; only the identifier moves.
            e_uri = custom_explanation_uri(f"Custom{uuid.uuid4().hex[:6]}")
            e_name = ""
            e_desc = ""
            e_questions = ""
            e_action = ""
        else:
            selected_e = next(e for e in expl_list if e["name"] == selected_expl_name)
            e_uri = selected_e["uri"]
            e_name = selected_e["name"]
            e_desc = selected_e["desc"]
            e_questions = selected_e["questions"]
            e_action = selected_e["action"]

        with st.form("explanation_form"):
            new_e_name = st.text_input("Explanation Type Name", value=e_name, disabled=DEMO_MODE)
            new_e_desc = st.text_area("Ontological Description", value=e_desc, height=100, disabled=DEMO_MODE)
            new_e_questions = st.text_area("Example Questions", value=e_questions, height=100, disabled=DEMO_MODE)
            new_e_action = st.text_area("LLM Execution Action", value=e_action, height=100, disabled=DEMO_MODE)

            col1, col2 = st.columns([2, 1])
            with col1:
                if st.form_submit_button("💾 Save Explanation Type", type="primary", disabled=DEMO_MODE):
                    if DEMO_MODE:
                        st.warning(f"⚠️ {DEMO_MODE_MESSAGE}")
                    elif not new_e_name.strip():
                        st.error("Explanation type name is required.")
                    else:
                        success, message = save_explanation_type(e_uri, new_e_name, new_e_desc, new_e_questions, new_e_action)
                        if success:
                            get_explanation_types.clear()
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(f"Failed to save: {message}")

            with col2:
                if selected_expl_name != "-- Create New Explanation Type --":
                    if st.form_submit_button("🗑️ Delete Explanation Type", type="secondary", disabled=DEMO_MODE):
                        if DEMO_MODE:
                            st.warning(f"⚠️ {DEMO_MODE_MESSAGE}")
                        elif delete_explanation_type(e_uri):
                            get_explanation_types.clear()
                            st.success(f"Deleted explanation type: {selected_expl_name}")
                            st.rerun()
                        else:
                            st.error("Failed to delete explanation type.")

    # ---------------- TAB 4: AGENT SETUP (System Prompt + Tool Prompts, per agent) ----------------
    elif selected_settings_section == "\U0001f9e9 Agent Setup":
        st.subheader("Agent Setup -- Prompt Hierarchy per Agent")
        st.caption(
            "Every agent has exactly one System Prompt (its overall role -- the parent) "
            "and one or more Tool Prompts (what it does for a specific task -- the "
            "children, which inherit that System Prompt). Saving applies immediately -- "
            "every subsequent request uses the new text."
        )

        # ---- At-a-glance tree diagram (all four agents, same shape) ----
        tree_lines = []
        for node in AGENT_PROMPT_TREE:
            tree_lines.append(f"{node['icon']} {node['agent']}")
            tree_lines.append(f"  └─ 🧩 System Prompt  (parent)")
            for i, tool_key in enumerate(node["tool_keys"]):
                branch = "└─" if i == len(node["tool_keys"]) - 1 else "├─"
                tool_title = AGENT_ROLE_LABELS[tool_key].split(" -- ", 1)[-1]
                tree_lines.append(f"       {branch} {tool_title}  (tool prompt)")
            tree_lines.append("")
        st.code("\n".join(tree_lines).rstrip(), language=None)

        def render_agent_role_form(agent_key, indent=False):
            """One editable Identity/Tool Prompt form, shared by every agent below so
            the save/reset/demo-mode logic is defined exactly once."""
            container = st.container()
            with container:
                if indent:
                    _, col = st.columns([0.06, 0.94])
                else:
                    col = st.container()
                with col:
                    with st.expander(AGENT_ROLE_LABELS[agent_key], expanded=False):
                        current_value = get_agent_role(agent_key, AGENT_ROLE_DEFAULTS.get(agent_key, ""))
                        with st.form(f"agent_role_form_{agent_key}"):
                            new_value = st.text_area(
                                "Role / System Prompt",
                                value=current_value,
                                height=120,
                                key=f"agent_role_textarea_{agent_key}",
                                # The "Role / System Prompt" label only makes sense on the
                                # top-level System Prompt form -- showing it again on every
                                # indented Tool Prompt below was confusing (it made a Tool
                                # Prompt's own text box look like it was also "the System
                                # Prompt"). The expander title above already names exactly
                                # what this box is, so just collapse the label for those.
                                label_visibility="collapsed" if indent else "visible",
                            )
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                if st.form_submit_button("💾 Save", type="primary", key=f"agent_role_save_{agent_key}"):
                                    if DEMO_MODE:
                                        st.warning(f"⚠️ {DEMO_MODE_MESSAGE}")
                                    elif not new_value.strip():
                                        st.error("Role text cannot be empty.")
                                    else:
                                        success, message = save_agent_definition(agent_key, new_value)
                                        if success:
                                            st.session_state.agent_roles[agent_key] = new_value
                                            st.success(message)
                                            st.rerun()
                                        else:
                                            st.error(f"Failed to save: {message}")
                            with col2:
                                if st.form_submit_button("↩️ Reset to Default", key=f"agent_role_reset_{agent_key}"):
                                    if DEMO_MODE:
                                        st.warning(f"⚠️ {DEMO_MODE_MESSAGE}")
                                    else:
                                        default_value = AGENT_ROLE_DEFAULTS.get(agent_key, "")
                                        success, message = save_agent_definition(agent_key, default_value)
                                        if success:
                                            st.session_state.agent_roles[agent_key] = default_value
                                            st.rerun()
                                        else:
                                            st.error(f"Failed to reset: {message}")

        # ---- Editable hierarchy: one block per agent, System Prompt then its indented Tool Prompts ----
        for node in AGENT_PROMPT_TREE:
            st.markdown(f"#### {node['icon']} {node['agent']}")
            render_agent_role_form(node["system_prompt_key"], indent=False)
            st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;**Tool Prompts** *(inherit the System Prompt above)*", unsafe_allow_html=True)
            for tool_key in node["tool_keys"]:
                render_agent_role_form(tool_key, indent=True)


def render_console_view():
    """Full-page System Console view (in-app errors + live agent log tail).
    Opened directly from its own sidebar icon/button -- separate from Settings --
    so logs can be checked in real time without going through the Settings tabs.
    """
    st.header("\U0001f5a5\ufe0f System Console")

    if st.button("\u2b05\ufe0f Return to Chat Interaction", key="console_return_to_chat"):
        st.session_state.active_view = "chat"
        st.rerun()

    st.markdown("---")
    st.subheader("System Console")
    st.caption("In-app errors and background agent output -- so a failure shows up here, not only in a terminal.")

    app_errors = st.session_state.get("error_log", [])
    if app_errors:
        st.markdown("**Recent in-app errors:**")
        for entry in reversed(app_errors[-30:]):
            st.markdown(
                f"<span style='color:#e05252;'>🔴 [{entry['time']}] "
                f"<b>{html.escape(entry['source'])}</b>: {html.escape(entry['message'])}</span>",
                unsafe_allow_html=True,
            )
        if st.button("Clear error log", key="clear_error_log_btn_tab"):
            st.session_state.error_log = []
            st.rerun()
    else:
        st.caption("No in-app errors logged this session. ✅")

    st.markdown("---")
    chosen_log = st.selectbox("Agent log to view", list(AGENT_LOG_FILES.keys()), key="console_log_selector_tab")
    tail_text = read_agent_log_tail(chosen_log)
    highlighted_lines = []
    for line in tail_text.splitlines():
        safe_line = html.escape(line)
        if "ERROR" in line or "CRITICAL" in line:
            highlighted_lines.append(f"<span style='color:#e05252;'>{safe_line}</span>")
        elif "WARNING" in line:
            highlighted_lines.append(f"<span style='color:#e5a04e;'>{safe_line}</span>")
        else:
            highlighted_lines.append(safe_line)
    st.markdown(
        "<div style='max-height:400px; overflow-y:auto; font-family:monospace; "
        "font-size:11px; white-space:pre-wrap; background:#12151c; color:#c7ceda; "
        f"padding:8px; border-radius:6px;'>{'<br>'.join(highlighted_lines) if highlighted_lines else '<i>No log output yet.</i>'}</div>",
        unsafe_allow_html=True,
    )
    if st.button("🔄 Refresh logs", key="refresh_console_logs_tab"):
        st.rerun()

# ================= VIEW ROUTER (2026-07-13 addition) =================
# Settings and Console are now separate full-page views instead of being appended
# below the chat on the same page -- only the active view's code runs on any given
# rerun, instead of the chat AND every admin tab all re-executing together.
if st.session_state.active_view == "settings":
    render_settings_view()
    st.stop()
elif st.session_state.active_view == "console":
    render_console_view()
    st.stop()

# ================= MAIN UI =================
# Top-right reset icon (2026-07-23 addition): sits next to the header so it's
# visible on every chat step (interview, feedback, ready_to_fetch,
# post_recommendation_chat) -- this block runs on every rerun of the "chat"
# view, before any step-specific branching below, rather than only being
# reachable from one particular step. A lightweight inline confirm guards
# against an accidental click now that it's always on-screen, but otherwise
# calls the exact same reset_session_state() the existing bottom-of-chat
# "Start New Session" button already uses -- no new reset logic, no behavior
# change to what a reset actually does.
header_col, reset_col = st.columns([0.94, 0.06])
with header_col:
    st.title("KGenXAI Demo: Explainable Product Recommendations")
    st.subheader("🤖 Orchestrator AI Agent")
with reset_col:
    st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)
    if st.button("↺", key="top_right_reset_btn", help="Reset the conversation and start a new session"):
        st.session_state.confirm_top_reset = True
        st.rerun()

if st.session_state.get("confirm_top_reset"):
    st.warning("Reset the conversation and start a brand new session? This clears your current profile and chat history.")
    confirm_col, cancel_col = st.columns(2)
    with confirm_col:
        if st.button("✅ Yes, reset", key="confirm_top_reset_yes"):
            st.session_state.confirm_top_reset = False
            reset_session_state()
    with cancel_col:
        if st.button("✖️ Cancel", key="confirm_top_reset_no"):
            st.session_state.confirm_top_reset = False
            st.rerun()

if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = []
    with st.spinner("Initializing Knowledge Graph Connection via ProductReviews Agent..."):
        st.session_state.product_reviews_topic = fetch_product_reviews_topic()
        greeting = f"Hello and welcome to the recommender system demo! I see our domain knowledge graph primarily contains {st.session_state.product_reviews_topic} products.\n\nHow would you like to build your profile today?\n1. Rate Random Samples\n2. Specify Your Favorites (e.g., I like hair products.)"
        st.session_state.messages.append({"role": "assistant", "content": greeting, "phase": "interview"})
    # Discovery layer (2026-07 addition): auto-run once per session, alongside the
    # existing greeting setup above. Purely additive -- does not affect greeting,
    # step routing, or anything else in this block.
    with st.spinner("Discovering agent capabilities from the Knowledge Graph..."):
        st.session_state.agent_capabilities_briefing = build_agent_capabilities_briefing()
    # Load any KG-edited agent role/system-prompt overrides (Admin > Agent Setup tab)
    # once per session, so per-message chat calls don't each re-query Fuseki.
    load_agent_roles_into_session()

# ---- Sidebar: discovered agent capabilities (2026-07 addition, read-only/informational) ----
with st.sidebar:
    with st.expander("🔎 Discovered Agent Capabilities", expanded=False):
        st.caption("Auto-discovered from the Knowledge Graph at session start -- this is what actually drives the system prompt below, not a hardcoded list.")
        st.markdown(
            st.session_state.get(
                "agent_capabilities_briefing",
                "_Not discovered yet -- will populate once the chat session initializes._",
            )
        )
        if st.button("🔄 Re-discover now"):
            with st.spinner("Re-querying agent capabilities..."):
                st.session_state.agent_capabilities_briefing = build_agent_capabilities_briefing()
            st.rerun()

# NOTE: the System Console (in-app errors + background agent logs) used to live here
# in the sidebar. Per 2026-07-08 feedback it now lives under System Admin > Console
# (see the "🖥️ Console" tab below), alongside the other admin/settings tabs.

# --- DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("reasoning"):
            with st.expander("👁️ System Logic & Transparent Data Flow", expanded=False):
                reasoning = msg["reasoning"]
                phase = msg.get("phase", "interview")

                if isinstance(reasoning, dict):
                    trace = reasoning.get("dynamic_trace", {})
                    action = trace.get("action")

                    # ---- Top row: what the agent decided to do, and why ----
                    st.markdown(f"**Intent:** {reasoning.get('intent_detected', 'Processing')}")
                    if reasoning.get("narrative"):
                        st.caption(reasoning["narrative"])

                    # ---- Pipeline diagram: matches the real code path for this action ----
                    # HARDENING: a plain profile_update turn with no recommendation
                    # trigger has no real KG/agent interaction to show -- just
                    # "User -> Orchestrator" and nothing else -- which read as a
                    # broken/incomplete chart in testing. Skip the diagram entirely
                    # for that specific case; the intent/narrative text above already
                    # communicates what happened.
                    is_trivial_profile_update = (
                        action == "profile_update"
                        and "Trigger Recommendation" not in str(reasoning.get("intent_detected", ""))
                    )
                    if not is_trivial_profile_update:
                        st.markdown("###### 📊 Pipeline Trace")
                        flowchart = generate_flowchart(reasoning, phase)
                        st.graphviz_chart(flowchart)

                    # ---- Knowledge graph nodes actually involved this turn ----
                    nodes = reasoning.get("kg_nodes_accessed", [])
                    if nodes:
                        st.markdown("###### 🗄️ Knowledge graph terms involved")
                        st.markdown("\n".join(f"- `{n}`" for n in nodes))
                    else:
                        st.caption("No knowledge graph access this turn -- only the in-memory profile was updated.")

                    # ---- Real exchanged data, per phase, pulled straight from the agents'
                    # own JSON responses (not re-described by an LLM) ----
                    if action == "recommendation_loop":
                        st.markdown("###### 🔁 What the Recommender Agent actually returned")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Strategy chosen", str(trace.get("strategy", "Unknown")))
                        c2.metric("Attempts taken", trace.get("attempts", 1))
                        c3.metric("Items returned", len(trace.get("results", [])))
                        if trace.get("saved_script"):
                            st.caption(f"Generated script saved as: `{trace['saved_script']}`")
                        if trace.get("execution_logs"):
                            with st.expander("📜 Raw execution log (from the generated script)", expanded=False):
                                st.code(trace["execution_logs"], language="text")

                    elif action == "explanation_generation":
                        st.markdown("###### 💬 What the Explainer Agent actually returned")
                        c1, c2 = st.columns(2)
                        c1.metric("Explanation style", str(trace.get("style", "General")))
                        c2.metric("Strategy being explained", str(trace.get("strategy_used", "Unknown")))
                        if trace.get("reason"):
                            st.caption(f"Why this style was chosen: {trace['reason']}")

                    elif action == "fuzzy_search":
                        st.markdown("###### 🔎 What the search actually matched")
                        st.caption(f"Searched for: {trace.get('entities_extracted', 'N/A')}")
                        st.caption(f"Found in the graph: {trace.get('results_returned', 'None')}")
                else:
                    st.caption(reasoning)

        if msg.get("content"):
            st.markdown(msg["content"], unsafe_allow_html=True)

# --- 2. CHOOSE PROFILE METHOD PHASE ---
if st.session_state.step == "choose_method":
    if user_input := st.chat_input(f"Ex: 'I want to rate random items' or 'I like specific {st.session_state.product_reviews_topic}'"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input, unsafe_allow_html=True)

        with st.spinner("Analyzing your choice..."):
            try:
                classification = classify_initial_intent(user_input)
                intent = classification.get("intent")
                extracted_items = classification.get("items", [])
                log_info(f"choose_method: classified intent='{intent}' items={extracted_items}")

                if intent == "CHOOSE_SAMPLES":
                    show_fresh_sample_items("Great! Please rate the following items (1-5 stars) or tell me which ones you like/dislike:")

                elif intent == "PROVIDE_ITEMS":
                    st.session_state.step = "search_specific"
                    msg = f"Awesome! Please type the names of the {st.session_state.product_reviews_topic} you already know and like."
                    st.session_state.messages.append({"role": "assistant", "content": msg, "phase": "interview"})
                    st.rerun()

                elif intent == "SEARCH_ITEMS":
                    process_search_items(extracted_items)

                else:
                    msg = "I didn't quite catch that. Do you want to rate random samples or tell me specific items you like?"
                    st.session_state.messages.append({"role": "assistant", "content": msg, "phase": "interview"})
                    st.rerun()
            except Exception as e:
                # HARDENING (2026-07-08 fix): previously this only showed a transient
                # st.error() banner with no chat message and no rerun -- so the user's
                # message just sat there with no reply, looking like the bot had frozen.
                report_chat_error("Orchestrator (choose_method)", "I couldn't process that choice", e)
                st.rerun()

# --- 3. SEARCH SPECIFIC ITEMS PHASE ---
elif st.session_state.step == "search_specific":
    if user_input := st.chat_input(f"Ex: 'I really love X and Y'"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input, unsafe_allow_html=True)
        with st.spinner("Extracting items..."):
            try:
                classification = classify_initial_intent(user_input)
                extracted_items = classification.get("items", [])
                if extracted_items: process_search_items(extracted_items)
                else:
                    msg = "I couldn't detect specific items. Try listing them clearly, or we can just rate some samples instead?"
                    st.session_state.messages.append({"role": "assistant", "content": msg, "phase": "interview"})
                    st.rerun()
            except Exception as e:
                report_chat_error("Orchestrator (search_specific)", "I couldn't extract items from that", e)
                st.rerun()

# --- 4. INTERVIEW PHASE ---
elif st.session_state.step == "interview":
    if user_input := st.chat_input("Ex: 'I like 1 and 3' or 'I am ready'"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input, unsafe_allow_html=True)

        # HARDENING (2026-07-14): classify intent before routing -- a request for
        # a different/fresh batch of items used to go straight to the free-form
        # interviewer LLM, which fabricated item names and broken image markup with
        # no real KG backing. See classify_interview_intent() for details.
        try:
            with st.spinner("Analyzing your message..."):
                interview_classification = classify_interview_intent(user_input)
            interview_intent = interview_classification.get("intent")
            interview_search_items = interview_classification.get("items", [])
            log_info(f"interview: classified intent='{interview_intent}' items={interview_search_items}")
        except Exception as e:
            log_error("Orchestrator (interview classifier)", f"Could not classify intent, defaulting to profile chat: {e}")
            interview_intent = "PROFILE_CHAT"
            interview_search_items = []

        if interview_intent == "MORE_SAMPLES":
            show_more_items("Understood. Let's try a different selection of items. Please rate these (1-5 stars) or let me know what you think:")
        elif interview_intent == "SEARCH_NEW_TOPIC" and interview_search_items:
            # HARDENING (2026-07-17): a brand new topic named mid-interview (e.g.
            # "I like sports" right after rating a batch of "creams" items) now
            # gets the same real KG search treatment as the original topic did in
            # search_specific, instead of being narrated as a soft preference by
            # the free-form interviewer with no actual item ever searched or shown.
            process_search_items(interview_search_items)
        else:
            with st.spinner("Updating Profile..."):
                try:
                    ai_response_text = chat_with_interviewer(user_input, st.session_state.messages[:-1])

                    reasoning_data, clean_text = extract_thinking(ai_response_text)

                    updates_summary = []

                    # Robust parsing that handles both codes and labels correctly
                    if "```json_update" in clean_text: 
                        parts = clean_text.split("```json_update")
                        clean_text = parts[0].strip()
                        for json_block in parts[1:]:
                            block_content = json_block.split("```")[0].strip()
                            try:
                                updates = json.loads(block_content)
                                updates_summary.append(str(updates))
                                current_time = int(datetime.datetime.now().timestamp())

                                if "add_history" in updates:
                                    for item in (updates["add_history"] if isinstance(updates["add_history"], list) else [updates["add_history"]]):
                                        item["timestamp"] = current_time
                                        st.session_state.user_profile["history"].append(item)
                                if "add_preference" in updates:
                                    pref_data = updates["add_preference"]
                                    st.session_state.user_profile["preferences"]["preferred_categories"].extend(pref_data if isinstance(pref_data, list) else [pref_data])
                                if "add_avoid" in updates:
                                    avoid_data = updates["add_avoid"]
                                    st.session_state.user_profile["preferences"]["avoid_categories"].extend(avoid_data if isinstance(avoid_data, list) else [avoid_data])
                            except json.JSONDecodeError: 
                                pass
                    elif "json_update" in clean_text:
                        parts = clean_text.split("json_update")
                        clean_text = parts[0].replace("```", "").strip()
                        for json_block in parts[1:]:
                            block_content = json_block.split("```")[0].strip()
                            try:
                                updates = json.loads(block_content)
                                updates_summary.append(str(updates))
                                current_time = int(datetime.datetime.now().timestamp())

                                if "add_history" in updates:
                                    for item in (updates["add_history"] if isinstance(updates["add_history"], list) else [updates["add_history"]]):
                                        item["timestamp"] = current_time
                                        st.session_state.user_profile["history"].append(item)
                                if "add_preference" in updates:
                                    pref_data = updates["add_preference"]
                                    st.session_state.user_profile["preferences"]["preferred_categories"].extend(pref_data if isinstance(pref_data, list) else [pref_data])
                                if "add_avoid" in updates:
                                    avoid_data = updates["add_avoid"]
                                    st.session_state.user_profile["preferences"]["avoid_categories"].extend(avoid_data if isinstance(avoid_data, list) else [avoid_data])
                            except json.JSONDecodeError: 
                                pass

                    if isinstance(reasoning_data, dict):
                        reasoning_data["dynamic_trace"] = {
                            "action": "profile_update",
                            "updates_applied": " | ".join(updates_summary) if updates_summary else "No changes"
                        }

                    is_ready = False
                    if isinstance(reasoning_data, dict) and reasoning_data.get("intent_detected") == "Trigger Recommendation":
                        is_ready = True
                    elif "sending your profile" in clean_text.lower() or "recommendation engine" in clean_text.lower():
                        is_ready = True

                    # HARDENING (2026-07-16): hard code-level guard, independent of
                    # prompt compliance -- never let a recommendation get triggered
                    # with zero actual rated items in history, even if the LLM
                    # decided the user was "ready" (e.g. after only setting a
                    # preferred_categories preference with no item ever
                    # successfully rated). The Recommender has nothing to seed a
                    # recommendation from in that case.
                    if is_ready and not st.session_state.user_profile.get("history"):
                        is_ready = False
                        clean_text = (
                            "I'd like to get you a recommendation, but I don't have any "
                            "rated items in your profile yet -- I need at least one to "
                            "work from. Want to rate a few samples, or tell me something "
                            "specific you like?"
                        )

                    st.session_state.messages.append({
                        "role": "assistant", "content": clean_text, "reasoning": reasoning_data, "phase": "interview"
                    })

                    if is_ready:
                        st.session_state.step = "ready_to_fetch"

                    st.rerun()
                except Exception as e:
                    # HARDENING (2026-07-08 fix): previously this whole block had NO
                    # try/except -- if the Gemini call raised (rate limit, safety block,
                    # transient network error, etc.), the script crashed with no chat
                    # message appended, which looked exactly like "the chatbot isn't
                    # replying," especially right after rating displayed sample items.
                    report_chat_error("Orchestrator (interview)", "I had trouble updating your profile from that", e)
                    st.rerun()

# --- 5. FETCHING PHASE ---
elif st.session_state.step == "ready_to_fetch":
    st.info("Your profile is complete! The system is ready to compute your recommendations.")
    with st.expander("Review your Profile before submitting"):
        st.json(st.session_state.user_profile)

    # HARDENING (2026-07-16): this phase used to be a dead end for adding more
    # items -- no chat_input at all, only the button below. Routes the same way
    # every other phase does: extract any named item(s) via the shared classifier
    # and hand off to process_search_items(), which presents them for rating and
    # drops the user back into the interview step to keep building their profile
    # before generating -- so "add an item" works from anywhere, not just the
    # interview phase itself.
    if extra_input := st.chat_input("Want to add another item first? Just tell me, or click Generate when ready."):
        st.session_state.messages.append({"role": "user", "content": extra_input})
        with st.chat_message("user"): st.markdown(extra_input, unsafe_allow_html=True)
        with st.spinner("Analyzing your message..."):
            try:
                classification = classify_initial_intent(extra_input)
                extracted_items = classification.get("items", [])
                if extracted_items:
                    process_search_items(extracted_items)
                elif classification.get("intent") == "CHOOSE_SAMPLES":
                    # 2026-07-16: a request like "show me more items to rate" here
                    # names nothing specific, so it wasn't being routed anywhere
                    # useful -- same gap as post_recommendation_chat's MORE_SAMPLES.
                    show_more_items("Sure! Here are some items to rate:")
                else:
                    msg = "I couldn't detect a specific item to add there. Try naming it again, or click 'Generate Recommendation Now' below when you're ready."
                    st.session_state.messages.append({"role": "assistant", "content": msg, "phase": "interview"})
                    st.rerun()
            except Exception as e:
                report_chat_error("Orchestrator (ready_to_fetch)", "I couldn't process that", e)
                st.rerun()

    if st.button("🚀 Generate Recommendation Now"):
        # HARDENING (2026-07-16): same guard as the interview-phase trigger, as
        # defense-in-depth in case this step is ever reached with an empty profile
        # through a different path.
        if not st.session_state.user_profile.get("history"):
            st.error("Your profile doesn't have any rated items yet -- add at least one before generating a recommendation.")
            st.stop()
        log_info("ready_to_fetch: sending profile to Recommender Agent")
        with st.status("🎯 Consulting Recommender Agent...", expanded=True) as status:
            try:
                payload = {
                    "user_profile": st.session_state.user_profile,
                    "api_key": st.session_state.api_key,
                    "selected_model": st.session_state.selected_model,
                    "fuseki_base": get_product_reviews_fuseki_base(),
                    "product_reviews_dataset": get_product_reviews_dataset_name(),
                    "algorithms_dataset": ALGORITHMS_DATASET_NAME,
                    # Agent Setup overrides (Admin tab) -- the Recommender Agent falls
                    # back to its own built-in defaults if these are None/absent.
                    "selector_system_prompt": get_agent_system_instruction(
                        "recommender_system_prompt", RECOMMENDER_SYSTEM_PROMPT_DEFAULT, "selector", SELECTOR_TOOL_PROMPT_DEFAULT
                    ),
                    "composer_system_prompt": get_agent_system_instruction(
                        "recommender_system_prompt", RECOMMENDER_SYSTEM_PROMPT_DEFAULT, "composer", COMPOSER_TOOL_PROMPT_DEFAULT
                    ),
                }
                response = requests.post(RECOMMENDER_API, json=payload, headers={'Content-Type': 'application/json'}, timeout=600)
                if response.status_code == 200:
                    raw_data = response.json()
                    status.update(label="✅ Recommendation Loop Complete", state="complete", expanded=False)

                    with st.spinner("Fetching thumbnails & formatting recommendation..."):
                        if 'results' in raw_data and isinstance(raw_data['results'], list):
                            # HARDENING: enrich_recommendations_with_images() now guarantees
                            # every item ends up with a non-empty `image` (real URL or the
                            # shared placeholder) -- see its docstring for the 3-pass lookup.
                            raw_data['results'] = enrich_recommendations_with_images(raw_data['results'])
                            # CONTENT SAFETY (2026-07-23): clean item names in place BEFORE
                            # raw_data is used for anything else below -- the formatter LLM
                            # prompt included the raw JSON verbatim and was told to "list the
                            # recommended items by name," so a flagged raw name previously
                            # still reached the user via that narrative text even when the
                            # gallery caption for the same item was already clean.
                            raw_data['results'] = sanitize_recommendation_results(raw_data['results'])

                        st.session_state.recommendation_result = raw_data
                        # Optional on both sides: an older Recommender that does
                        # not return this leaves it None, and the Explainer then
                        # falls back to the log files exactly as before.
                        st.session_state.current_execution_id = raw_data.get("execution_id")

                        model_name = st.session_state.get("selected_model", "gemini-3-flash-preview")
                        model = genai.GenerativeModel(
                            model_name,
                            system_instruction=get_orchestrator_system_instruction("formatter", FORMATTER_TOOL_PROMPT_DEFAULT),
                        )
                        fmt_prompt = f"""
Format this recommender output into a friendly response.
Raw Data: {json.dumps(raw_data)}
Requirements:

ALWAYS prefix your response with a <thinking>...</thinking> block.
Inside the tags, provide STRICTLY a VALID JSON OBJECT mapping:
{{
"intent_detected": "Algorithm Execution",
"narrative": "Explain WHY the algorithm/strategy was chosen based on the profile, in 1-2 sentences."
}}

List the recommended items by name in your narrative text.

Do NOT attempt to embed any HTML or image tags yourself -- a gallery image for every
returned item is appended automatically after your response, so just focus on the
explanatory narrative text.
"""
                        fmt_response = model.generate_content(fmt_prompt).text
                        fmt_reasoning, fmt_clean = extract_thinking(fmt_response)
                        # CONTENT SAFETY (2026-07-23): defense-in-depth -- raw_data['results']
                        # is already sanitized above, but this catches anything explicit/
                        # adult, sexual, racial, or gender-based slur-related the formatter
                        # LLM might still introduce on its own in freeform prose.
                        fmt_clean = sanitize_narrative_text(fmt_clean)

                        # HARDENING: never rely on the formatting LLM to remember to attach
                        # images -- deterministically append a guaranteed image gallery for
                        # every item actually returned, so "all items returned have their
                        # images next to them" holds regardless of what the LLM produced above.
                        gallery_html = build_image_gallery_html(raw_data.get('results', []))
                        if gallery_html:
                            fmt_clean = f"{fmt_clean}\n\n{gallery_html}"

                        # The fields below describe what ACTUALLY happened on the Recommender
                        # Agent's side (per its own JSON response), rather than letting the
                        # formatting LLM call guess at ontology nodes it has no way to verify.
                        # See get_available_recipes() -> select_best_algorithm() ->
                        # get_algorithm_source() -> generate_dynamic_script() -> execute_code()
                        # in EO_Recommender_Agent.py for the real sequence these values come from.
                        if isinstance(fmt_reasoning, dict):
                            fmt_reasoning["kg_nodes_accessed"] = [
                                "mls:Algorithm (XAI_RecommendationAlgorithms)",
                                "mls:ImplementationCharacteristic (Usage Manual)",
                                "p-plan:Step + hasInputVar / hasOutputVar + "
                                "mls:hasHyperParameter (XAI_RecommendationAlgorithms)",
                                "pko:ProcedureExecution (XAI_ExecutionLogs, written)",
                            ]
                            fmt_reasoning["ontology_mapping"] = "mls:Implementation"
                            fmt_reasoning["dynamic_trace"] = {
                                "action": "recommendation_loop",
                                "strategy": raw_data.get("strategy", "Unknown Strategy"),
                                "attempts": raw_data.get("attempts", 1),
                                "results": raw_data.get("results", []),
                                "saved_script": raw_data.get("saved_script"),
                                "execution_logs": raw_data.get("logs", ""),
                            }

                        st.session_state.messages.append({
                            "role": "assistant", "content": fmt_clean, "reasoning": fmt_reasoning, "phase": "recommendation"
                        })
                        log_info(f"ready_to_fetch: recommendation complete, strategy={raw_data.get('strategy')}, items={len(raw_data.get('results', []))}")
                        st.session_state.step = "feedback"
                        st.rerun()
                else:
                    status.update(label="❌ Agent Error", state="error")
                    # Same hardening as the Explainer branch: surface the Recommender's
                    # own reported error (it returns {"error"/"message": ...} on failure)
                    # plus the HTTP status, so the System Console shows the real cause.
                    try:
                        body = response.json()
                        detail = body.get("final_error") or body.get("message") or body.get("error") or response.text[:300]
                    except Exception:
                        body = {}
                        detail = response.text[:300]

                    if isinstance(body, dict) and body.get("no_results"):
                        # HARDENING (2026-07-14): a genuine "no qualifying candidates"
                        # result is not a system malfunction -- the reason string is
                        # short, human-readable, and safe to show directly (unlike a
                        # stack trace), so show it plainly instead of the generic
                        # "an error occurred, check the console" message.
                        report_chat_error(
                            "Recommender Agent",
                            f"I couldn't generate a recommendation: {detail}",
                            phase="recommendation",
                        )
                    else:
                        report_chat_error(
                            "Recommender Agent",
                            "The recommendation engine returned an error",
                            f"HTTP {response.status_code}: {detail}",
                            phase="recommendation",
                        )
                    st.rerun()
            except requests.exceptions.Timeout:
                status.update(label="❌ Request Timeout", state="error")
                report_chat_error("Recommender Agent", "The recommendation request timed out (the algorithm may be processing a large dataset, or the server may be slow)", phase="recommendation")
                st.rerun()
            except Exception as e:
                status.update(label="❌ Connection Failed", state="error")
                report_chat_error("Recommender Agent", "I couldn't connect to the recommendation engine", e, phase="recommendation")
                st.rerun()

# --- 6. FEEDBACK PHASE ---
elif st.session_state.step == "feedback":
    st.markdown("---")
    st.write("Please rate these recommendations before we continue:")
    rating = st.slider("Rate this recommendation (1-5)", 1, 5, 3)
    if st.button("Submit Rating"):
        if save_session_to_fuseki(rating):
            st.success("Rating Saved to XAI_InteractiveExplanations Data!")
            st.session_state.step = "post_recommendation_chat"
            neutral_msg = "Your feedback has been saved. I'm still here if you'd like to chat further or ask why these items were recommended!"
            st.session_state.messages.append({"role": "assistant", "content": neutral_msg, "phase": "explanation"})
            st.rerun()

    # HARDENING (2026-07-17): rating used to be a mandatory gate -- asking for an
    # explanation or adding another item both only worked AFTER clicking "Submit
    # Rating" (which moves to post_recommendation_chat, the only phase with real
    # explanation capability). This phase's chat input previously only understood
    # named items (via classify_initial_intent), so asking "why was X recommended?"
    # here just returned "I couldn't detect a specific item to add there." Now uses
    # the exact same 3-way classifier and handlers as post_recommendation_chat, so
    # explanations, adding items, and requesting more samples all work whether or
    # not the user has rated anything yet.
    if extra_input := st.chat_input("Ex: 'how does this work?', 'I also like X', or 'show me more items'"):
        st.session_state.messages.append({"role": "user", "content": extra_input})
        with st.chat_message("user"): st.markdown(extra_input, unsafe_allow_html=True)
        try:
            with st.spinner("Analyzing your message..."):
                classification = classify_post_recommendation_intent(extra_input)
            intent = classification.get("intent")
            items = classification.get("items", [])
            log_info(f"feedback: classified intent='{intent}' items={items}")
        except Exception as e:
            log_error("Orchestrator (feedback classifier)", f"Could not classify intent, defaulting to explanation: {e}")
            intent, items = "ASK_EXPLANATION", []

        if intent == "ADD_PREFERENCE" and items:
            process_search_items(items)
        elif intent == "MORE_SAMPLES":
            show_more_items("Sure! Here are some more items to rate:")
        else:
            consult_explainer_agent(extra_input)

# --- 7. POST-RECOMMENDATION CHAT PHASE ---
elif st.session_state.step == "post_recommendation_chat":
    if user_input := st.chat_input("Ex: 'how does this work?' or 'Why does item X make sense?'"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input, unsafe_allow_html=True)

        # HARDENING (2026-07-14): classify intent before routing -- a request to
        # add more items to the profile used to be sent straight to the Explainer
        # Agent unconditionally, which has no way to act on it (it can only
        # describe the existing recommendation). See
        # classify_post_recommendation_intent() for details.
        try:
            with st.spinner("Analyzing your message..."):
                post_rec_classification = classify_post_recommendation_intent(user_input)
            post_rec_intent = post_rec_classification.get("intent")
            post_rec_items = post_rec_classification.get("items", [])
            log_info(f"post_recommendation_chat: classified intent='{post_rec_intent}' items={post_rec_items}")
        except Exception as e:
            log_error("Orchestrator (post_recommendation_chat classifier)", f"Could not classify intent, defaulting to explanation: {e}")
            post_rec_intent, post_rec_items = "ASK_EXPLANATION", []

        if post_rec_intent == "ADD_PREFERENCE" and post_rec_items:
            # Reuses the exact same profile-update path as the interview phase --
            # adds the item(s) to the profile, shows them to the user, and moves
            # the step back to "interview" so they can say "I'm ready" to
            # regenerate recommendations with the updated profile.
            process_search_items(post_rec_items)
        elif post_rec_intent == "MORE_SAMPLES":
            # HARDENING (2026-07-16): "give me more products to rate" names no
            # specific item, so it used to have nowhere to go but ASK_EXPLANATION
            # -- getting answered by the Explainer Agent instead of actually
            # pulling fresh items. Reuses the exact same real, KG-backed,
            # never-repeats-an-item path the interview phase uses -- and stays
            # on-topic if the user had been searching a specific topic earlier.
            show_more_items("Sure! Here are some more items to rate -- this'll help refine your profile further:")
        else:
            consult_explainer_agent(user_input)

    st.markdown("---")
    if st.button("Start New Session", key="restart_btn_chat"): reset_session_state()


# ================= ADMIN SIDEBAR =================
with st.sidebar:
    st.markdown("### \u2699\ufe0f System Admin")
    if st.button("\u2699\ufe0f Configure KGs / Agent Settings", use_container_width=True, key="open_settings_btn"):
        st.session_state.active_view = "settings"
        st.rerun()
    if st.button("\U0001f5a5\ufe0f Console (live logs)", use_container_width=True, key="open_console_btn"):
        st.session_state.active_view = "console"
        st.rerun()

