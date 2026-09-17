from flask import Flask, request, jsonify
import requests
import google.generativeai as genai
import logging
import traceback
import re
import random
import hashlib
import time

# ================= CONFIGURATION & LOGGING =================
app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | PRODUCT_REVIEWS_AGENT | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Target the XAI_ProductReviews dataset (the product/review "domain" graph).
# DEFAULT_FUSEKI_BASE is used whenever a request doesn't specify its own 'fuseki_base'
# (e.g. calls from older clients, or manual testing). DEFAULT_DATASET_NAME is used
# whenever a request doesn't specify its own 'product_reviews_dataset' -- the
# Orchestrator's Admin > Configuration tab lets users override the dataset's NAME
# (not the server) per-session by sending 'product_reviews_dataset' on each request.
DEFAULT_FUSEKI_BASE = "https://linked.aub.edu.lb:8080/fuseki"
DEFAULT_DATASET_NAME = "XAI_ProductReviews"
# Was the literal ("username", "aubfuseki"). Credentials now come from the
# environment via kgenxai_config (FUSEKI_USER / FUSEKI_PASSWORD), which is
# what makes this file publishable. Rotate the password: it has been in the
# source and the endpoints are open.
from kgenxai_config import AUTH

# ================= AGENT IDENTITY (SYSTEM PROMPT) =================
# The agent's ROLE is defined once here, separate from the per-call task content
# (schema + user request, built fresh in build_and_execute_nl_query below).
# Default/fallback identity; overridable via `nl_to_sparql_system_prompt` in the
# request body (e.g. from the Orchestrator's Admin > Agent Setup tab).
NL_TO_SPARQL_SYSTEM_PROMPT = (
    "You are a Data Engineer building SPARQL queries for a Knowledge Graph. You only "
    "ever write strictly safe, read-only SELECT queries."
)

def _endpoints_for(fuseki_base, dataset_name=None):
    """Builds the three ProductReviews dataset endpoints from a given base URL and
    (optionally renamed) dataset name."""
    base = (fuseki_base or DEFAULT_FUSEKI_BASE).strip().rstrip("/")
    name = (dataset_name or "").strip() or DEFAULT_DATASET_NAME
    return {
        "query": f"{base}/{name}/query",
        "data": f"{base}/{name}/data",
        "update": f"{base}/{name}/update",
    }

# Kept as module-level constants too, for any code (or tests) that still references
# them directly -- they reflect the default Fuseki server and default dataset name.
_default_endpoints = _endpoints_for(DEFAULT_FUSEKI_BASE, DEFAULT_DATASET_NAME)
PRODUCT_REVIEWS_ENDPOINT = _default_endpoints["query"]
DATA_ENDPOINT = _default_endpoints["data"]
UPDATE_ENDPOINT = _default_endpoints["update"]

# ================= CACHING LAYER =================
QUERY_CACHE = {}
CACHE_TTL = 3600  # Cache queries for 1 hour

def get_cached_result(query, endpoint):
    # Cache key includes the endpoint, not just the query text, so the same query
    # string sent against two different Fuseki servers (e.g. a user switches the
    # Fuseki Base URL mid-session) never returns a stale result from the other server.
    cache_key = hashlib.md5(f"{endpoint}|{query}".encode('utf-8')).hexdigest()
    if cache_key in QUERY_CACHE:
        cached_data, timestamp = QUERY_CACHE[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return cached_data
    return None

def set_cached_result(query, endpoint, data):
    cache_key = hashlib.md5(f"{endpoint}|{query}".encode('utf-8')).hexdigest()
    QUERY_CACHE[cache_key] = (data, time.time())

# ================= OPTIMIZATION LAYER =================
def is_safe_query(query):
    """Safety check to ensure no destructive operations are executed."""
    upper_query = query.upper()
    forbidden_keywords = ["INSERT ", "DELETE ", "UPDATE ", "DROP ", "CLEAR ", "CREATE "]
    for f in forbidden_keywords:
        if f in upper_query:
            return False
    return True

def optimize_sparql_query(query):
    """
    Intercepts notoriously slow SPARQL patterns and rewrites them for speed
    on large Fuseki datasets, delegating heavy sorting to Python memory.
    """
    is_random = False
    orig_limit = None
    optimized = query

    # 1. Enforce Type Boundaries on expensive unbounded fuzzy searches
    if "?item a ?type" in optimized and "CONTAINS" in optimized.upper():
        optimized = optimized.replace("?item a ?type", "?item a <http://schema.org/Product>")
        logger.info("🔧 Optimizer: Injected schema:Product boundary to prevent full-graph text scan.")

    # 2. Eliminate full-table sorting from ORDER BY RAND()
    if re.search(r'(?i)ORDER\s+BY\s+RAND\(\)', optimized):
        optimized = re.sub(r'(?i)ORDER\s+BY\s+RAND\(\)', '', optimized)
        is_random = True
        
        limit_match = re.search(r'(?i)LIMIT\s+(\d+)', optimized)
        if limit_match:
            orig_limit = int(limit_match.group(1))
            optimized = re.sub(r'(?i)LIMIT\s+\d+', 'LIMIT 2000', optimized)
        else:
            orig_limit = 5
            optimized += " LIMIT 2000"
        logger.info(f"🔧 Optimizer: Stripped ORDER BY RAND(), delegating sort to Python.")

    return optimized, is_random, orig_limit


# ================= ENDPOINTS =================

@app.route('/api/execute_safe_sparql', methods=['POST'])
def execute_safe_sparql():
    """Executes predefined or safely constructed SPARQL read queries with High-Performance Overrides."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be valid JSON with an object at the top level."}), 400

    query = data.get('query')
    endpoints = _endpoints_for(data.get('fuseki_base'), data.get('product_reviews_dataset'))
    product_reviews_endpoint = endpoints["query"]

    if not query:
        logger.error("No query provided in request.")
        return jsonify({"error": "No query provided"}), 400

    if not is_safe_query(query):
        logger.warning("Attempted to execute an unsafe query.")
        return jsonify({"error": "Query is unsafe"}), 403

    try:
        # Check High-Speed Cache First
        cached_data = get_cached_result(query, product_reviews_endpoint)
        if cached_data:
            logger.info("⚡ Serving SPARQL response instantly from ultra-fast cache.")
            return jsonify(cached_data)

        # Optimize query for Large Datasets
        optimized_query, is_random, orig_limit = optimize_sparql_query(query)

        res = requests.post(product_reviews_endpoint, data={'query': optimized_query}, auth=AUTH, timeout=30)
        if res.status_code != 200:
            return jsonify({"error": "SPARQL execution failed", "details": res.text}), res.status_code
            
        res_data = res.json()
        
        # Apply Python-side random sampling to bypass slow DB sorting
        if is_random and 'results' in res_data and 'bindings' in res_data['results']:
            bindings = res_data['results']['bindings']
            if len(bindings) > orig_limit:
                res_data['results']['bindings'] = random.sample(bindings, orig_limit)
                
        # Store successful read in Cache
        set_cached_result(query, product_reviews_endpoint, res_data)
        
        return jsonify(res_data)
    except Exception as e:
        logger.exception("Error executing SPARQL query.")
        return jsonify({"error": str(e)}), 500


def _discover_schema_text(fuseki_base=None, dataset_name=None):
    """
    Core schema-discovery logic, usable both from the Flask route below and from
    other functions in this module (e.g. build_and_execute_nl_query) without going
    through a Response object.

    Returns (schema_text, error_message). Exactly one of the two will be None.
    """
    product_reviews_endpoint = _endpoints_for(fuseki_base, dataset_name)["query"]
    logger.info("Discovering schema for XAI_ProductReviews...")
    # OPTIMIZATION: Wrap queries in sub-select LIMITS so schema discovery probes a block
    # of the data instantly rather than scanning millions of triples.
    q_types = """
    SELECT ?type (COUNT(?s) as ?count) 
    WHERE { { SELECT ?s ?type WHERE { ?s a ?type } LIMIT 50000 } } 
    GROUP BY ?type ORDER BY DESC(?count) LIMIT 10
    """

    q_preds = """
    SELECT ?p (SAMPLE(?o) as ?example)
    WHERE { { SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 50000 } }
    GROUP BY ?p LIMIT 50
    """

    try:
        types_res = requests.post(product_reviews_endpoint, data={'query': q_types}, auth=AUTH, timeout=30).json()
        preds_res = requests.post(product_reviews_endpoint, data={'query': q_preds}, auth=AUTH, timeout=30).json()

        schema_text = "Discovered Types:\n"
        for b in types_res.get('results', {}).get('bindings', []):
            schema_text += f"- {b['type']['value']} (Approx Count: {b['count']['value']})\n"

        schema_text += "\nDiscovered Properties & Examples:\n"
        for b in preds_res.get('results', {}).get('bindings', []):
            schema_text += f"- {b['p']['value']} (Example: {b['example']['value']})\n"

        return schema_text, None
    except Exception as e:
        logger.exception("Error discovering schema")
        return None, str(e)


@app.route('/api/discover_schema', methods=['GET'])
def discover_schema():
    """Dynamically probes the ProductReviews dataset to build a schema understanding."""
    # GET request -- fuseki_base/product_reviews_dataset come from query string params, e.g.
    # /api/discover_schema?fuseki_base=https://my-server:8080/fuseki&product_reviews_dataset=XAI_ProductReviews
    fuseki_base = request.args.get('fuseki_base')
    dataset_name = request.args.get('product_reviews_dataset')
    schema_text, error = _discover_schema_text(fuseki_base, dataset_name)
    if error:
        return jsonify({"error": error}), 500
    return jsonify({"schema_text": schema_text})


@app.route('/api/nl_to_sparql', methods=['POST'])
def build_and_execute_nl_query():
    """Uses Gemini to translate a Natural Language query to SPARQL, then executes it."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be valid JSON with an object at the top level."}), 400

    nl_query = data.get('query')
    product_reviews_endpoint = _endpoints_for(data.get('fuseki_base'), data.get('product_reviews_dataset'))["query"]
    system_prompt = data.get('nl_to_sparql_system_prompt') or NL_TO_SPARQL_SYSTEM_PROMPT
    
    if not nl_query:
        return jsonify({"error": "No query provided"}), 400
        
    try:
        # 1. First, dynamically get the schema context
        schema_info, schema_error = _discover_schema_text(data.get('fuseki_base'), data.get('product_reviews_dataset'))
        if schema_error:
            schema_info = "Unknown schema."
            logger.warning(f"Schema discovery failed, proceeding with unknown schema: {schema_error}")
        
        prompt = f"""
        SCHEMA CONTEXT:
        {schema_info}
        
        USER REQUEST:
        {nl_query}
        
        Task: Write a strictly SAFE (SELECT) SPARQL query that fulfills the user request based on the schema.
        
        CRITICAL PERFORMANCE RULES FOR LARGE DATASETS:
        1. AVOID `ORDER BY RAND()`. It causes fatal performance degradation. Use `LIMIT` instead.
        2. CONSTRICT SEARCH SPACE: Always explicitly define the subject type (e.g., `?s a schema:Product`) BEFORE applying expensive text filters like `CONTAINS` or `REGEX`.
        3. AGGRESSIVE LIMITS: Always append a `LIMIT` clause (e.g., `LIMIT 10`) to prevent massive payloads.
        4. SUB-QUERIES: If scanning for distinct traits across millions of rows, use a sub-query with a `LIMIT` to sample data rapidly.
        
        Output ONLY the raw SPARQL string. Do not include markdown tags like ```sparql. 
        """
        
        model = genai.GenerativeModel('gemini-3-flash-preview', system_instruction=system_prompt)
        response = model.generate_content(prompt)
        
        # Clean markdown if present
        sparql_query = response.text.replace("```sparql", "").replace("```", "").strip()
        logger.info(f"Generated dynamic SPARQL:\n{sparql_query}")
        
        # 2. Safety Check & Execution
        if not is_safe_query(sparql_query):
            logger.warning("LLM generated an unsafe query.")
            return jsonify({"error": "Generated query is unsafe."}), 403
            
        res = requests.post(product_reviews_endpoint, data={'query': sparql_query}, auth=AUTH, timeout=30)
        
        if res.status_code != 200:
            return jsonify({"error": "SPARQL execution failed", "details": res.text, "generated_query": sparql_query}), res.status_code
            
        return jsonify({
            "generated_query": sparql_query,
            "results": res.json()
        })
        
    except Exception as e:
        logger.exception("Error in dynamic query building")
        return jsonify({"error": str(e)}), 500


# ================= WRITE ENDPOINTS =================
# These exist so EO_ProductReviews_Agent.py is the SOLE component that ever talks to
# the XAI_ProductReviews Fuseki dataset -- no other agent (Orchestrator, Recommender,
# Explainer) holds Fuseki credentials or a direct endpoint URL for this dataset anymore.
# Unlike /api/execute_safe_sparql, these intentionally perform writes, so they don't go
# through is_safe_query (which exists to block writes on the *read* endpoint).

@app.route('/api/upload_graph', methods=['POST'])
def upload_graph():
    """
    Accepts raw Turtle data (Content-Type: text/turtle) and uploads it to the
    XAI_ProductReviews dataset's /data endpoint, exactly as the Orchestrator's
    upload_graph_to_fuseki() used to do directly. Used by the Orchestrator's
    dataset ingestion pipeline (Amazon datasets + custom uploads).

    Optional ?fuseki_base=<url>&product_reviews_dataset=<name> query params select a
    non-default Fuseki server/dataset name (the request body is raw Turtle here, not
    JSON, so these can't be body fields).
    """
    turtle_data = request.get_data()
    if not turtle_data:
        return jsonify({"error": "No Turtle data provided in request body."}), 400

    data_endpoint = _endpoints_for(request.args.get('fuseki_base'), request.args.get('product_reviews_dataset'))["data"]

    try:
        res = requests.post(
            data_endpoint,
            data=turtle_data,
            headers={'Content-Type': 'text/turtle; charset=utf-8'},
            auth=AUTH,
            timeout=120,
        )
        if res.status_code not in (200, 201, 204):
            logger.error(f"Upload failed (HTTP {res.status_code}): {res.text[:300]}")
            return jsonify({"error": "Upload failed", "details": res.text[:500]}), res.status_code

        # Any write invalidates previously cached reads.
        QUERY_CACHE.clear()
        logger.info("📥 Uploaded Turtle graph to XAI_ProductReviews and cleared the query cache.")
        return jsonify({"status": "success"})
    except requests.exceptions.Timeout:
        logger.error("Upload to Fuseki timed out.")
        return jsonify({"error": "Upload to Fuseki timed out."}), 504
    except Exception as e:
        logger.exception("Error uploading graph.")
        return jsonify({"error": str(e)}), 500


@app.route('/api/clear_graph', methods=['POST'])
def clear_graph():
    """
    Deletes ALL triples from XAI_ProductReviews, in batches, exactly matching the
    behavior previously implemented inline in the Orchestrator's admin tab. Moving
    it here means the Orchestrator no longer needs a direct XAI_ProductReviews query/update endpoint or
    Fuseki credentials -- it just calls this one endpoint and watches progress.

    Optional ?fuseki_base=<url>&product_reviews_dataset=<name> query params select a
    non-default Fuseki server/dataset name (no JSON body is sent for this endpoint,
    so these can't be body fields either).
    """
    batch_size = 500000
    endpoints = _endpoints_for(request.args.get('fuseki_base'), request.args.get('product_reviews_dataset'))
    product_reviews_endpoint = endpoints["query"]
    update_endpoint = endpoints["update"]
    try:
        count_res = requests.post(
            product_reviews_endpoint,
            data={'query': "SELECT (COUNT(?s) AS ?count) WHERE { ?s ?p ?o }"},
            auth=AUTH,
            timeout=60,
        )
        if count_res.status_code != 200:
            return jsonify({"error": "Failed to count existing triples", "details": count_res.text[:300]}), count_res.status_code

        bindings = count_res.json().get('results', {}).get('bindings', [])
        total_triples = int(bindings[0]['count']['value']) if bindings else 0

        if total_triples == 0:
            QUERY_CACHE.clear()
            return jsonify({"status": "success", "message": "Graph was already empty.", "deleted": 0})

        total_batches = (total_triples + batch_size - 1) // batch_size
        delete_query = f"""
        DELETE {{ ?s ?p ?o }}
        WHERE {{
            {{
                SELECT ?s ?p ?o
                WHERE {{ ?s ?p ?o }}
                LIMIT {batch_size}
            }}
        }}
        """

        for batch_num in range(total_batches):
            res = requests.post(
                update_endpoint,
                data={'update': delete_query},
                auth=AUTH,
                timeout=120,
            )
            if res.status_code not in (200, 201, 204):
                logger.error(f"Delete batch {batch_num + 1}/{total_batches} failed (HTTP {res.status_code})")
                return jsonify({
                    "error": f"Failed during delete batch {batch_num + 1} of {total_batches}",
                    "details": res.text[:500],
                }), res.status_code

        QUERY_CACHE.clear()
        logger.info(f"🗑️ Cleared {total_triples} triples from XAI_ProductReviews in {total_batches} batch(es).")
        return jsonify({"status": "success", "deleted": total_triples, "batches": total_batches})
    except requests.exceptions.Timeout:
        logger.error("Clear graph operation timed out.")
        return jsonify({"error": "Clear graph operation timed out."}), 504
    except Exception as e:
        logger.exception("Error clearing graph.")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    logger.info("🚀 Starting Product Reviews Agent on port 5002...")
    app.run(port=5002, debug=True, use_reloader=False)