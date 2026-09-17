import google.generativeai as genai
from flask import Flask, request, jsonify
import requests
import re
import json
import io
import sys
import os
import contextlib
import traceback
import logging

# ================= CONFIGURATION & LOGGING =================
app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | RECOMMENDER_AGENT | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

from kgenxai_config import (
    AUTH, DEFAULT_FUSEKI_BASE, DEFAULT_ALGORITHMS_DATASET,
    endpoint, sparql_prefixes,
)
from execution_recorder import ExecutionRecorder, new_execution_id

# ================= AGENT IDENTITY (SYSTEM PROMPTS) =================
# An agent's ROLE ("who it is") is defined once, here, and passed to Gemini via
# `system_instruction=` -- separate from the per-call task prompt (built fresh each
# time in select_best_algorithm/generate_dynamic_script below). Previously the role
# sentence ("You are a Semantic Data Engineer...") was baked directly into the
# one-off task prompt and resent every call, which mixed up the two concerns. These
# constants are the fallback/default identities; the Admin "AGENT Agent Setup" tab
# (in the Orchestrator) can override them by fetching from the KG and passing an
# explicit `system_instruction` override into the functions below.
SELECTOR_SYSTEM_PROMPT = (
    "You are an Expert Recommender System Architect. Your sole responsibility is to "
    "choose the single most appropriate recommendation algorithm for a given user "
    "profile, strictly based on the provided algorithm manuals. You never write code "
    "and you never fabricate an algorithm name that wasn't provided to you."
)

COMPOSER_SYSTEM_PROMPT = (
    "You are a Semantic Data Engineer responsible for writing robust, fully runnable "
    "Python scripts that implement a given recommendation algorithm workflow against "
    "a SPARQL-backed knowledge graph. You always follow the mandatory logging setup "
    "and output contract you are given exactly, and you never invent data you were not "
    "given access to."
)

# NOTE: XAI_ProductReviews has no direct Fuseki endpoint or credential here anymore.
# EO_ProductReviews_Agent.py is the sole owner of that dataset -- both the schema scan
# below AND the data-fetching code this file asks Gemini to generate now go through
# its HTTP API, never straight to Fuseki with SPARQLWrapper.
# AUTH now comes from kgenxai_config, which reads FUSEKI_USER / FUSEKI_PASSWORD
# from the environment. It used to be the literal ("username", "aubfuseki") --
# which could not be published, and which should be treated as compromised
# given the endpoints are open.
DEFAULT_ALGORITHMS_DATASET_NAME = DEFAULT_ALGORITHMS_DATASET
PRODUCT_REVIEWS_SCHEMA_API = "http://127.0.0.1:5002/api/discover_schema"
PRODUCT_REVIEWS_QUERY_API = "http://127.0.0.1:5002/api/execute_safe_sparql"

def _algorithms_endpoint_for(fuseki_base, dataset_name=None):
    """Builds the XAI_RecommendationAlgorithms query endpoint from a given base URL
    and (optionally renamed, via Admin > Configuration) dataset name."""
    base = (fuseki_base or DEFAULT_FUSEKI_BASE).strip().rstrip("/")
    name = (dataset_name or "").strip() or DEFAULT_ALGORITHMS_DATASET_NAME
    return f"{base}/{name}/query"

# ================= 1. DYNAMIC DISCOVERY =================

def discover_ontology(fuseki_base=None, product_reviews_dataset=None):
    """Queries the ProductReviews agent to discover schema dynamically."""
    params = {}
    if fuseki_base:
        params['fuseki_base'] = fuseki_base
    if product_reviews_dataset:
        params['product_reviews_dataset'] = product_reviews_dataset
    try:
        res = requests.get(
            PRODUCT_REVIEWS_SCHEMA_API,
            params=params or None,
            timeout=10,
        ).json()
        return res.get('schema_text', 'Schema discovery failed.')
    except Exception as e:
        logger.error(f"Error calling ProductReviews API for discovery: {e}")
        return f"Error discovering ontology: {str(e)}"

def get_available_recipes(fuseki_base=None, algorithms_dataset=None):
    """
    Fetch available algorithms (mls:Algorithm) AND their strategic 'Usage Manuals'
    (rdf:value of mls:ImplementationCharacteristic quality) from the Function Graph.
    """
    algorithms_endpoint = _algorithms_endpoint_for(fuseki_base, algorithms_dataset)
    # rdf:value is retained here: it is valid RDF, it is on a read path, and
    # migration 05 adds mls:hasValue alongside it rather than replacing it, so
    # this query keeps working throughout.
    query = f"""
    {sparql_prefixes('rdf', 'rdfs', 'mls')}

    SELECT ?name ?spec WHERE {{
        ?method a mls:Algorithm ;
                rdfs:label ?name .
        ?workflow mls:implements ?method ;
                  mls:hasQuality ?specNode .
        ?specNode rdf:value ?spec .
    }}
    """
    try:
        # Force Fuseki to return JSON
        headers = {'Accept': 'application/sparql-results+json'}
        res = requests.post(algorithms_endpoint, data={'query': query}, auth=AUTH, headers=headers, timeout=30)
        
        if res.status_code == 200:
            return [{ "name": b['name']['value'], "spec": b['spec']['value']} for b in res.json()['results']['bindings']]
        else:
            logger.error(f"Fuseki returned HTTP {res.status_code} in recipes query: {res.text}")
            return []
    except Exception as e: 
        logger.error(f"Error fetching recipes: {e}")
        return []

def get_algorithm_source(func_name, fuseki_base=None, algorithms_dataset=None):
    """
    Fetches the linked pko:Step steps of the algorithm, orders them using the 
    linked list properties (pko:hasFirstStep and pko:nextStep), and returns
    them as a structured instruction string.
    """
    algorithms_endpoint = _algorithms_endpoint_for(fuseki_base, algorithms_dataset)
    # Step membership is pko:hasStep, Procedure -> Step (defect 2b). Every
    # structured field is OPTIONAL, so a step that has not been enriched yields
    # exactly the text it did before -- which is what allows the procedural
    # enrichment to be applied one algorithm at a time.
    query = f"""
    {sparql_prefixes('rdf', 'rdfs', 'mls', 'pko', 'pplan', 'dcterms', 'ex')}

    SELECT ?step ?comment ?description ?next ?first ?stepNumber ?constraint
           (GROUP_CONCAT(DISTINCT ?inText;  separator=", ") AS ?inputs)
           (GROUP_CONCAT(DISTINCT ?outText; separator=", ") AS ?outputs)
           (GROUP_CONCAT(DISTINCT ?paramText; separator="; ") AS ?parameters)
    WHERE {{
        ?method a mls:Algorithm ;
                rdfs:label "{func_name}" .
        ?workflow mls:implements ?method .

        ?workflow pko:hasStep ?step .
        ?step rdfs:comment ?comment .

        OPTIONAL {{ ?step dcterms:description ?description }}
        OPTIONAL {{ ?step pko:stepNumber ?stepNumber }}
        OPTIONAL {{ ?step ex:generationConstraint ?constraint }}
        OPTIONAL {{
            ?step pplan:hasInputVar ?inVar .
            ?inVar rdfs:label ?inLabel .
            OPTIONAL {{ ?inVar ex:dataShape ?inShape }}
            BIND(CONCAT(?inLabel, IF(BOUND(?inShape),
                        CONCAT(" (", ?inShape, ")"), "")) AS ?inText)
        }}
        OPTIONAL {{
            ?step pplan:hasOutputVar ?outVar .
            ?outVar rdfs:label ?outLabel .
            OPTIONAL {{ ?outVar ex:dataShape ?outShape }}
            BIND(CONCAT(?outLabel, IF(BOUND(?outShape),
                        CONCAT(" (", ?outShape, ")"), "")) AS ?outText)
        }}
        OPTIONAL {{
            ?step mls:hasHyperParameter ?param .
            ?param rdfs:label ?paramName .
            OPTIONAL {{ ?setting mls:specifiedBy ?param ; mls:hasValue ?paramValue }}
            BIND(CONCAT(?paramName, IF(BOUND(?paramValue),
                        CONCAT(" = ", ?paramValue), "")) AS ?paramText)
        }}

        OPTIONAL {{ ?step pko:nextStep ?next }}
        OPTIONAL {{
            ?workflow pko:hasFirstStep ?first .
            FILTER(?step = ?first)
        }}
    }}
    GROUP BY ?step ?comment ?description ?next ?first ?stepNumber ?constraint
    """
    try:
        # Force Fuseki to return JSON
        headers = {'Accept': 'application/sparql-results+json'}
        res = requests.post(algorithms_endpoint, data={'query': query}, auth=AUTH, headers=headers, timeout=30)
        
        if res.status_code == 200:
            bindings = res.json()['results']['bindings']
            if not bindings:
                logger.warning(f"No workflow steps found for algorithm: {func_name}")
                return None
            
            # Reconstruct the sequence from the linked list structure
            steps_dict = {}
            first_step_uri = None

            for b in bindings:
                step_uri = b['step']['value']
                comment = b['comment']['value']
                next_uri = b.get('next', {}).get('value')
                is_first = b.get('first', {}).get('value') is not None

                steps_dict[step_uri] = {
                    'comment': comment,
                    'next': next_uri,
                    'uri': step_uri,
                    # WP2 structured fields; absent on un-migrated steps.
                    'description': b.get('description', {}).get('value'),
                    'inputs': b.get('inputs', {}).get('value'),
                    'outputs': b.get('outputs', {}).get('value'),
                    'parameters': b.get('parameters', {}).get('value'),
                    'constraint': b.get('constraint', {}).get('value'),
                }
                if is_first:
                    first_step_uri = step_uri

            # Fallback if first step wasn't explicitly matched by the filter
            if not first_step_uri and steps_dict:
                pointed_to = set(s['next'] for s in steps_dict.values() if s['next'])
                for step_uri in steps_dict:
                    if step_uri not in pointed_to:
                        first_step_uri = step_uri
                        break

            # Absolute fallback: choose any starting point
            if not first_step_uri and steps_dict:
                first_step_uri = list(steps_dict.keys())[0]

            ordered_steps = []
            current_uri = first_step_uri
            visited = set()
            step_num = 1

            while current_uri and current_uri in steps_dict and current_uri not in visited:
                visited.add(current_uri)
                step_info = steps_dict[current_uri]
                ordered_steps.append(_compose_step_text(step_num, step_info))
                current_uri = step_info['next']
                step_num += 1

            # Append any orphaned steps if present
            for step_uri, step_info in steps_dict.items():
                if step_uri not in visited:
                    ordered_steps.append(
                        _compose_step_text(step_num, step_info, orphan=True))
                    step_num += 1

            return "\n".join(ordered_steps)
        else:
            logger.error(f"Fuseki returned HTTP {res.status_code} in source query: {res.text}")
            return None
    except Exception as e: 
        logger.error(f"Error fetching algorithm source: {e}")
        return None

def get_workflow_context(func_name, fuseki_base=None, algorithms_dataset=None):
    """Resolve the workflow, algorithm, step and variable URIs for an algorithm.

    Without this the execution graph is an island: ExecutionRecorder was being
    constructed with only the algorithm's display name, so pko:hasExecutedStep
    and p-plan:correspondsToVariable were never emitted and nothing tied a run
    back to the procedure it executed. The Explainer's trace query requires
    hasExecutedStep, so it matched nothing and fell back to the log file --
    silently, which is why it looked like it was working.

    Step URIs are read from the graph rather than derived. Two naming schemes
    are in use: hand-authored steps under /Trace/ and Admin-created steps as
    <workflow>/step<n>, so any deterministic guess would be wrong for one of
    them.

    Returns a dict, or an empty dict if anything goes wrong -- execution
    logging must never be able to fail a recommendation.
    """
    query = f"""
    {sparql_prefixes('rdf', 'rdfs', 'mls', 'pko', 'pplan')}

    SELECT ?workflow ?algorithm ?step ?stepNumber ?var ?varLabel WHERE {{
        ?algorithm a mls:Algorithm ; rdfs:label "{func_name}" .
        ?workflow mls:implements ?algorithm ; pko:hasStep ?step .
        OPTIONAL {{ ?step pko:stepNumber ?stepNumber }}
        OPTIONAL {{
            {{ ?step pplan:hasInputVar ?var }} UNION {{ ?step pplan:hasOutputVar ?var }}
            ?var rdfs:label ?varLabel .
        }}
    }}
    """
    context = {"workflow_uri": None, "algorithm_uri": None,
               "step_uris": {}, "variable_uris": {}}
    try:
        endpoint_url = _algorithms_endpoint_for(fuseki_base, algorithms_dataset)
        response = requests.post(
            endpoint_url, data={'query': query}, auth=AUTH, timeout=15,
            headers={'Accept': 'application/sparql-results+json'})
        if response.status_code != 200:
            return context
        for b in response.json().get('results', {}).get('bindings', []):
            context["workflow_uri"] = b.get('workflow', {}).get('value')
            context["algorithm_uri"] = b.get('algorithm', {}).get('value')
            number = b.get('stepNumber', {}).get('value')
            step = b.get('step', {}).get('value')
            if number and step:
                context["step_uris"][str(number)] = step
            label = b.get('varLabel', {}).get('value')
            var = b.get('var', {}).get('value')
            if label and var:
                key = "".join(c for c in label.lower() if c.isalnum())
                context["variable_uris"][key] = var
        logger.info(
            f"Workflow context for '{func_name}': "
            f"{len(context['step_uris'])} step(s), "
            f"{len(context['variable_uris'])} declared variable(s).")
    except Exception as e:
        logger.warning(f"Could not resolve workflow context for {func_name}: {e}")
    return context


def _compose_step_text(step_num, step_info, orphan=False):
    """Render one step for the code-generation prompt.

    This is the highest-risk function in the WP2 change, because its output is
    what the composer LLM turns into Python. Two properties keep it safe:

    1. The text is a SUPERSET of today's. The description reads almost
       identically to the legacy comment, and the structured Inputs / Outputs /
       Parameters / Constraints lines are added after it. No prior instruction
       is ever removed, only made explicit.

    2. When a step has no structured fields it emits BYTE-IDENTICAL text to
       today. That is what allows the enrichment to be migrated one algorithm
       at a time with the cached scripts still matching the golden baseline.
    """
    marker = " (orphan)" if orphan else ""
    has_structure = any(step_info.get(k) for k in
                        ("description", "inputs", "outputs", "parameters",
                         "constraint"))
    if not has_structure:
        return f"Step {step_num}{marker}: {step_info['comment']}"

    body = step_info.get("description") or step_info["comment"]
    lines = [f"Step {step_num}{marker}: {body}"]
    if step_info.get("inputs"):
        lines.append(f"    Inputs: {step_info['inputs']}")
    if step_info.get("outputs"):
        lines.append(f"    Outputs: {step_info['outputs']}")
    if step_info.get("parameters"):
        lines.append(f"    Parameters: {step_info['parameters']}")
    if step_info.get("constraint"):
        lines.append(f"    Constraint: {step_info['constraint']}")
    return "\n".join(lines)


# ================= 2. THE SELECTOR =================

def select_best_algorithm(user_profile, available_recipes, system_prompt=None):
    """
    Uses Chain-of-Thought (CoT) JSON prompting to force the LLM to evaluate 
    the user's history size before making a decision.

    `system_prompt` optionally overrides SELECTOR_SYSTEM_PROMPT (e.g. with a version
    edited via the Orchestrator's Admin > Agent Setup tab, fetched from the KG).
    """
    model = genai.GenerativeModel(
        'gemini-3-flash-preview',
        system_instruction=system_prompt or SELECTOR_SYSTEM_PROMPT,
    )
    recipes_text = "\n\n".join([f"--- ALGORITHM: {r['name']} ---\n{r['spec']}" for r in available_recipes])
    
    prompt = f"""
    ### TASK
    Select the single most appropriate recommendation algorithm for a user based STRICTLY on their profile data and the provided Algorithm Manuals.
    
    ### 1. USER PROFILE (The Context)
    {json.dumps(user_profile, indent=2)}
    
    ### 2. AVAILABLE ALGORITHMS & MANUALS (The Rules)
    {recipes_text}
    
    ### 3. DECISION LOGIC (Chain of Thought required)
    - Step 1: Examine the 'history' array in the User Profile. How many items are in it?
    - Step 2: Read the "When to Use" section of EACH algorithm manual and choose accordingly.
    
    ### OUTPUT FORMAT
    You MUST return a valid JSON object with EXACTLY these two keys:
    {{
        "reasoning": "Explain your logic step-by-step. Mention the size of the history array and why that led to your choice.",
        "algorithm_name": "The exact name of the chosen algorithm (e.g., 'Collaborative Filtering Algorithm' or 'Content-Based Filtering Algorithm')"
    }}
    """
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        decision_data = json.loads(response.text)
        
        clean_name = decision_data.get("algorithm_name", "").strip()
        logger.info(f"🧠 [SELECTOR REASONING]: {decision_data.get('reasoning')}")
        
        valid_names = [r['name'] for r in available_recipes]
        if clean_name in valid_names:
            return clean_name
        logger.warning(f"Gemini selected invalid '{clean_name}'. Defaulting to first option.")
        return available_recipes[0]['name']
    except Exception as e:
        logger.error(f"Selection Agent failed: {e}. Defaulting to first option.")
        return available_recipes[0]['name']

# ================= 3. THE COMPOSER =================

def generate_dynamic_script(user_profile, ontology_scan, algo_name, algo_workflow, previous_code=None, error_log=None, existing_working_code=None, fuseki_base=None, system_prompt=None, product_reviews_dataset=None):
    """
    Asks Gemini to write (or adjust/fix) the glue script based on the semantic workflow.
    Handles from-scratch generation, reusability adjustment, and error fixing.

    `system_prompt` optionally overrides COMPOSER_SYSTEM_PROMPT (e.g. with a version
    edited via the Orchestrator's Admin > Agent Setup tab, fetched from the KG).
    """
    model = genai.GenerativeModel(
        'gemini-3-flash-preview',
        system_instruction=system_prompt or COMPOSER_SYSTEM_PROMPT,
    )
    
    base_prompt = f"""
    WRITE A PYTHON SCRIPT.

    ### 1. GOAL
    Retrieve data from a SPARQL endpoint, process it into a DataFrame, implement the workflow provided, and store the final recommendation result in a variable named `solution`.

    ### 2. DYNAMIC SCHEMA (Database Structure)
    {ontology_scan}

    ### 3. THE ALGORITHM WORKFLOW (To be implemented in Python)
    {algo_workflow}

    ### 4. MANDATORY LOGGING SETUP (CRITICAL)
    You MUST include this EXACT code block at the very top of your script to ensure step-by-step standardized logging. Do not alter this setup block:
    ```python
    import logging
    import sys
    import pandas as pd
    import numpy as np
    import requests
    import pprint

    # Standardized logging setup
    #
    # CONCURRENCY FIX (WP3): the log used to be named per ALGORITHM and opened
    # with mode='w'. Two people running the same algorithm at once truncated
    # each other's trace, and because the Explainer resolved traces by
    # algorithm name, one user could be shown the other's execution. Both files
    # are now written: the per-execution one is authoritative, the legacy one
    # is kept so anything that reads it keeps working.
    # EXECUTION_ID is injected by the Recommender before this script runs. The
    # fallback keeps the script runnable standalone, e.g. when a developer
    # executes a cached generated_*.py directly for debugging.
    EXECUTION_ID = globals().get("EXECUTION_ID") or "standalone"
    log_filename = f"execution_trace_{algo_name.replace(' ', '_')}_{{EXECUTION_ID}}.log"
    legacy_log_filename = "execution_trace_{algo_name.replace(' ', '_')}.log"
    logger = logging.getLogger("DynamicScriptLogger")
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    legacy_fh = logging.FileHandler(legacy_log_filename, mode='w', encoding='utf-8')
    ch = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)
    legacy_fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(legacy_fh)
    logger.addHandler(ch)

    def log_var(name, val, detailed=True):
        \"\"\"Rich logger: ALWAYS writes the FULL, untruncated value (not just a
        shape/length summary) -- this is the ONLY source of concrete data (actual
        scores, item names, candidate lists, similarity values) the Explainer
        Agent has to work with for its Mathematical Rationale section. Call this
        on every meaningful variable (candidate scores, top-N results, computed
        similarities/lift/confidence, the final ranked list before filtering,
        etc.) so a reader can see the real values, not just a count. The
        `detailed` argument is kept for backward compatibility with older calls
        but no longer suppresses output -- every call now logs the full value.\"\"\"
        try:
            if isinstance(val, pd.DataFrame):
                cols = list(val.columns)
                logger.debug(f"VAR [{{name}}] | Type: DataFrame | Shape: {{val.shape}} | Columns: {{cols}}")
                if not val.empty:
                    logger.debug(f"VAR [{{name}}] | Full data (all {{val.shape[0]}} rows):\\n{{val.to_string()}}")
            elif isinstance(val, pd.Series):
                logger.debug(f"VAR [{{name}}] | Type: Series | Shape: {{val.shape}}")
                if not val.empty:
                    logger.debug(f"VAR [{{name}}] | Full data (all {{len(val)}} values):\\n{{val.to_string()}}")
            elif isinstance(val, (np.ndarray, list, tuple, set, dict)):
                logger.debug(f"VAR [{{name}}] | Type: {{type(val).__name__}} | Length: {{len(val)}}")
                logger.debug(f"VAR [{{name}}] | Full value:\\n{{pprint.pformat(val, width=120)}}")
            else:
                logger.debug(f"VAR [{{name}}] | Type: {{type(val).__name__}} | Value: {{val}}")
        except Exception as e:
            logger.debug(f"VAR [{{name}}] | Could not log value due to error: {{str(e)}}")

    def log_step_io(step_number, inputs=None, outputs=None):
        \"\"\"Record one workflow step's real inputs and outputs as knowledge
        graph instances (WP3).

        Call this ONCE per workflow step, passing dictionaries of the actual
        values the step consumed and produced, e.g.

            log_step_io(2,
                        inputs={{'user_item_matrix': df_matrix}},
                        outputs={{'top_n_neighbours': top_neighbours}})

        This is what lets an explanation say what the value of a named variable
        was on THIS run, as a graph query rather than by extracting numbers from
        prose. It is additive: log_var() below is unchanged and still writes the
        full value to the log file, so nothing that exists today stops working.

        Never raises. A failure to log must never fail a recommendation.
        \"\"\"
        try:
            recorder = globals().get("_EXECUTION_RECORDER")
            if recorder is not None:
                recorder.log_step_io(step_number, inputs=inputs, outputs=outputs)
            logger.debug(
                f"STEP {{step_number}} | inputs={{list((inputs or {{}}).keys())}} "
                f"| outputs={{list((outputs or {{}}).keys())}}")
        except Exception as e:
            logger.debug(f"log_step_io({{step_number}}) ignored an error: {{e}}")

    # Tracks every real product/entity URI actually returned by run_sparql() below --
    # used after execution to confirm no URI in `solution` was invented (CRITICAL,
    # do not remove or rename this).
    _VERIFIED_KG_URIS = set()

    def run_sparql(query):
        \"\"\"
        Runs a SPARQL SELECT against the product/review knowledge graph.
        This is the ONLY way this script is allowed to read that data -- it goes
        through the ProductReviews Agent's safe-query endpoint, not a direct
        connection to the triple store. Returns a list of binding dicts
        (the same shape as a raw SPARQL JSON response's results.bindings),
        or an empty list on failure.

        Every URI value seen in the results is recorded into `_VERIFIED_KG_URIS`
        (CRITICAL -- do not remove this bookkeeping) -- it is how the caller checks,
        after your script finishes, that every URI in `solution` actually came from
        this knowledge graph rather than being invented.
        \"\"\"
        try:
            resp = requests.post(
                "{PRODUCT_REVIEWS_QUERY_API}",
                json={{"query": query, "fuseki_base": {fuseki_base!r}, "product_reviews_dataset": {product_reviews_dataset!r}}},
                timeout=30,
            )
            if resp.status_code != 200:
                logger.error(f"ProductReviews Agent query failed (HTTP {{resp.status_code}}): {{resp.text[:300]}}")
                return []
            bindings = resp.json().get("results", {{}}).get("bindings", [])
            for binding in bindings:
                for cell in binding.values():
                    if isinstance(cell, dict) and cell.get("type") == "uri" and cell.get("value"):
                        _VERIFIED_KG_URIS.add(cell["value"])
            return bindings
        except Exception as e:
            logger.error(f"Error calling ProductReviews Agent: {{e}}")
            return []

    logger.info(f"========== STARTING NEW EXECUTION: {algo_name} ==========")
    ```

    ### 5. INSTRUCTIONS
    1. **Execution**: Write all logic at the top level. Do NOT use `if __name__ == "__main__":`.
    2. **Connection (CRITICAL -- READ CAREFULLY)**:
       - You do NOT have a direct Fuseki endpoint or credentials for the product/review
         data, and you must NOT use SPARQLWrapper, rdflib, or any direct HTTP call to
         a Fuseki URL for that data.
       - The ONLY way to read product/review data is to call the `run_sparql(query)`
         helper defined above (it POSTs your query to the ProductReviews Agent and
         returns the bindings list). Write your SPARQL SELECT query as a string and
         pass it to `run_sparql(...)`, exactly as you would with any other SPARQL
         client -- the helper handles the HTTP call, auth, and error handling for you.
       - Example: `bindings = run_sparql("SELECT ?s ?p ?o WHERE {{ ?s ?p ?o }} LIMIT 10")`
       - Build a pandas DataFrame from the returned bindings list (each binding is a
         dict like `{{"s": {{"value": "..."}}, "p": {{"value": "..."}}, ...}}`).
    3. **Strategy for Data Consistency (extended for images)**:
       - To avoid "N/A" metadata, use a **Dependent Query Strategy**:
         - Fetch Interactions (User/Product/Rating) via `run_sparql(...)`.
         - Extract distinct Product URIs and use them in a `VALUES` clause in a second
           `run_sparql(...)` call to fetch Metadata (Name/Brand) **and `schema:image`
           in that SAME query** -- e.g.
           `SELECT ?product ?name ?image WHERE {{ VALUES ?product {{ ... }} ?product schema:name ?name . OPTIONAL {{ ?product schema:image ?image }} }}`.
       - Over-fetch: pull more candidate items than you finally need (e.g. 3x the
         target count) BEFORE filtering by image availability in step 6 below, so
         that filtering out imageless items still leaves enough left over.
    4. **Read The Profile From `USER_PROFILE_PATH` (CRITICAL -- do NOT hardcode it)**:
       The current user's full profile (same shape as the JSON shown further below) is
       written, fresh, to its own JSON file before your script runs -- the file path is
       given to you as the pre-defined global string `USER_PROFILE_PATH`. Load it like
       this at the top of your script:
       ```
       import json
       with open(USER_PROFILE_PATH) as _f:
           user_profile = json.load(_f)
       ```
       Then use `user_profile.get("history", [])` etc. (For backward compatibility the
       same data is also available directly as the dict `USER_PROFILE_HISTORY`, but
       prefer reading the file above in new scripts.)
       Do **NOT** write ANY filename string literal yourself to load the profile --
       not directly in an `open(...)` call, and not indirectly by assigning a guessed
       name to a variable first (e.g. `PROFILE_FILE = "user_profile.json"` is just as
       forbidden as `open("user_profile.json")`, since the file that name refers to
       does not exist). `USER_PROFILE_PATH` already IS the correct path -- use that
       exact bare variable name, never a string you write yourself.
       Do **NOT** copy the specific item names, ratings, or preference values from the
       example JSON into your script as literals -- this exact script file is saved and
       reused as-is for future requests from OTHER users, so any hardcoded profile data
       would silently produce wrong recommendations for everyone after the first run.
       This is checked automatically after generation: a script found to contain a
       literal item name from the current profile is rejected before it even runs.
       The JSON shown below is for you to understand the shape/schema ONLY.
    5. **URI-First History Matching (CRITICAL -- reduces "Could not find URI for item" gaps)**:
       - Each entry in the user's `history` (see USER PROFILE below) MAY already carry a
         `"uri"` field -- this is the item's REAL, already-resolved product URI from the
         knowledge graph (set earlier in the pipeline when the item was shown to the user
         from a sample list or search result).
       - For any history entry that HAS a `"uri"`, you MUST use that URI directly (e.g. in
         a `VALUES ?item {{ <uri1> <uri2> ... }}` clause) to fetch its data -- do NOT
         re-derive it via fuzzy/name-based search, and do NOT skip it or fall back to
         approximate matching just because other entries lack a URI.
       - Only fall back to name-based fuzzy matching (e.g. `CONTAINS(LCASE(?name), ...)`)
         for the entries that do NOT have a `"uri"` field. Log a `logger.warning(...)` for
         each such name-only entry you couldn't resolve, exactly as before.
       - This matters for accuracy: silently treating a URI-bearing item the same as an
         unresolved one throws away a known, exact signal the rest of the pipeline already
         paid to establish.
    5b. **Step-Level Tracing (CRITICAL)**: Call `log_step_io(step_number, inputs={{...}}, outputs={{...}})`
       exactly ONCE per workflow step listed in section 3, passing the REAL values that
       step consumed and produced. Use the variable names given in that step's
       "Inputs:" and "Outputs:" lines as the dictionary keys where they are provided.
       Example for step 2 of a collaborative filtering workflow:
           log_step_io(2, inputs={{'user_item_matrix': df_matrix}},
                          outputs={{'top_n_neighbours': top_neighbours}})
       This records the execution as a knowledge graph so the explanation can cite
       what actually happened. It is IN ADDITION to log_var below, not instead of it.

    6. **Variable-Level Tracing (CRITICAL)**: Use `log_var('variable_name', variable_value, detailed=True)`
       on every variable that has real explanatory value -- this is the ONLY source
       of concrete data the Explainer Agent has to cite specific numbers/names from
       later, so under-logging here directly causes vague, generic explanations.
       At minimum, always log with `detailed=True`:
       - The mined rules / candidate set BEFORE final filtering (e.g. the full
         itemset-support-confidence-lift table), not just its length.
       - The actual similarity/confidence/lift/support SCORES computed for the
         top candidates, with the item names or URIs they belong to -- never just
         a count like "20 candidates found".
       - The final ranked list right before it's cut down to the returned `solution`.
       - Any threshold-relaxation step (log the threshold value tried AND how many
         candidates it produced, at each tier).
       A log file that only shows shapes and counts forces the Explainer to write a
       vague, non-specific explanation -- log_var's detailed mode is intentionally
       generous (up to 25 rows of a DataFrame, up to 3000 characters of a
       list/string) specifically so real numbers and names are actually available.
    7. **No Silent Fallback To A Different Strategy (CRITICAL -- AUTOMATICALLY
       DETECTED AND REJECTED, DO NOT ATTEMPT IT)**: If, after correctly implementing
       THIS algorithm's real logic against the real data, there are genuinely zero
       qualifying candidates (e.g., no other users share enough rated items for
       Collaborative Filtering to find neighbors, or no co-occurrence data exists for
       Association Rule Mining's seed item), that is a valid, honest outcome -- do
       **NOT** write ANY code path that retrieves a different set of items as a
       substitute when this happens. This specifically means: do not query for or
       return globally popular items, top-rated items, "high-quality image-bearing
       products", items from "the same broad domain/category", or any other
       generic/default recommendation -- for ANY reason, under ANY variable or
       function name, calling it by ANY name (e.g. "category-based similarity",
       "semantic shift", a "secondary scan", or anything else), even if you never
       use the word "fallback" at all. Renaming the concept does not help: if your
       logs honestly report that this algorithm's real computation found nothing
       (e.g. an empty co-occurrence table, "0 rows", "no similar users found"), and
       `solution` ends up non-empty anyway, that contradiction alone is
       automatically detected and rejected -- regardless of what you call whatever
       produced those items. A generated script is automatically scanned for this
       pattern (in its source AND its execution logs) and will be rejected outright
       if found, so there is no benefit to writing one, under any name.
       Doing this misrepresents which algorithm actually produced the
       result, which is worse than an honest empty result. Instead, in this
       situation:
       - Set `solution = []` (an empty list) -- do not populate it with anything else.
       - Also set a variable `no_results_reason` (a short string) explaining briefly
         why, e.g. `no_results_reason = "No other users share enough rated items with
         this profile for Item-Based Collaborative Filtering to find neighbors."`
       - Use `logger.warning(...)` to record this in the execution trace as well,
         but do NOT use the word "fallback" or describe retrieving substitute items
         -- just state plainly that no qualifying candidates were found and why.
       An honest empty result with a clear reason is a correct, successful run of
       this script; a fabricated substitute using different criteria is not, and
       will always be caught and rejected.

    ### 6. OUTPUT CONTRACT (CRITICAL -- STRICTLY ENFORCED, NON-NEGOTIABLE)
    Assign the final result to a variable named `solution`. `solution` MUST be a
    Python `list` of `dict`, where **each dict is exactly one recommended PRODUCT
    ITEM** -- never anything else. Concretely:

    - **ITEMS ONLY.** Every element of `solution` must be a single `schema:Product`
      instance from the knowledge graph (the thing the user would actually receive
      or buy). NEVER return:
        * categories, genres, tags, or any grouping/aggregate value,
        * users, user profiles, or user URIs,
        * raw counts, scores, or matrices with no product identity attached,
        * duplicate rows for the same product.
    - **IMAGES ONLY.** Every dict in `solution` MUST include a non-empty `"image"`
      key holding the product's `schema:image` URL, resolved via the Dependent
      Query Strategy above. If a candidate item has NO `schema:image` value in the
      graph, DROP that candidate from the final list -- do not include it with a
      blank/placeholder image, and do not stop early; use the over-fetched pool
      from step 3 to backfill with the next-best candidate that DOES have an image.
    - **Required shape** for every element of `solution`:
      ```python
      {{"item": "<product display name>", "uri": "<product URI>", "image": "<schema:image URL>"}}
      ```
      `item` and `image` are STRICTLY mandatory and must be non-empty strings -- a
      missing or empty one on ANY element fails the whole run. Include `uri` whenever
      you have it (it improves downstream matching precision), but it is not itself a
      pass/fail condition -- **and it MUST be a real URI your script obtained from a
      `run_sparql(...)` result** (a `?product`/`?item`-type binding), never a value you
      constructed or guessed (e.g. by pattern-matching an ASIN into a URL). The caller
      verifies every `uri` you return against the URIs `run_sparql()` actually returned
      during this run, and silently drops any that don't match -- so an invented URI
      doesn't reach the user, but it does mean that field is wasted. If you don't have
      a real resolved URI for an item, omit the `uri` key entirely rather than guessing.
    - If, after using the full over-fetched pool, fewer than the target number of
      items have images, return however many qualifying items you found (a shorter
      but 100% image-complete list) rather than padding with imageless items.
    - **Genuinely empty is allowed.** If the algorithm's real logic -- after a
      correct, honest attempt -- finds zero qualifying candidates at all (not "zero
      after under-fetching," a real zero), set `solution = []` and `no_results_reason`
      as described in section 5, step 7 above. This is NOT an OUTPUT CONTRACT
      violation. Do not manufacture items from a different strategy just to avoid
      an empty list.
    """

    if previous_code and error_log:
        prompt = f"""
        {base_prompt}

        ### 7. ! CRITICAL FIX INSTRUCTIONS !
        Your previous attempt FAILED. You must fix the code based on the error below.

        **PREVIOUS CODE:**
        ```python
        {previous_code}
        ```

        **ERROR LOG:**
        {error_log}

        **TASK:**
        Rewrite the script to fix the error above. Ensure imports are correct and logic handles the data shape correctly.
        If the error indicates a violation of the OUTPUT CONTRACT (section 6) -- e.g. non-item
        entries, missing images, or items without a resolved `schema:image` -- fix the query/
        filtering logic so every element of `solution` is a real product item with an image.
        Ensure you keep the mandatory logger setup!
        Return ONLY the corrected Python code.
        """
    elif existing_working_code:
        prompt = f"""
        {base_prompt}

        ### 7. ! MIGRATE EXISTING SCRIPT !
        A script for this algorithm exists on disk from a previous run, but it hardcodes a
        specific user's profile data as literals instead of reading it from
        `USER_PROFILE_PATH` at runtime (see section 4 above) -- so it cannot safely be
        reused as-is for other users, and is being sent to you for a one-time rewrite
        instead of being executed directly.

        **CURRENT USER PROFILE (schema/shape reference only -- do NOT copy these specific
        values into the script as literals):**
        {json.dumps(user_profile, indent=2)}

        **PREVIOUS SCRIPT:**
        ```python
        {existing_working_code}
        ```

        **TASK:**
        Rewrite the script so it keeps the same overarching algorithm logic, but replaces
        every hardcoded profile/history/preference literal with a read from the
        `USER_PROFILE_PATH` JSON file instead (see section 4 above for the exact loading
        code). After this rewrite, the exact same script file must produce correct results
        for ANY future user profile, not just the one shown above, since it will be cached
        and reused verbatim without being sent back to an LLM.
        Make sure the script is fully runnable and still complies with the OUTPUT CONTRACT
        in section 6: `solution` must be product items only (no categories/users), each
        with a non-empty `image`, filtered/over-fetched exactly as described there.
        Ensure you keep the mandatory logger setup!
        Return ONLY the updated Python code.
        """
    else:
        # Generate from Scratch
        prompt = f"""
        {base_prompt}

        **CURRENT USER PROFILE (schema/shape reference only -- read via `USER_PROFILE_PATH`
        at runtime, do NOT copy these specific values into the script as literals):**
        {json.dumps(user_profile, indent=2)}

        Return ONLY the Python code.
        """
    
    response = model.generate_content(prompt)
    return response.text.replace("```python", "").replace("```", "").strip()

# ================= 4. THE EXECUTOR =================

def validate_solution(data):
    """
    Enforces the OUTPUT CONTRACT from generate_dynamic_script() in code, not just in
    the prompt -- so a non-compliant script fails validation and is fed back into the
    existing 3-attempt retry loop with a precise error, instead of silently accepting
    categories/users/imageless rows as if they were valid recommendations.

    Returns (True, None) if valid, or (False, "<reason>") if not.

    NOTE (2026-07-14 fix): an empty `solution` is intentionally NOT treated as a
    violation here anymore -- see execute_code() below, which handles a genuinely
    empty, honestly-reasoned result as its own "no_results" outcome instead of
    retry-fodder. Retrying an empty result used to pressure the composer LLM into
    fabricating a fallback (e.g. globally popular items) just to produce something
    non-empty, which silently misrepresented which algorithm actually ran.
    """
    if not isinstance(data, list):
        return False, f"OUTPUT CONTRACT VIOLATION: `solution` must be a list, got {type(data).__name__}."

    seen_uris = set()
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            return False, f"OUTPUT CONTRACT VIOLATION: element {i} of `solution` is a {type(entry).__name__}, not a dict. Every element must be one product item dict with 'item', 'uri', and 'image' keys."

        name = entry.get("item") or entry.get("name") or entry.get("title")
        if not name or not str(name).strip():
            return False, f"OUTPUT CONTRACT VIOLATION: element {i} of `solution` has no non-empty 'item' (product name). This looks like it may not be a real product item -- do not return categories, users, or aggregates."

        image = entry.get("image")
        if not image or not str(image).strip():
            return False, f"OUTPUT CONTRACT VIOLATION: element {i} ('{name}') of `solution` has no non-empty 'image'. Every returned item MUST have a resolved schema:image URL -- drop imageless candidates and backfill from the over-fetched pool instead of including them."

        uri = entry.get("uri")
        if uri:
            if uri in seen_uris:
                return False, f"OUTPUT CONTRACT VIOLATION: duplicate product URI '{uri}' in `solution`. Each item must appear only once."
            seen_uris.add(uri)

    return True, None

def strip_unverified_uris(solution, verified_uris):
    """Defends against the composer LLM inventing a plausible-looking product URI that
    was never actually returned by run_sparql() -- the only source of truth for real
    KG items (this is the root cause behind cases like a recommended item's URI not
    existing in Fuseki at all). `uri` is optional in
    the OUTPUT CONTRACT, so we don't fail the whole item over this -- we just remove
    the untrustworthy `uri` field and keep the item's name/image, and report how many
    were stripped so it's visible in the response/logs rather than silently dropped."""
    stripped = 0
    cleaned = []
    for entry in solution:
        if isinstance(entry, dict) and entry.get("uri") and entry["uri"] not in verified_uris:
            entry = dict(entry)
            entry.pop("uri", None)
            stripped += 1
        cleaned.append(entry)
    return cleaned, stripped

def profile_looks_hardcoded(script_code, user_profile):
    """Static-text safety net (2026-07-14 addition): scans the generated script's
    SOURCE TEXT (before it's ever executed) for the current user's actual rated item
    names appearing as literal strings -- i.e. the LLM copying real profile data into
    the script instead of reading it from USER_PROFILE_PATH/USER_PROFILE_HISTORY at
    runtime. This exists because a plain instruction not to do this isn't always
    followed reliably, and a script with profile data baked in would silently give
    every future user the same, stale recommendation once cached and reused.
    Returns the first offending item name found, or None if the script looks clean.
    """
    history = (user_profile or {}).get("history", [])
    for entry in history:
        name = (entry or {}).get("item_id") or (entry or {}).get("item")
        # Require a reasonably specific name (short/generic names like "Soap" could
        # coincidentally appear in comments/URIs and would false-positive).
        if name and len(name) >= 6 and name in script_code:
            return name
    return None

# Phrases indicating the script has fallen back to a DIFFERENT strategy (e.g.
# globally popular/generic items) instead of honestly reporting a genuinely empty
# result -- see script_contains_forbidden_fallback() below. Kept as a module-level
# list (rather than one giant regex) so it's easy to see/extend exactly what's
# being screened for.
FORBIDDEN_FALLBACK_PHRASES = [
    "fallback mechanism",
    "fallback trigger",
    "fallback strategy",
    "triggered a fallback",
    "trigger the fallback",
    "fallback recommendation",
    "default recommendation",
    "generic recommendation",
    "backup strategy",
    "backup recommendation",
    # 2026-07-14 round 3 additions: the LLM found ways to describe the exact same
    # disallowed behavior without using the word "fallback" at all (e.g. "Semantic
    # Shift... Category-based similarity... highest-rated products within the same
    # Health & Personal Care category"). A phrase blacklist can never be complete
    # against a model that can endlessly rephrase -- see
    # primary_signal_empty_but_solution_nonempty() below for a much more robust,
    # wording-independent check -- but these are added too since they cost nothing.
    "category-based similarity",
    "semantic shift",
    "within the same",
    "same broad category",
    "same broad domain",
    "broad category",
    "broad domain",
    "highest-rated products",
    "high-quality, image-bearing",
]

# Patterns in the runtime logs indicating the algorithm's OWN primary computation
# (the actual co-occurrence/similarity/feature-match search this algorithm exists to
# do) came back empty. If the final `solution` is non-empty anyway, that is a
# logical contradiction -- the extra items had to come from SOMEWHERE else, which is
# exactly the "ran a second, different query when the real one came up empty"
# pattern this whole set of fixes targets. This does not depend on the LLM's choice
# of words for what it did next, only on it having HONESTLY logged that the real
# computation found nothing -- which the mandatory logging setup already encourages
# it to do via log_var()/logger calls.
EMPTY_PRIMARY_SIGNAL_PATTERNS = [
    r"\(0,\s*0\)",              # pandas empty DataFrame shape repr, e.g. "Shape: (0, 0)"
    # The word boundary is load-bearing. Without it this matched the "0" inside
    # "20 rows", so log_var's own "Full data (all 20 rows)" line -- printed on a
    # perfectly successful run -- was read as evidence that the computation had
    # found nothing, and every result of exactly 10, 20, 30 ... items was
    # rejected as a contract violation.
    r"\b0\s+rows\b",
    r"no pairs? found",
    r"no rules? found",
    r"no co-?occurrences?",
    r"no association rules?",
    r"no similar users? found",
    r"no candidates? found",
    r"empty dataframe",
]

def primary_signal_empty_but_solution_nonempty(logs, solution, recorder=None):
    """Wording-independent structural check (2026-07-14, round 3): if the script's
    OWN logs report that its real computation found nothing, but `solution` ended up
    non-empty anyway, those items could not have legitimately come from that
    computation -- regardless of what the script's comments/variable names call
    whatever produced them. Returns the matched pattern, or None if no contradiction
    is detected.

    2026-09-01: given recorded step outputs, the recorded evidence is used in
    preference to the log regex.

    The regex scans the ENTIRE runtime log, so it cannot tell an intermediate
    query that legitimately returned nothing -- on the way to a real result --
    from the primary computation actually coming back empty. That was tolerable
    while scripts logged little. Once steps declared their inputs and outputs,
    scripts began logging far more intermediate state, and the check started
    firing on healthy runs: a first-pass query returning "0 rows" before the
    step relaxed its threshold and succeeded is exactly the behaviour the step
    constraints ASK for.

    When the script has recorded its step outputs, the final step's output size
    answers the question directly:

        final output empty + non-empty solution  -> a real contradiction, and
          one the regex would have missed entirely if the script never printed
          a matching phrase
        final output non-empty                   -> the primary computation
          demonstrably produced the result, so an earlier "0 rows" in the log
          was an intermediate step, not the outcome

    This is strictly more accurate than the regex in both directions, not more
    permissive. Where no step outputs were recorded -- an older cached script,
    or one that never calls log_step_io -- the regex behaviour is unchanged.
    """
    if not isinstance(solution, list) or len(solution) == 0:
        return None  # nothing to contradict

    if recorder is not None:
        try:
            final_empty = recorder.final_primary_output_empty()
        except Exception:
            final_empty = None
        if final_empty is True:
            return "final recorded step produced no output"
        if final_empty is False:
            return None

    logs_lower = (logs or "").lower()
    for pattern in EMPTY_PRIMARY_SIGNAL_PATTERNS:
        if re.search(pattern, logs_lower):
            return pattern
    return None

def script_contains_forbidden_fallback(script_code, logs=""):
    """Code-level enforcement of the "no silent fallback to a different strategy"
    rule (2026-07-14 addition, round 2). The prompt-level instruction alone wasn't
    reliable enough -- the composer LLM kept writing an explicit fallback (e.g.
    "retrieve high-quality, image-bearing products within the same broad domain")
    when its real algorithm found nothing, returning it as if it were a genuine,
    non-empty `solution`. That sails straight through validate_solution() (it's
    structurally a fine list of items with images), so this specifically screens
    the script's SOURCE TEXT and its captured runtime logs for language indicating
    a fallback-to-a-different-strategy occurred, regardless of whether `solution`
    ended up empty or not. Returns the offending phrase, or None if clean.
    """
    haystack = f"{script_code}\n{logs}".lower()
    for phrase in FORBIDDEN_FALLBACK_PHRASES:
        if phrase in haystack:
            return phrase
    return None

def script_hardcodes_profile_filename(script_code):
    """Detects a specific anti-pattern seen in production: a script using a
    hardcoded, guessed filename (e.g. `'user_profile.json'`) to load the user's
    profile, instead of reading the path from the correct, injected
    `USER_PROFILE_PATH` variable -- even though the literal string
    "USER_PROFILE_PATH" might ALSO appear elsewhere in the same script (e.g. in a
    comment/docstring, or an unused reference), which is exactly what let this slip
    past the looser "is USER_PROFILE_PATH mentioned anywhere in the source"
    cache-reuse check: the script LOOKED compliant by that check, but actually
    loaded from a different, guessed filename that doesn't exist, so the real
    profile data was never loaded at all -- `user_history` silently ended up empty.

    2026-07-16 update: broadened to match the guessed filename literal ANYWHERE in
    the source, not just directly inside an `open(...)` call -- the first version
    of this check missed a case where the script assigned the guessed filename to a
    variable first (e.g. `PROFILE_FILE = "user_profile.json"`) and passed THAT
    variable to `open()`, which the narrower direct-literal-only regex didn't catch.
    A compliant script never needs to write a profile-like filename as a string
    literal anywhere at all -- it only ever references the bare `USER_PROFILE_PATH`
    variable -- so any such literal, found anywhere, is treated as a violation.
    Returns the offending literal filename, or None if the script looks clean.
    """
    for match in re.finditer(r'[\'"]([^\'"]*\.json)[\'"]', script_code):
        literal = match.group(1)
        if literal == "current_user_profile.json":
            continue  # this module's own default path pattern, not a bad guess
        lowered = literal.lower()
        if "profile" in lowered or "user_history" in lowered:
            return literal
    return None

def execute_code(script_code, user_profile=None, profile_path=None,
                 recorder=None):
    """Executes the generated code with full error capturing, then validates that
    `solution` complies with the OUTPUT CONTRACT (items only, images only).

    The current user's profile is made available to the script TWO ways (2026-07-14):
    1. `USER_PROFILE_PATH` -- a file path (string) to a JSON file containing the
       profile, written fresh before every execution. This is now the PRIMARY,
       documented mechanism (see generate_dynamic_script()'s OUTPUT CONTRACT) --
       reading from an actual file the script never had to write itself makes it
       obvious, on inspection, that the script contains no embedded copy of anyone's
       data. This directly addresses the profile being found hardcoded inside a
       generated .py file: the data now lives in its own JSON file next to the
       script, not inside it.
    2. `USER_PROFILE_HISTORY` -- the same data as an in-memory dict, kept for
       backward compatibility with scripts generated before this change.
    Either way, this is what makes a cached script (see the API endpoint below) safe
    to reuse as-is for a *different* user's request without any LLM involvement --
    the logic is cached, the data is injected/written fresh on every call.
    """
    log_capture = io.StringIO()

    hardcoded_item = profile_looks_hardcoded(script_code, user_profile)
    if hardcoded_item:
        return {
            "status": "error",
            "error": (
                f"OUTPUT CONTRACT VIOLATION: the script appears to hardcode a "
                f"specific user's profile data directly in its source (found the "
                f"literal item name '{hardcoded_item}'), instead of reading it from "
                f"USER_PROFILE_PATH / USER_PROFILE_HISTORY at runtime. This would "
                f"make the cached script wrong for every other user. Rewrite it to "
                f"read the profile only from those, with no item names, ratings, or "
                f"preference values copied in as literals."
            ),
            "logs": "",
        }

    forbidden_phrase = script_contains_forbidden_fallback(script_code)
    if forbidden_phrase:
        return {
            "status": "error",
            "error": (
                f"OUTPUT CONTRACT VIOLATION: the script's source contains fallback-"
                f"to-a-different-strategy language (found the phrase '{forbidden_phrase}'). "
                f"If the algorithm's real logic finds no qualifying candidates, set "
                f"solution = [] and no_results_reason to explain why -- do NOT "
                f"retrieve generic/popular/broad-domain items as a substitute. "
                f"Remove the fallback logic entirely and rewrite the no-candidates "
                f"case to report an honest empty result instead."
            ),
            "logs": "",
        }

    hardcoded_filename = script_hardcodes_profile_filename(script_code)
    if hardcoded_filename:
        return {
            "status": "error",
            "error": (
                f"OUTPUT CONTRACT VIOLATION: the script opens a hardcoded, guessed "
                f"filename ('{hardcoded_filename}') to load the user's profile, "
                f"instead of reading the path given to it via the injected "
                f"USER_PROFILE_PATH variable. That guessed filename does not exist, "
                f"so the profile never actually loads -- rewrite it to use exactly "
                f"`open(USER_PROFILE_PATH)`, not a literal filename string."
            ),
            "logs": "",
        }

    try:
        profile_data = user_profile or {}
        if profile_path is None:
            profile_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "current_user_profile.json"
            )
        try:
            with open(profile_path, "w", encoding="utf-8") as pf:
                json.dump(profile_data, pf)
        except Exception as e:
            logger.warning(f"Could not write USER_PROFILE_PATH file ({profile_path}): {e}")

        # WP3: the recorder and execution id are injected the same way the
        # profile is. A script that never calls log_step_io simply leaves the
        # recorder with run-level data only, which is the graceful-degradation
        # tier for cached scripts generated under the old contract.
        exec_globals = {
            "USER_PROFILE_HISTORY": profile_data,
            "USER_PROFILE_PATH": profile_path,
            "_EXECUTION_RECORDER": recorder,
            "EXECUTION_ID": getattr(recorder, "execution_id", ""),
        }
        with contextlib.redirect_stdout(log_capture):
            exec(script_code, exec_globals)
        if 'solution' not in exec_globals:
            return {"status": "error", "error": "Script did not define 'solution'.", "logs": log_capture.getvalue()}

        solution = exec_globals['solution']

        # HARDENING (2026-07-14, round 2): re-check for fallback language against
        # the ACTUAL captured runtime logs too -- the static source-text check above
        # catches most cases, but a script could in principle build the message
        # dynamically. A non-empty `solution` full of substituted generic items
        # still passes validate_solution() below (structurally it's a fine list),
        # so this is what actually catches "I got real-looking items, but they came
        # from a fallback, not the real algorithm" cases.
        runtime_logs = log_capture.getvalue()
        forbidden_phrase_runtime = script_contains_forbidden_fallback("", runtime_logs)
        if forbidden_phrase_runtime:
            return {
                "status": "error",
                "error": (
                    f"OUTPUT CONTRACT VIOLATION: the script's own execution log "
                    f"reports a fallback to a different strategy (found the phrase "
                    f"'{forbidden_phrase_runtime}'). If the algorithm's real logic "
                    f"finds no qualifying candidates, set solution = [] and "
                    f"no_results_reason to explain why -- do NOT retrieve generic/"
                    f"popular/broad-domain items as a substitute."
                ),
                "logs": runtime_logs,
            }

        # HARDENING (2026-07-14, round 3): wording-independent structural check --
        # catches the exact case that slipped past both phrase lists above (the LLM
        # described the same disallowed behavior as "Semantic Shift... Category-
        # based similarity" instead of using the word "fallback"). If the script's
        # own logs honestly report its real computation came back empty, but
        # `solution` is non-empty anyway, those items could not have legitimately
        # come from that computation, no matter what the script calls whatever
        # produced them.
        contradiction_pattern = primary_signal_empty_but_solution_nonempty(
            runtime_logs, solution, recorder)
        if contradiction_pattern:
            return {
                "status": "error",
                "error": (
                    f"OUTPUT CONTRACT VIOLATION: the script's own logs report that "
                    f"its primary computation found nothing (matched pattern "
                    f"'{contradiction_pattern}' in the logs), yet `solution` is "
                    f"non-empty. Those items could not have legitimately come from "
                    f"this algorithm's real logic -- whatever produced them (a "
                    f"second query, a different similarity measure, a category "
                    f"scan, etc.) is a forbidden substitute strategy, regardless of "
                    f"what it's called in the code. When the primary computation "
                    f"genuinely finds nothing, set solution = [] and set "
                    f"no_results_reason to explain why -- do not query for anything "
                    f"else afterward."
                ),
                "logs": runtime_logs,
            }

        # HARDENING (2026-07-13): drop any URI the script itself never actually fetched
        # from the Knowledge Graph via run_sparql() -- see strip_unverified_uris() above.
        if isinstance(solution, list):
            verified_uris = exec_globals.get('_VERIFIED_KG_URIS', set())
            solution, stripped_count = strip_unverified_uris(solution, verified_uris)
            if stripped_count:
                logger.warning(
                    f"Stripped {stripped_count} unverified URI(s) from solution -- not "
                    f"present in any run_sparql() result this script actually fetched."
                )

        is_valid, reason = validate_solution(solution)
        if not is_valid:
            return {"status": "error", "error": reason, "logs": log_capture.getvalue()}

        if isinstance(solution, list) and len(solution) == 0:
            # HARDENING (2026-07-14): a genuinely empty, honestly-reasoned result is
            # a real outcome, not a bug to paper over with a fallback -- surface it
            # plainly instead of silently returning "success" with nothing in it.
            no_results_reason = exec_globals.get(
                "no_results_reason",
                "The algorithm found no qualifying candidates for this profile.",
            )
            return {"status": "no_results", "reason": no_results_reason, "logs": log_capture.getvalue()}

        return {"status": "success", "logs": log_capture.getvalue(), "data": solution}
    except Exception as e:
        return {"status": "error", "error": f"Script Execution Failed: {str(e)}", "logs": log_capture.getvalue() + traceback.format_exc()}

def save_successful_code(script_code, algo_name):
    """Saves the working script to a .py file locally."""
    filename = f"generated_{algo_name.replace(' ', '_')}.py"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(script_code)
        logger.info(f"💾 Saved script to: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to save script: {e}")
        return None

# ================= API ENDPOINT =================

@app.route('/api/autonomous-recommend', methods=['POST'])
def autonomous_agent():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be valid JSON with an object at the top level."}), 400

        user_profile = data.get('user_profile', {})
        api_key = data.get('api_key')
        fuseki_base = data.get('fuseki_base')  # optional; falls back to DEFAULT_FUSEKI_BASE everywhere below
        # Optional dataset NAME overrides (Orchestrator's Admin > Configuration tab).
        # Fall back to this file's/the ProductReviews Agent's own defaults when absent.
        product_reviews_dataset = data.get('product_reviews_dataset')
        algorithms_dataset = data.get('algorithms_dataset')
        # Optional Agent Setup overrides (from the Orchestrator's Admin > Agent Setup tab).
        # Fall back to this file's own SELECTOR_SYSTEM_PROMPT / COMPOSER_SYSTEM_PROMPT
        # defaults when not provided, so behavior is unchanged unless someone has
        # explicitly edited these roles in the KG.
        selector_system_prompt = data.get('selector_system_prompt')
        composer_system_prompt = data.get('composer_system_prompt')
        
        if not api_key: return jsonify({"error": "Gemini API key missing"}), 400
        genai.configure(api_key=api_key)
        
        ontology_scan = discover_ontology(fuseki_base, product_reviews_dataset)
        recipes = get_available_recipes(fuseki_base, algorithms_dataset)
        if not recipes: return jsonify({"error": "No recipes found in Algorithms DB"}), 500
        
        logger.info("🤔 Analyzing user profile...")
        chosen_recipe = select_best_algorithm(user_profile, recipes, system_prompt=selector_system_prompt)
        logger.info(f"👉 Strategy Selected: {chosen_recipe}")

        algo_workflow = get_algorithm_source(chosen_recipe, fuseki_base, algorithms_dataset)

        existing_script_file = f"generated_{chosen_recipe.replace(' ', '_')}.py"
        existing_working_code = None
        if os.path.exists(existing_script_file):
            with open(existing_script_file, "r", encoding="utf-8") as f:
                existing_working_code = f.read()

        # WP3: one identifier for this run, used for the execution graph, the
        # log filename and the profile filename, and returned to the caller so
        # the Explainer can ask for exactly this run.
        execution_id = new_execution_id()
        workflow_context = get_workflow_context(
            chosen_recipe, fuseki_base, algorithms_dataset)
        recorder = ExecutionRecorder(
            chosen_recipe, execution_id=execution_id,
            workflow_uri=workflow_context.get("workflow_uri"),
            algorithm_uri=workflow_context.get("algorithm_uri"),
            step_uris=workflow_context.get("step_uris"),
            variable_uris=workflow_context.get("variable_uris"))
        recorder.start()
        recorder.log_run_io(inputs={"user_profile": user_profile})
        logger.info(f"Execution id for this run: {execution_id}")

        # CONCURRENCY FIX (WP3): this file used to be named per ALGORITHM and
        # rewritten before every execution, so two people using the same
        # algorithm at the same time could read each other's profile. That is a
        # correctness and privacy defect, not just a logging one. Naming it per
        # execution removes the race.
        profile_json_path = (
            f"generated_{chosen_recipe.replace(' ', '_')}_{execution_id}_profile.json")

        current_code = None
        execution_result = {}

        # CACHE FAST PATH: if a script for this algorithm was already generated and
        # cached in a prior run, AND it was generated under the current contract
        # (reads the profile from the injected `USER_PROFILE_HISTORY` global rather
        # than hardcoding it -- see execute_code() and generate_dynamic_script()
        # above), execute it directly for THIS request with NO LLM call at all. Once
        # a script exists per algorithm, it's just run, not regenerated.
        # Scripts saved before this change hardcode a specific user's data and are NOT
        # safe to reuse blindly; those fall through to the existing generate/repair loop
        # below, which migrates them to the new contract (see the "MIGRATE EXISTING
        # SCRIPT" prompt branch in generate_dynamic_script()) so they become fast-path
        # cacheable from then on.
        cache_is_reusable = bool(existing_working_code) and (
            "USER_PROFILE_PATH" in existing_working_code or "USER_PROFILE_HISTORY" in existing_working_code
        )
        if cache_is_reusable:
            logger.info(f"📂 Reusing cached script for {chosen_recipe} directly (no LLM call).")
            execution_result = execute_code(existing_working_code, user_profile,
                                            profile_path=profile_json_path,
                                            recorder=recorder)
            if execution_result['status'] == 'success':
                logger.info("✅ Cached script executed successfully.")
                recorder.log_run_io(outputs={"solution": execution_result['data']})
                recorder.finish(status="Completed")
                recorder.flush(fuseki_base)
                return jsonify({
                    "status": "success", "strategy": chosen_recipe, "attempts": 0,
                    "saved_script": existing_script_file, "results": execution_result['data'],
                    "logs": execution_result['logs'], "from_cache": True,
                    "execution_id": execution_id
                })
            if execution_result['status'] == 'no_results':
                # HARDENING (2026-07-14): this is the cached script correctly running
                # and finding nothing -- NOT a bug to send back for LLM "repair" (that
                # repair pressure is exactly what used to produce a fabricated
                # popular-items fallback). Surface it as a real, honest failure instead.
                reason = execution_result.get("reason", "No qualifying candidates were found for this profile.")
                logger.info(f"ℹ️ Cached script for {chosen_recipe} found no qualifying candidates: {reason}")
                # An honest empty result is a real outcome, so it is recorded as
                # a completed execution rather than discarded.
                recorder.log_run_io(outputs={"no_results_reason": reason})
                recorder.finish(status="Completed")
                recorder.flush(fuseki_base)
                return jsonify({
                    "status": "error", "message": reason, "no_results": True,
                    "strategy": chosen_recipe, "logs": execution_result.get("logs"),
                    "execution_id": execution_id
                }), 422
            logger.warning(
                f"⚠️ Cached script for {chosen_recipe} failed at runtime "
                f"({execution_result.get('error')}); falling back to LLM repair."
            )

        for attempt in range(3):
            logger.info(f"🔄 Attempt {attempt + 1}/3 for {chosen_recipe}...")
            if attempt == 0:
                if cache_is_reusable:
                    # The cache fast path above already tried this exact code and it
                    # failed -- feed it to the LLM as a repair target (same shape as the
                    # "previous_code/error_log" branch) instead of the "adjust" branch.
                    current_code = generate_dynamic_script(user_profile, ontology_scan, chosen_recipe, algo_workflow, previous_code=existing_working_code, error_log=execution_result.get('logs', '') + "\n" + execution_result.get('error', ''), fuseki_base=fuseki_base, system_prompt=composer_system_prompt, product_reviews_dataset=product_reviews_dataset)
                else:
                    logger.info(f"📂 Found existing script for {chosen_recipe}. Adjusting...") if existing_working_code else None
                    current_code = generate_dynamic_script(user_profile, ontology_scan, chosen_recipe, algo_workflow, existing_working_code=existing_working_code, fuseki_base=fuseki_base, system_prompt=composer_system_prompt, product_reviews_dataset=product_reviews_dataset)
            else:
                logger.warning("⚠️ Retrying with LLM fix...")
                current_code = generate_dynamic_script(user_profile, ontology_scan, chosen_recipe, algo_workflow, previous_code=current_code, error_log=execution_result.get('logs', '') + "\n" + execution_result.get('error', ''), fuseki_base=fuseki_base, system_prompt=composer_system_prompt, product_reviews_dataset=product_reviews_dataset)

            execution_result = execute_code(current_code, user_profile,
                                            profile_path=profile_json_path,
                                            recorder=recorder)

            if execution_result['status'] == 'success':
                logger.info("✅ Execution Success!")
                saved_file = save_successful_code(current_code, chosen_recipe)
                recorder.log_run_io(outputs={"solution": execution_result['data']})
                recorder.finish(status="Completed")
                recorder.flush(fuseki_base)
                return jsonify({
                    "status": "success", "strategy": chosen_recipe, "attempts": attempt + 1,
                    "saved_script": saved_file, "results": execution_result['data'],
                    "logs": execution_result['logs'], "execution_id": execution_id
                })

            if execution_result['status'] == 'no_results':
                # HARDENING (2026-07-14): a genuinely empty, honestly-reasoned result is
                # a correct run of the script, not something 3 retries will "fix" --
                # retrying it just pressures the LLM to fabricate a fallback strategy
                # (e.g. globally popular items) to force a non-empty result, which is
                # exactly the silent-fallback behavior this fix removes. The script is
                # still worth caching (it's correct), so save it, then fail honestly.
                reason = execution_result.get("reason", "No qualifying candidates were found for this profile.")
                logger.info(f"ℹ️ {chosen_recipe} found no qualifying candidates: {reason}")
                saved_file = save_successful_code(current_code, chosen_recipe)
                recorder.log_run_io(outputs={"no_results_reason": reason})
                recorder.finish(status="Completed")
                recorder.flush(fuseki_base)
                return jsonify({
                    "status": "error", "message": reason, "no_results": True,
                    "strategy": chosen_recipe, "saved_script": saved_file,
                    "logs": execution_result.get("logs"),
                    "execution_id": execution_id
                }), 422
        
        logger.error("❌ All attempts failed.")
        # A failed run is still worth recording -- "what happened on the run
        # that produced nothing" is a question the evaluation study will ask.
        recorder.finish(status="Cancelled")
        recorder.flush(fuseki_base)
        return jsonify({"status": "error", "message": "Failed after 3 retries",
                        "final_error": execution_result.get('error'),
                        "logs": execution_result.get('logs'),
                        "execution_id": execution_id}), 500
    except Exception as e:
        logger.exception("Critical error in Recommender")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Recommender Agent on port 5003...")
    app.run(debug=True, use_reloader=False, port=5003)