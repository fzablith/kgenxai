import google.generativeai as genai
from flask import Flask, request, jsonify
import requests
import json
import os
import glob
import traceback
import logging

# ================= CONFIGURATION & LOGGING =================
app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | EXPLAINER_AGENT | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Namespaces, dataset names and credentials all come from kgenxai_config.
# The Fuseki password used to be a literal here; it is now read from the
# environment (FUSEKI_USER / FUSEKI_PASSWORD) so the source can be published.
from kgenxai_config import (
    AUTH, DEFAULT_FUSEKI_BASE, DEFAULT_ALGORITHMS_DATASET,
    DEFAULT_EXPLANATIONS_DATASET, DEFAULT_EXECUTIONS_DATASET,
    NS, MLS_TERMS, endpoint, sparql_prefixes,
)

# ================= AGENT IDENTITY (SYSTEM PROMPT) =================
# The agent's ROLE is defined once, here, separate from the per-call task content
# (which stays dynamic: EO explanation types, rules, and required sections below
# still depend on what's actually in the KG for this request). This is the
# default/fallback identity; the Orchestrator's Admin > Agent Setup tab can override
# it by passing `explainer_system_prompt` in the request body.
EXPLAINER_SYSTEM_PROMPT = (
    "You are an advanced Explainable AI (XAI) Agent natively aligned with the "
    "Explanation Ontology (EO). Your goal is to provide fluid, data-driven, and highly "
    "structured explanations based on a user's conversational query."
)

PRODUCT_REVIEWS_API = "http://127.0.0.1:5002/api/execute_safe_sparql"  # Ping the ProductReviews Agent for XAI_ProductReviews queries

DEFAULT_ALGORITHMS_DATASET_NAME = DEFAULT_ALGORITHMS_DATASET
DEFAULT_EXPLANATIONS_DATASET_NAME = DEFAULT_EXPLANATIONS_DATASET

def _endpoints_for(fuseki_base, algorithms_dataset=None, explanations_dataset=None):
    """Builds the Algorithms and Explanations dataset endpoints from a given base URL
    and (optionally renamed, via Admin > Configuration) dataset names."""
    base = (fuseki_base or DEFAULT_FUSEKI_BASE).strip().rstrip("/")
    algo_name = (algorithms_dataset or "").strip() or DEFAULT_ALGORITHMS_DATASET_NAME
    expl_name = (explanations_dataset or "").strip() or DEFAULT_EXPLANATIONS_DATASET_NAME
    return {
        "algorithms": f"{base}/{algo_name}/query",
        "explanations": f"{base}/{expl_name}/query",
    }


def _executions_endpoint(fuseki_base=None, executions_dataset=None):
    """Query endpoint for XAI_ExecutionLogs (decision D1: its own dataset)."""
    return endpoint(fuseki_base, executions_dataset,
                    DEFAULT_EXECUTIONS_DATASET, "query")


# ================= HELPERS =================

def fetch_explanation_types(fuseki_base=None, explanations_dataset=None):
    explanations_endpoint = _endpoints_for(fuseki_base, explanations_dataset=explanations_dataset)["explanations"]
    # EO defines no Explanation class; its root explanation class is the dedalo
    # ep:Explanation that EO imports (defect 4). The graph is migrated to match,
    # so there is exactly one form to match here.
    query = f"""
    {sparql_prefixes('rdfs', 'eo', 'ep', 'ex')}

    SELECT ?name ?desc ?questions ?action WHERE {{
        ?s a ep:Explanation ;
           rdfs:label ?name ;
           rdfs:comment ?desc ;
           ex:exampleQuestions ?questions ;
           ex:llmAction ?action .
    }} ORDER BY ?name
    """
    try:
        response = requests.post(explanations_endpoint, data={'query': query}, auth=AUTH, timeout=10)
        if response.status_code == 200:
            results = []
            for b in response.json().get('results', {}).get('bindings', []):
                results.append({
                    "name": b['name']['value'],
                    "desc": b['desc']['value'],
                    "questions": b['questions']['value'],
                    "action": b['action']['value']
                })
            return results
        return []
    except Exception as e:
        logger.error(f"Error fetching explanation types from DB: {e}")
        return []

def fetch_item_context(item_uri, fuseki_base=None, product_reviews_dataset=None):
    """Retrieves properties of an item by querying the ProductReviews Agent."""
    query = f"SELECT ?p ?o WHERE {{ <{item_uri}> ?p ?o . }}"
    try:
        response = requests.post(
            PRODUCT_REVIEWS_API,
            json={'query': query, 'fuseki_base': fuseki_base, 'product_reviews_dataset': product_reviews_dataset},
            timeout=10,
        )
        if response.status_code == 200:
            results = response.json().get('results', {}).get('bindings', [])
            attributes = {}
            for res in results:
                pred = res['p']['value']
                obj = res['o']['value']
                key_name = pred.split('#')[-1].split('/')[-1] 
                attributes[key_name] = obj
            return attributes
        return {}
    except Exception as e:
        logger.error(f"Error fetching context for {item_uri} via ProductReviews Agent: {e}")
        return {}

def fetch_function_context(strategy_name, fuseki_base=None, algorithms_dataset=None):
    """
    Fetches the linked pko:Step steps of the algorithm and sorts them sequence-wise
    using the pko:hasFirstStep and pko:nextStep list links.
    """
    algorithms_endpoint = _endpoints_for(fuseki_base, algorithms_dataset=algorithms_dataset)["algorithms"]
    # PKO defines no Step class and no isStepOf property (defects 2a, 2b); step
    # membership is pko:hasStep, Procedure -> Step. The structured fields (WP2)
    # are OPTIONAL so a step that has not been enriched still returns exactly
    # what it returned before.
    query = f"""
    {sparql_prefixes('rdf', 'rdfs', 'mls', 'pko', 'pplan', 'dcterms', 'ex')}

    SELECT ?step ?comment ?description ?next ?first ?stepNumber
           (GROUP_CONCAT(DISTINCT ?inLabel;  separator=", ") AS ?inputs)
           (GROUP_CONCAT(DISTINCT ?outLabel; separator=", ") AS ?outputs)
           (GROUP_CONCAT(DISTINCT ?paramText; separator="; ") AS ?parameters)
    WHERE {{
        ?method a mls:Algorithm ;
                rdfs:label "{strategy_name}" .
        ?workflow mls:implements ?method .

        ?workflow pko:hasStep ?step .
        ?step rdfs:comment ?comment .

        OPTIONAL {{ ?step dcterms:description ?description }}
        OPTIONAL {{ ?step pko:stepNumber ?stepNumber }}
        OPTIONAL {{ ?step pplan:hasInputVar  ?inVar  . ?inVar  rdfs:label ?inLabel  }}
        OPTIONAL {{ ?step pplan:hasOutputVar ?outVar . ?outVar rdfs:label ?outLabel }}
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
    GROUP BY ?step ?comment ?description ?next ?first ?stepNumber
    """
    try:
        # Force Fuseki to return JSON
        headers = {'Accept': 'application/sparql-results+json'}
        response = requests.post(algorithms_endpoint, data={'query': query}, auth=AUTH, headers=headers, timeout=10)
        if response.status_code == 200:
            bindings = response.json().get('results', {}).get('bindings', [])
            if not bindings:
                logger.warning(f"No workflow steps found in algorithms graph for strategy: {strategy_name}")
                return "Unknown Workflow"
            
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
                    # WP2: structured fields, absent on un-migrated steps.
                    'description': b.get('description', {}).get('value'),
                    'inputs': b.get('inputs', {}).get('value'),
                    'outputs': b.get('outputs', {}).get('value'),
                    'parameters': b.get('parameters', {}).get('value'),
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

            # Absolute fallback
            if not first_step_uri and steps_dict:
                first_step_uri = list(steps_dict.keys())[0]

            ordered_steps = []
            current_uri = first_step_uri
            visited = set()
            step_num = 1

            while current_uri and current_uri in steps_dict and current_uri not in visited:
                visited.add(current_uri)
                step_info = steps_dict[current_uri]
                ordered_steps.append(_render_step(step_num, step_info))
                current_uri = step_info['next']
                step_num += 1

            # Append orphans
            for step_uri, step_info in steps_dict.items():
                if step_uri not in visited:
                    ordered_steps.append(
                        _render_step(step_num, step_info, orphan=True))
                    step_num += 1

            return "\n".join(ordered_steps)
    except Exception as e:
        logger.error(f"Error fetching function context: {e}")
    return "Unknown Workflow"

def _render_step(step_num, step_info, orphan=False):
    """Render one workflow step for the explanation prompt.

    WP2 gives steps declared inputs, outputs and parameters. Where those exist
    they are stated explicitly; where they do not, this emits exactly the same
    single line it emits today. That fallback is what lets the procedural
    enrichment be migrated one algorithm at a time without the Explainer
    behaving differently for the ones not yet done.
    """
    marker = " (orphan)" if orphan else ""
    body = step_info.get('description') or step_info['comment']
    lines = [f"Step {step_num}{marker}: {body}"]
    if step_info.get('inputs'):
        lines.append(f"    Inputs: {step_info['inputs']}")
    if step_info.get('outputs'):
        lines.append(f"    Outputs: {step_info['outputs']}")
    if step_info.get('parameters'):
        lines.append(f"    Parameters: {step_info['parameters']}")
    return "\n".join(lines)


def fetch_execution_trace(execution_id, fuseki_base=None, executions_dataset=None):
    """Read a recorded execution from XAI_ExecutionLogs (WP3/WP4).

    Returns text in the SAME shape the prompt already expects for
    "Execution Trace Logs", deliberately: the smallest possible change to the
    model's input is the strongest guarantee that its output does not drift.
    The prompt, its heading and the four-section output contract are untouched.

    Returns None when there is no usable trace, so the caller falls through to
    the log files exactly as before.
    """
    if not execution_id:
        return None

    query = f"""
    {sparql_prefixes('rdf', 'rdfs', 'dcterms', 'pko', 'pplan', 'mls',
                     'prov', 'adms', 'ex')}

    SELECT ?stepNumber ?stepLabel ?description ?started ?ended
           ?varLabel ?direction ?value ?summary
    WHERE {{
        ?execution dcterms:identifier "{execution_id}" .
        ?stepExec pko:isIncludedInProcedureExecution ?execution .
        # hasExecutedStep is OPTIONAL on purpose. It was required, which meant
        # that if a run was recorded without its link back to the declared step
        # -- as happened when the recorder was built without the workflow URI --
        # this query matched nothing, returned None, and the agent fell back to
        # the log file without any sign that the execution graph held data.
        # A partial trace is far more useful than a silent fallback.
        OPTIONAL {{ ?stepExec pko:hasExecutedStep ?step }}
        OPTIONAL {{ ?stepExec pko:stepNumber ?execStepNumber }}
        OPTIONAL {{ ?step pko:stepNumber ?definedStepNumber }}
        BIND(COALESCE(?definedStepNumber, ?execStepNumber) AS ?stepNumber)
        OPTIONAL {{ ?step rdfs:label ?stepLabel }}
        OPTIONAL {{ ?step dcterms:description ?description }}
        OPTIONAL {{ ?stepExec prov:startedAtTime ?started }}
        OPTIONAL {{ ?stepExec prov:endedAtTime ?ended }}
        OPTIONAL {{
            {{
                ?stepExec prov:used ?entity .
                BIND("input" AS ?direction)
            }} UNION {{
                ?entity prov:wasGeneratedBy ?stepExec .
                BIND("output" AS ?direction)
            }}
            OPTIONAL {{ ?entity pplan:correspondsToVariable ?var .
                        ?var rdfs:label ?varLabel }}
            OPTIONAL {{ ?entity <{MLS_TERMS['has_value']}> ?value }}
            OPTIONAL {{ ?entity ex:valueSummary ?summary }}
        }}
    }}
    ORDER BY ?stepNumber ?direction
    """
    try:
        endpoint_url = _executions_endpoint(fuseki_base, executions_dataset)
        headers = {'Accept': 'application/sparql-results+json'}
        response = requests.post(endpoint_url, data={'query': query},
                                 auth=AUTH, headers=headers, timeout=20)
        if response.status_code != 200:
            logger.warning(
                f"Execution graph returned HTTP {response.status_code} for "
                f"execution {execution_id}; falling back to log files.")
            return None

        bindings = response.json().get('results', {}).get('bindings', [])
        if not bindings:
            logger.info(
                f"No execution graph entries for {execution_id}; falling back "
                f"to log files.")
            return None

        steps = {}
        for b in bindings:
            num = b.get('stepNumber', {}).get('value', '?')
            entry = steps.setdefault(num, {
                'label': b.get('stepLabel', {}).get('value'),
                'description': b.get('description', {}).get('value'),
                'started': b.get('started', {}).get('value'),
                'ended': b.get('ended', {}).get('value'),
                'inputs': [], 'outputs': [],
            })
            direction = b.get('direction', {}).get('value')
            if not direction:
                continue
            record = {
                'variable': b.get('varLabel', {}).get('value', 'value'),
                'value': b.get('value', {}).get('value'),
                'summary': b.get('summary', {}).get('value'),
            }
            bucket = entry['inputs'] if direction == 'input' else entry['outputs']
            if record not in bucket:
                bucket.append(record)

        def sort_key(k):
            try:
                return (0, int(k))
            except (TypeError, ValueError):
                return (1, str(k))

        lines = [f"EXECUTION TRACE (execution id: {execution_id})",
                 "Recorded as a knowledge graph; every value below is read from "
                 "the recorded execution, not reconstructed from prose."]
        for num in sorted(steps, key=sort_key):
            info = steps[num]
            header = f"\nSTEP {num}"
            if info.get('label'):
                header += f": {info['label']}"
            lines.append(header)
            if info.get('description'):
                lines.append(f"  Description: {info['description']}")
            if info.get('started'):
                timing = f"  Started: {info['started']}"
                if info.get('ended'):
                    timing += f" | Ended: {info['ended']}"
                lines.append(timing)
            for direction, key in (("INPUT", 'inputs'), ("OUTPUT", 'outputs')):
                for record in info[key]:
                    lines.append(f"  {direction} [{record['variable']}]")
                    if record.get('summary'):
                        lines.append(f"    Summary: {record['summary']}")
                    if record.get('value'):
                        lines.append(f"    Value: {record['value']}")

        trace = "\n".join(lines)
        logger.info(
            f"Loaded execution trace from the knowledge graph "
            f"({len(steps)} step(s), {len(trace)} chars).")
        return trace
    except Exception as e:
        logger.warning(
            f"Could not read the execution graph for {execution_id} ({e}); "
            f"falling back to log files.")
        return None


def fetch_execution_log(strategy_name, execution_id=None):
    if not strategy_name or strategy_name == 'Unknown':
        logger.warning("Strategy name is unknown. Cannot fetch execution logs.")
        return "No execution logs available."

    # Bug fix (2026-07): strategy_used sometimes arrives with incidental leading/
    # trailing whitespace or different casing than what the Recommender agent used
    # to name the file (execution_trace_{algo_name.replace(' ', '_')}.log). An exact
    # mismatch here used to silently fall through to "not found," which is what was
    # causing the Mathematical Rationale section to go missing for some explanation
    # requests during testing -- the LLM had no log content to extract scores from,
    # so it just omitted the section. This normalizes the name and, if the exact
    # file still isn't found, falls back to a case-insensitive glob match on disk
    # before giving up. Behavior for the exact-match case (the common path) is
    # unchanged.
    clean_strategy_name = strategy_name.strip()
    log_filename = f"execution_trace_{clean_strategy_name.replace(' ', '_')}.log"

    # CONCURRENCY FIX (WP3, Action Item 6): the legacy filename is per ALGORITHM
    # and the Recommender opens it with mode='w', so two people using the same
    # algorithm at the same time overwrite each other's trace -- and whoever
    # asks for an explanation second can be handed the other person's run. When
    # an execution id is available we resolve the per-execution file first,
    # which is unique per run. The legacy file is still written and still read
    # as a fallback, so nothing that depends on it breaks.
    resolved_path = None
    if execution_id:
        per_execution = (
            f"execution_trace_{clean_strategy_name.replace(' ', '_')}"
            f"_{execution_id}.log")
        if os.path.exists(per_execution):
            resolved_path = per_execution
            logger.info(f"Using per-execution log: {per_execution}")

    if resolved_path is None:
        logger.info(f"Looking for Recommender logs at: {log_filename}")

    if resolved_path:
        pass
    elif os.path.exists(log_filename):
        resolved_path = log_filename
    else:
        # Fallback: case-insensitive match against any execution_trace_*.log file
        # in the working directory, in case of a naming mismatch (casing/whitespace).
        candidates = glob.glob("execution_trace_*.log")
        target = log_filename.lower()
        for candidate in candidates:
            if candidate.lower() == target:
                resolved_path = candidate
                logger.info(f"Resolved log file via case-insensitive fallback: {candidate}")
                break

    if resolved_path:
        try:
            with open(resolved_path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"Successfully loaded log file. Size: {len(content)} chars.")
                # 2026-07-23: raised from 60,000 -- log_var() on the Recommender side
                # was changed to ALWAYS emit the full, untruncated value for every
                # logged variable (previously capped at 300/3000 chars per call,
                # which is what made the Mathematical Rationale section thin/vague).
                # Logs are expected to run considerably larger now, so this cap is
                # raised well past typical log sizes; it still exists purely as a
                # safety net against a truly pathological single execution, not as a
                # routine truncation path.
                if len(content) > 300000:
                    logger.info("Log file too large. Truncating to last 300,000 chars.")
                    return "...[TRUNCATED]...\n" + content[-300000:]
                return content
        except Exception as e:
            logger.error(f"Error reading log file: {str(e)}")
            return f"Error reading log file: {str(e)}"

    logger.warning(f"Log file '{log_filename}' NOT FOUND (also checked case-insensitive matches).")
    return f"Execution log file '{log_filename}' not found."


# ================= CORE =================

@app.route('/api/explain', methods=['POST'])
def generate_explanation():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be valid JSON with an object at the top level."}), 400

        api_key = data.get('api_key')
        if not api_key: return jsonify({"error": "Gemini API key missing"}), 400
        
        # Configure Gemini per request dynamically
        genai.configure(api_key=api_key)
        
        user_profile = data.get('user_profile', {})
        recommendation_result = data.get('recommendation_result', {})
        if not isinstance(recommendation_result, dict):
            recommendation_result = {}
        user_query = data.get('user_query', '') 
        items = recommendation_result.get('results', []) 
        if not isinstance(items, list):
            items = []
        strategy_used = recommendation_result.get('strategy', 'Unknown')
        fuseki_base = data.get('fuseki_base')  # optional; falls back to DEFAULT_FUSEKI_BASE everywhere below
        # Optional dataset NAME overrides (Orchestrator's Admin > Configuration tab).
        product_reviews_dataset = data.get('product_reviews_dataset')
        algorithms_dataset = data.get('algorithms_dataset')
        explanations_dataset = data.get('explanations_dataset')
        # WP3/WP4: optional on both sides. An older Orchestrator that does not
        # send these still works, and this agent then behaves exactly as before.
        executions_dataset = data.get('executions_dataset')
        execution_id = (recommendation_result.get('execution_id')
                        or data.get('execution_id'))
        # Optional Agent Setup override (from the Orchestrator's Admin > Agent Setup tab).
        # Falls back to this file's own EXPLAINER_SYSTEM_PROMPT default when not provided.
        explainer_system_prompt = data.get('explainer_system_prompt') or EXPLAINER_SYSTEM_PROMPT
        
        logger.info(f"🚀 Starting explanation process for strategy: '{strategy_used}'")
        logger.info(f"🗣️ User Query: '{user_query}'")
        
        if not items:
            logger.warning("No items provided in formal results. Explaining via logs instead.")

        def _item_uri(item):
            if isinstance(item, str):
                return item
            if isinstance(item, dict):
                return item.get('uri') or item.get('s') or item.get('id')
            return None

        def _item_name(item):
            if isinstance(item, str):
                return item
            if isinstance(item, dict):
                return item.get('item') or item.get('name') or ''
            return ''

        # HARDENING (2026-07-14): previously ONLY the first 3 items were ever shown
        # to the LLM at all (both for KG enrichment AND in the prompt itself) -- so
        # a question about an item further down the list (e.g. item #7) got answered
        # as if that item weren't recommended at all, when it just wasn't in the
        # first-3 window. Two fixes:
        # 1. Always give the LLM the full list of recommended item NAMES (cheap, no
        #    extra KG calls), so it never claims a real item is absent.
        # 2. If the user's question appears to name a specific item beyond the first
        #    3, deep-enrich THAT item's real KG context too, not just the top 3.
        full_item_names = [n for n in (_item_name(it) for it in items) if n]

        items_to_enrich = list(items[:3])
        query_lower = (user_query or "").lower()
        if query_lower:
            for item in items[3:]:
                name = _item_name(item).lower()
                if not name:
                    continue
                name_words = [w for w in name.split() if len(w) >= 5]
                word_hits = sum(1 for w in name_words if w in query_lower)
                if (len(name) >= 6 and name in query_lower) or (name_words and word_hits >= max(2, len(name_words) // 2)):
                    items_to_enrich.append(item)

        logger.info(f"🌐 Fetching graph context for {len(items_to_enrich)} item(s) via ProductReviews Agent...")
        enriched_items = []
        for item in items_to_enrich:
            item_uri = _item_uri(item)
            
            if item_uri:
                context = fetch_item_context(item_uri, fuseki_base, product_reviews_dataset)
                enriched_items.append({
                    "summary_from_recommender": item,
                    "graph_knowledge": context
                })
            else:
                enriched_items.append({"summary_from_recommender": item})

        strategy_desc = fetch_function_context(strategy_used, fuseki_base, algorithms_dataset)

        # WP4 resolution order: the execution graph, then the per-execution log
        # file, then the legacy per-algorithm log file, then the existing
        # "not found" string (which the prompt already handles by stating that
        # no trace was available). Each tier degrades to the one below it, and
        # the last tier is exactly today's behaviour.
        execution_trace_logs = fetch_execution_trace(
            execution_id, fuseki_base, executions_dataset)
        if not execution_trace_logs:
            execution_trace_logs = fetch_execution_log(strategy_used, execution_id)

        logger.info("📖 Fetching explanation types from XAI_InteractiveExplanations...")
        db_explanation_types = fetch_explanation_types(fuseki_base, explanations_dataset)
        
        types_prompt_block = ""
        if not db_explanation_types:
            logger.warning("Falling back to basic XAI set (Fuseki missing or empty).")
            types_prompt_block = "1. 'Trace-Based': Detail the execution steps.\n2. 'Rationale': Justify using specific mathematical scores."
        else:
            for i, et in enumerate(db_explanation_types):
                types_prompt_block += f"{i+1}. \"{et['name']}\": {et['desc']}\n   -> EXAMPLE QUESTIONS: {et['questions']}\n   -> ACTION: {et['action']}\n\n"
            
            types_prompt_block += f"{len(db_explanation_types)+1}. \"General\": The user is just making a statement (e.g., 'Thanks!', 'Cool').\n   -> ACTION: Acknowledge conversationally.\n"

        system_instruction = f"""
        {explainer_system_prompt}
        
        You have access to:
        1. User's Profile
        2. Recommendation Strategy used
        3. Detailed attributes of the recommended items from the Knowledge Graph
        4. The User's Conversational Query
        5. Execution Trace Logs
        
        CRITICAL INSTRUCTION:
        You MUST classify the user's query into one of these EXACT EO Explanation Types:

        {types_prompt_block}

        RULES FOR EXPLANATION TEXT:
        1. DO NOT write a single dense paragraph. You must provide a structured breakdown to maximize transparency, in a form that fits the SELECTED EO Explanation Type's own nature and ACTION guidance given above -- do not force the same template onto every type.
        2. Use MARKDOWN formatting extensively (e.g., bolding, bullet points), whichever structure you use.
        3. "General" queries (e.g. "Thanks!") may skip straight to a short conversational reply. For every other query type, tailor your structure to the type you classified:
           - **Mechanistic/technical types** (e.g. Trace-Based, Rationale, Statistical, Scientific, or any type whose own description/action above calls for citing scores, steps, or data): use ALL FOUR of these sections, and do not silently drop one:
             - **🔍 Data Context:** (What specific items from the user's history or item properties were considered)
             - **⚙️ Algorithm Execution:** (Step-by-step of how the '{strategy_used}' strategy processed the data. DO NOT just summarize; if specific features or categories are analyzed, list them.)
             - **🧮 Mathematical Rationale:** (Extract and explicitly state the exact scores, weights, or similarities found in the Execution Trace Logs. CRITICAL: Go one step deeper! If the logs mention 'top candidates', 'latent factors', or 'primary attributes', you MUST explicitly list their actual names/values. Do not just say "20 factors" or "top 20 candidates"—list the top candidates and their specific mathematical scores directly from the logs. IF the Execution Trace Logs provided to you are empty, missing, or say "not found" -- you MUST still include this section header, and explicitly state in it that no execution trace was available for this strategy run, rather than omitting the section entirely.)
             - **🎯 Conclusion:** (A brief summary of why it was recommended)
           - **Non-mechanistic types** (e.g. Everyday, Case-Based, Contextual, Fairness, Impact, Responsibility, or any type whose own description/action above calls for a relatable, non-technical account): do NOT include scores, weights, vector similarities, or other mathematical/technical detail, and do NOT include a "🧮 Mathematical Rationale" section at all (not even to say it's empty) -- explain the recommendation the way that type's own description/action calls for instead (e.g. a real-world analogy, a plain-language reason, a comparable everyday situation), while still grounding it in the actual recommended items and profile data so it never reads as generic filler. A short "**🔍 Data Context**" and "**🎯 Conclusion**" are still appropriate; "⚙️ Algorithm Execution" and "🧮 Mathematical Rationale" are not, for these types.
        4. NEVER fabricate numbers or data. Only use concrete values and explicit entity names present in the provided logs. If the log only has mathematical vectors without human-readable names, explicitly state the highest correlated vectors. (This still applies whenever you do cite numbers -- it does not override rule 3's guidance on when numbers belong at all.)
        5. Use the algorithm name exactly as given in '{strategy_used}' -- do not rename
           it, merge it with a related technique's name, or describe it as an alias/
           implementation of a differently-named algorithm (e.g. do not call an
           algorithm named "X" "Y (via X)", or vice versa) unless the Theoretical
           Workflow text you were given explicitly says so. If it genuinely is an
           implementation detail of a broader technique, you may mention that once,
           clearly attributed to the Theoretical Workflow text -- do not invent the
           relationship yourself.
        6. You are explaining an ALREADY-COMPUTED recommendation -- you cannot trigger a
           new algorithm run. If the user asks whether a different algorithm was, or
           could be, used instead, make clear that your answer describes the algorithm
           that already ran ('{strategy_used}'), and that trying a different algorithm
           means going back and generating a new recommendation, not something you can
           do from this chat. Never phrase your answer as if a fresh attempt with a
           different algorithm was just performed in response to their question.
        7. Before claiming an item was NOT recommended, check the "Full List of
           Recommended Items" below -- it contains every item's name, not just the
           ones with deep Knowledge Graph detail. If the item the user is asking
           about appears there, it WAS recommended, even if you don't have full
           graph_knowledge detail for it -- explain it using whatever context you do
           have (its name, its position in the list, the algorithm's general
           approach) rather than incorrectly telling the user it wasn't recommended.
        
        OUTPUT FORMAT (Strict JSON):
        {{
            "selected_style": "Exact name of EO style",
            "reason_for_style": "Reasoning on why query matches this style.",
            "explanation_text": "The structured Markdown explanation broken into the distinct steps outlined above."
        }}
        """
        
        user_prompt = f"""
        **User Profile:**
        {json.dumps(user_profile)}

        **Theoretical Workflow:**
        {strategy_used}
        {strategy_desc}

        **Full List of Recommended Items (names only, in ranked order -- this is
        EVERY item that was actually recommended, even ones without deep Knowledge
        Graph detail below):**
        {json.dumps(full_item_names)}

        **Recommended Items (deep Knowledge Graph detail for the top items AND any
        item the user's query specifically named):**
        {json.dumps(enriched_items)}
        
        **Execution Trace Logs:**
        {execution_trace_logs}

        **User's Query:**
        "{user_query}"
        """

        logger.info("🧠 Calling Gemini to generate explanation...")
        model = genai.GenerativeModel('gemini-3-flash-preview', system_instruction=system_instruction)
        response = model.generate_content(
            user_prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        explanation_json = json.loads(response.text)
        
        if isinstance(explanation_json, list):
            explanation_json = explanation_json[0] if len(explanation_json) > 0 else {}
        
        logger.info(f"✨ Complete! Selected Style: {explanation_json.get('selected_style')}")
        
        return jsonify(explanation_json)

    except Exception as e:
        logger.exception("Generation Failed!")
        return jsonify({"error": str(e), "explanation_text": "I couldn't generate a specific response for that right now."}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Explainer Agent on port 5001...")
    app.run(debug=True, use_reloader=False, port=5001)