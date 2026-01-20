import json
import re
import requests
from typing import Any, Dict, List, Union, Optional
from ollama import Client
from rdflib import Graph, Literal, RDF, OWL
import logging

from openai import OpenAI
import json5  # tolerant parser, pip install json5
from typing import Dict, Any
import re, json, ast
from simplellm.codebase.config.logging_config import setup_root_logger
import tiktoken
# === Logging Setup ===
logger = logging.getLogger(__name__)

class LLMMapper:
    def __init__(self, name: str, llm_type: str, llm_params: Dict[str, Any]):
        self.name = name
        self.llm_type = llm_type.lower()
        self.llm_params = llm_params
        self.llm = self._init_llm()


        self.datatype_properties = set()
        self.object_properties = set()
        self.relevant_keys = set()
        self.ontology_graph = None
        self.FIXED_TEMP_MODELS= ["gpt-5-mini"  ,"gpt-5"]


    def _init_llm(self):
        if self.llm_type == "openai":
            if "api_key" not in self.llm_params or "model" not in self.llm_params:
                raise ValueError("OpenAI needs 'api_key' and 'model'")
            client = OpenAI(api_key=self.llm_params["api_key"])
            return {
                "provider": "openai",
                "client": client,
                "model": self.llm_params["model"],
                "temperature": 0 ,# self.llm_params["temperature"]
                "weight": self.llm_params["weight"],
            }

        elif self.llm_type == "ollama":

            if "model" not in self.llm_params:
                raise ValueError("Ollama needs 'model'")
            client = Client(host=self.llm_params["host"])
            return {
                "provider": "ollama",
                "model": self.llm_params["model"],
                "client": client
            }

        else:
            raise ValueError(f"Unknown LLM-Type: {self.llm_type}")


    def _load_ontology(self, ontology_str: str):

        self.ontology_graph = Graph()
        try:
            self.ontology_graph.parse(data=ontology_str, format="turtle")
        except Exception as e:
            raise ValueError(f"Error while Parsing the ontology: {e}")


        self.datatype_properties = {
            str(s)
            for s, _, o in self.ontology_graph.triples((None, RDF.type, OWL.DatatypeProperty))
        }


        self.object_properties = {
            str(s)
            for s, _, o in self.ontology_graph.triples((None, RDF.type, OWL.ObjectProperty))
        }


        self.ontology_classes = {
            str(s)
            for s, _, o in self.ontology_graph.triples((None, RDF.type, OWL.Class))
        }

    def _clean_literal(self, lit: str) -> str:

        if "^^" in lit:
            lit = lit.split("^^")[0]
        return lit.strip().strip('"').strip("'")

    def set_weight(self, new_weight: float):
        self.llm["weight"] = new_weight

    def get_weight(self) -> float:
        return self.llm["weight"]

    def validate_ttl_model(self, ttl_str: str, ontology_prefixes: str) -> dict:

        result = {
            "ttl_valid": True,
            "ontology_valid": True,
            "errors": []
        }

        ttl_str = ttl_str.strip()
        if not ttl_str:
            result["ttl_valid"] = False
            result["ontology_valid"] = False
            result["errors"].append("Leerer TTL-String.")
            return result


        if "@prefix" not in ttl_str:
            ttl_str = f"{ontology_prefixes.strip()}\n{ttl_str}"

        g = Graph()
        try:
            g.parse(data=ttl_str, format="turtle")
        except Exception as e:
            result["ttl_valid"] = False
            result["ontology_valid"] = False
            result["errors"].append(f"TTL parsing error: {e}")
            return result

        if not self.ontology_graph:
            return result


        for s, p, o in g:
            p_str = str(p)
            s_str = str(s)
            if p_str in self.datatype_properties and not isinstance(o, Literal):
                result["ontology_valid"] = False
                result["errors"].append(f"{p_str} is a DatatypeProperty, but object is not a literal: {o}")
            elif p_str in self.object_properties and isinstance(o, Literal):
                result["ontology_valid"] = False
                result["errors"].append(f"{p_str} is a ObjectProperty, but object is a literal: {o}")
            elif p_str not in self.datatype_properties and p_str not in self.object_properties:
                result["ontology_valid"] = False
                result["errors"].append(f"{p_str} is not a valid relation from the ontology")
            elif s_str not in self.ontology_classes:
                result["ontology_valid"] = False
                result["errors"].append(f"{s_str} is not a valid Class from the ontology")


        ontology_classes = getattr(self, "ontology_classes", set())
        instance_types = {str(o) for _, _, o in g.triples((None, RDF.type, None))}

        for cls in instance_types:
            if cls not in ontology_classes:
                result["ontology_valid"] = False
                result["errors"].append(f"Class {cls} not found in Ontology.")
        return result

    def count_tokens(self,text: Union[str, list, dict], model: str = "gpt-4o-mini") -> int:
        if not isinstance(text, str):
            import json
            text = json.dumps(text, ensure_ascii=False)

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

        tokens = enc.encode(text)
        return len(tokens)


    def _clean_llm_response_text(self, text: str) -> str:

        text = text.strip()

        text = re.sub(r"^```(json|JSON)?", "", text)
        text = re.sub(r"```$", "", text)

        text = text.strip().strip('"').strip("'")

        text = text.replace("{{", "{").replace("}}", "}")
        return text

    def _parse_llm_json(self, text: str) -> dict[str, Any]:

        cleaned = self._clean_llm_response_text(text)
        try:
            data = json.loads(cleaned)
            return self._postprocess_candidates(data)
        except json.JSONDecodeError as e:
            logger.warning(f"No parsing possible for llm response: {e}")

            try:
                import json5
                data = json5.loads(cleaned)
                return self._postprocess_candidates(data)
            except Exception:
                import ast
                try:
                    data = ast.literal_eval(cleaned)
                    return self._postprocess_candidates(data)
                except Exception:
                    logger.error("Parsing total failure.")
                    return {}

    def _postprocess_candidates(self, data: dict[str, Any]) -> dict[str, Any]:

        if "mappings_candidates" not in data:
            return data

        for key, candidates in data["mappings_candidates"].items():
            if not isinstance(candidates, list):
                continue


            n = len(candidates)
            for i, cand in enumerate(candidates):
                cand["score"] = n - i


            candidates.sort(key=lambda x: x["score"], reverse=True)
            data["mappings_candidates"][key] = candidates

        return data

    def _call_llm(self, prompt: str) -> dict[str, Any]:

        provider = self.llm["provider"]

        if provider == "openai":
            client = self.llm["client"]
            temp_adjustable = False
            if any(m in self.llm["model"] for m in self.FIXED_TEMP_MODELS):
                temp_adjustable = False
            else:

                temp_adjustable = True

            if temp_adjustable:
                response = client.chat.completions.create(
                model=self.llm["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0
                )
            else:
                response = client.chat.completions.create(
                    model=self.llm["model"],
                    messages=[{"role": "user", "content": prompt}]
                )

            response = client.chat.completions.create(
                model=self.llm["model"],
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.choices[0].message.content
            logger.debug(f"Response from LLM: {content}")
            content_cleaned = self._clean_llm_response_text(content)


            try:
                return json.loads(content_cleaned)
            except json.JSONDecodeError as e:
                logger.debug(f"Parsing JSON failed. {e}")
            logger.warning(f"[WARN] Unexpected LLM-Content-Type: {type(content_cleaned)} \nContent: {content_cleaned}")
            return {}

        elif provider == "ollama":
            client = self.llm["client"]
            tokens = self.count_tokens(prompt, model="cl100k_base")

            response = client.generate(model=self.llm["model"], prompt=prompt,     options={
        "num_ctx": 65536  # 64K Tokens
    })




            content = response.get("message", {}).get("content", response)


            if isinstance(content, str):

                cleaned_content = self._clean_ollama_response(content)
                return self._parse_llm_json(cleaned_content)

            elif isinstance(content, dict):

                return self._postprocess_candidates(content)

            logger.warning(f"[WARN] Unexpected Ollama-Content-Type: {type(content)}")
            return {}

        else:
            raise ValueError(f"Unknown Provider {provider}")

    def _clean_ollama_response(self, content: str) -> str:


        if content.startswith("```json") and content.endswith("```"):
            content = content[7:-3].strip()
        return content

    def _extract_leaf_paths(self, data, parent_key=""):

        leaf_paths = set()

        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{parent_key}.{key}" if parent_key else key
                if isinstance(value, (dict, list)):
                    leaf_paths.update(self._extract_leaf_paths(value, full_key))
                else:
                    leaf_paths.add(full_key)

        elif isinstance(data, list):

            for i, item in enumerate(data):
                leaf_paths.update(self._extract_leaf_paths(item, parent_key))

        else:

            leaf_paths.add(parent_key)

        return leaf_paths

    def _generate_mapping_structure(
            self,
            json_data: Union[List, Dict],
            ontology_prefixes: str,
            unmapped_attributes: Optional[Dict] = None
    ) -> Dict[str, Any]:

        prefix_lines = []
        for line in ontology_prefixes.splitlines():
            if line.strip().startswith("@prefix"):
                prefix_lines.append(line.strip())
        prefix_block = "\n".join(prefix_lines)

        unmapped_keys = set(unmapped_attributes.keys()) if unmapped_attributes else set()
        leaf_paths = self._extract_leaf_paths(json_data)
        mappings = {}
        for p in leaf_paths:

            if any(p == uk for uk in unmapped_keys):
                continue
            mappings[p] = []
        self.relevant_keys = set(mappings.keys())


        mapping_structure = {"prefix": prefix_block, "mappings_candidates": mappings}
        logger.debug(f"Mapping structure: {json.dumps(mapping_structure, indent=4)}")

        return mapping_structure

    def _generate_prompt(
        self,
        pre_outline: Dict[str, Any],
        json_data: Union[List, Dict],
        ontology: str,
        documentation: Optional[str] = "",
        historical_mappings: Optional[Dict] = None,
        candidate_mapping_amount: int = 15
    ) -> str:
        historical_mappings = historical_mappings or {}
        return f"""
You are an expert in semantic data modeling.

- Do not include explanations, comments, or metadata.
- Do not escape characters.
- Do not wrap your response in quotes.
- You map ontology level (no instances or blank nodes)

Here is the current structure:
{json.dumps(pre_outline, indent=4)}

Generate {candidate_mapping_amount} candidate mappings per key as TTL triples (fill empty the arrays with the mappings).
Each mapping: {{"candidate": "<TTL triple in the form of: Class-of-Ontology DatatypeProperty-Relation-Ontology \\\"The-key-you-map\\\">.", "score": <rank>}}.
Order by confidence (descending).
You DON'T map any actually data behind lead nodes.
THE TRIPLE HAS TO BE TTL VALID (as in , I can use it out of the box, provided I already use the prefixes , formatting like "." at the end etc.).
The Object in the triple is always the a literal with the key-path you are generating the candidate for.
Do NOT change key names or prefix section.
You will not give the array an xsd type via the "^^" notation.

YOU ONLY ANSWER with with the the same structure from "current structure" , but the arrays filled with the mappings. Nothing else
Your answer has to be json valid.

Original JSON data:
{json.dumps(json_data, indent=4)}

Documentation (if any):
{documentation}

Historical mappings:
{json.dumps(historical_mappings, indent=4)}

Ontology:
{ontology}
"""

    def safe_json_loads(self, text: str) -> Dict[str, Any]:

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fixed = re.sub(r'(?<=\s)"([a-zA-Z0-9_.-]+)"(?=[\s.,])', r'\\"\1\\"', text)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass


        try:
            return json5.loads(text)
        except Exception:
            pass


        return {}

    def _merge_llm_response(self, template_json: Dict[str, Any], llm_text: Union[str, dict], prefixes: str) -> Dict[
        str, Any]:

        # Ensure debug section exists (some templates omit it)
        template_json.setdefault("debug", {})
        template_json["debug"].setdefault("errors_messages", [])
        template_json["debug"].setdefault("mapping_candidates_debug", {})

        if isinstance(llm_text, dict):
            response_json = llm_text
        elif isinstance(llm_text, str):
            llm_text = self._clean_llm_response_text(llm_text)
            response_json = self.safe_json_loads(llm_text)
        else:
            logger.error(f"[ERROR] Unexpected Type for llm_text: {type(llm_text)}")
            template_json["debug"]["errors_messages"].append(f"Unexpected type. Expected json ready to parsed, but got {type(llm_text)}.\nResponse: {llm_text}")
            return template_json

        if not response_json:
            template_json["debug"]["errors_messages"].append("Empty or not parseable response.")
            logger.warning("[WARN] Empty or not parsable LLM-Response.")
            return template_json

        mappings = response_json.get("mappings") or response_json.get("mappings_candidates")
        if not mappings:
            template_json["debug"]["errors_messages"].append("Empty or no mappings found in response.")
            logger.warning("[WARN] No 'mappings' or 'mappings_candidates' in LLM-Output found.")
            return template_json

        for key, candidates in mappings.items():
            if key not in template_json["mappings_candidates"]:
                continue

            valid_list = []
            debug = []
            total = len(candidates)

            for idx, c in enumerate(candidates):
                if not isinstance(c, dict):
                    continue
                ttl = c.get("candidate", "")
                if not ttl:
                    continue

                score = total - idx  # 🔁 neue Logik: Länge minus Index
                check = self.validate_ttl_model(ttl, prefixes)
                debug.append({"candidate": ttl, "score": score , "debug": check})
                if check["ttl_valid"] and check["ontology_valid"]:
                    valid_list.append({"candidate": ttl, "score": score})

            template_json["mappings_candidates"][key] = valid_list
            template_json["debug"]["mapping_candidates_debug"][key] = debug

        return template_json

    def mapping(
            self,
            json_data: Union[List, Dict],
            ontology: str,
            documentation: Optional[str] = "",
            historical_mappings: Optional[Dict] = None,
            unmapped_attributes: Optional[Dict] = None,
            candidate_mapping_amount: int = 15
    ) -> Dict[str, Any]:
        logger.info(f"Mapping process started for {self.name} ({self.llm_type})")

        self._load_ontology(ontology)


        self.ontology_graph = Graph()
        try:
            self.ontology_graph.parse(data=ontology, format="turtle")
        except Exception as e:
            logger.warning(f"[WARN] Ontology couldn't be loaded: {e}")


        pre_outline = self._generate_mapping_structure(json_data, ontology, unmapped_attributes)


        prompt = self._generate_prompt(pre_outline, json_data, ontology, documentation, historical_mappings,
                                       candidate_mapping_amount)



        llm_output = self._call_llm(prompt)

        if not llm_output:
            llm_output = pre_outline

        logger.debug(f"[LLM-Output {self.name} ({self.llm_type}] {llm_output}")

        prefix_lines = "\n".join([line for line in ontology.splitlines() if line.strip().startswith("@prefix")])

        result = self._merge_llm_response(pre_outline, llm_output, prefix_lines)
        return result




    def select_best_candidate(self, key, json_data, documentation, historical, candidate_list, struct):

        if not candidate_list:
            return None

        prompt = f"""
        Your are on ontology mapping agent. 
        You will be given an array of candidates.
        Your task is: To the given key: {key} . Select exactly 1 candidate. Respond only with it and nothing more.
        The candidate is the whole json object in the array. Meaning your response is only  {{"candidate": "Class-of-Ontology DatatypeProperty-Relation-Ontology \\\"The-key-you-map\\\" ." , "score": <the rank>}}
        Don't use wrappers, comments, etc.
        
        Here is the candidate list
        {json.dumps(candidate_list, indent=2)}
        
        Here the original json data:
        {json.dumps(json_data, indent=2)}
    
        Here a documentation (if given):
        {documentation}
        
        Here is the current mapping_structure (if a candidate array has only 1 element, it means that one was chosen)
        {struct}
        Here a historical mappings (if given):
        {json.dumps(historical, indent=2)}
        """

        result = self._call_llm(prompt)
        return result

