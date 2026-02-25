import logging
import os
from typing import Any, Dict, List, Optional, Union
from openai import OpenAI
import json
from rdflib import XSD
import re
from dateutil import parser as dateparser
from rdflib import Graph, RDF, RDFS, OWL, Namespace
logger = logging.getLogger(__name__)


class AttributeMapper:

    def __init__(self, attribute: str, input_data, gpt_model: str = "gpt-5", api_key: str = None , candidate_amount = 3, first_split_amount = 0):
        logger.info(f"Initializing AttributeMapper with attribute: {attribute}")
        self.name = attribute
        self.gpt_model = gpt_model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY or pass api_key to AttributeMapper.")
        self.client = OpenAI(api_key=api_key)

        self.input_data = input_data


        self.state: Dict[str, Any] = {
            "attribute": self.name,
            "candidates": [],
            "validated_candidates": [],
            "matrix": [],
            "final_mapping": None,
        }

        self.logs: Dict[str, Any] = {
            "generate_mappings": {},
            "validate_mappings": {},
            "documentation_reasoning": {},
            "historical_references_reasoning": {},
            "select_final_mappings": {},
        }

        self.candidate_amount = candidate_amount

        if first_split_amount:
            self.first_valid_amount = first_split_amount
        else:
            self.first_valid_amount = candidate_amount


    def load_state(self, data: Dict[str, Any]) -> None:

        logger.debug(f"Loading attribute {self.name} state")
        self.state = data.get("state", {}) or {}
        self.logs = data.get("logs", {}) or {}

    def export_state(self) -> Dict[str, Any]:

        logger.debug(f"Exporting attribute {self.name} state")
        return {
            "state": self.state,
            "logs": self.logs,
        }

    def get_name(self):
        return self.name


    def generate_mappings(self, candidate_amount_to_generate=0, reason=None) -> None:

        if candidate_amount_to_generate:
            candidate_amount = candidate_amount_to_generate
        else:
            candidate_amount = self.candidate_amount

        logger.info(f"Generating mappings for attribute {self.name}")
        debug_info: Dict[str, Any] = {
            "input": {
                "attribute": self.name,
                "has_documentation": bool(self.input_data["documentation"]),
                "has_historical_references": bool(self.input_data["historical_references"]),
            }
        }

        grouping = self.input_data.get("grouping_context")

        grouping_text = "null"
        if isinstance(grouping, dict) and grouping.get("groups"):
            lines = []
            for g in grouping.get("groups", []):
                gid = g.get("group_id")
                label = g.get("label")
                attrs = g.get("attributes", [])
                lines.append(f"- {gid} ({label}): {', '.join(attrs)}")
            grouping_text = "\n".join(lines)

        if not reason:
            user_prompt = f"""
        You are a mapping expert.
        Your task is to generate exactly {candidate_amount} ontology-based mapping candidates 

        You will be given:
        - Original JSON data
        - Documentation (optional)
        - Historical references (optional)
        - Ontology (TTL)

        Your job:
        For the specific attribute (the attribute is the key path in the original json data), generate a json array with {candidate_amount} mapping candidates.
        A mapping candidate MUST be a JSON object with these fields:

        {{
          "object": "<a class from the ontology>",
          "relation": "<a relation from the ontology with owl datatype property>",
          "reason": "<short reason for the class / relation you chose >"
        }}

        IMPORTANT RULES:
        - DO NOT include the literal object of the TTL triple.
        - DO NOT output the TTL triple itself.
        - DO NOT escape anything manually.
        - DO NOT wrap your response in backticks, markdown, or quotes.
        - DO NOT output extra text.
        - Your ENTIRE reply MUST be a valid JSON array of exactly {candidate_amount} mapping candidates objects.

        The literal for the triple 
        will be added later programmatically by the system.

        ============================
        CONTEXT DATA
        ============================
        Grouping context (blackboard):
        {grouping_text}

        Instruction:
        If the current attribute is in a group, prefer mapping candidates that are semantically consistent with the group’s shared concept.
        Avoid subject/predicate choices that diverge from the group unless strong evidence contradicts it.
        
        Original JSON data:
        {json.dumps(self.input_data["json_data"], indent=4)}

        Documentation:
        {self.input_data["documentation"]}

        Historical references:
        {json.dumps(self.input_data["historical_references"], indent=4)}

        Ontology:
        {self.input_data["ontology"]}

        Attribute to map:
        {self.name}

        Now output ONLY the JSON array of mapping candidates:
        """

        else:
            user_prompt = f"""
               You are a mapping expert.
               Your task is to generate exactly {candidate_amount} ontology-based mapping candidates 

               You will be given:
               - Original JSON data
               - Documentation (optional)
               - Historical references (optional)
               - Ontology (TTL)

               Your job:
               For the specific attribute (the attribute is the key path in the original json data), generate a json array with {candidate_amount} mapping candidates.
               A mapping candidate MUST be a JSON object with these fields:

               {{
                 "object": "<a class from the ontology>",
                 "relation": "<a relation from the ontology with owl datatype property>",
                 "reason": "<short reason for the class / relation you chose >"
               }}

               IMPORTANT RULES:
               - DO NOT include the literal object of the TTL triple.
               - DO NOT output the TTL triple itself.
               - DO NOT escape anything manually.
               - DO NOT wrap your response in backticks, markdown, or quotes.
               - DO NOT output extra text.
               - Your ENTIRE reply MUST be a valid JSON array of exactly {candidate_amount} mapping candidates objects.

               The literal for the triple 
               will be added later programmatically by the system.

               ============================
               CONTEXT DATA
               ============================
               Grouping context (blackboard):
               {grouping_text}

               Instruction:
               If the current attribute is in a group, prefer mapping candidates that are semantically consistent with the group’s shared concept.
               Avoid subject/predicate choices that diverge from the group unless strong evidence contradicts it.


               Original JSON data:
               {json.dumps(self.input_data["json_data"], indent=4)}

               Documentation:
               {self.input_data["documentation"]}

               Historical references:
               {json.dumps(self.input_data["historical_references"], indent=4)}

               Ontology:
               {self.input_data["ontology"]}

               Attribute to map:
               {self.name}

               Your Mapping Reason:
               {reason}
               Now output ONLY the JSON array of mapping candidates:
               """


        raw_llm = self._call_llm_as_json(
            user_prompt=user_prompt,
        )

        debug_info["raw_llm_response"] = raw_llm
        debug_info["error"] = []

        candidates: List[Dict[str, Any]] = []

        if isinstance(raw_llm, list):
            for idx, item in enumerate(raw_llm):
                if not isinstance(item, dict):
                    logger.debug(f"LLM candidate #{idx} is not a dict: {item!r}")
                    continue


                class_name = item.get("object")
                prop_name = item.get("relation")
                reason = item.get("reason")

                if not class_name or not prop_name:
                    logger.debug(f"Invalid mapping candidate #{idx}: {item!r}")
                    continue


                triple = f'{class_name} {prop_name} \"{self.name}\".'

                candidates.append(
                    {
                        "candidate": triple,
                        "reason": reason,
                    }
                )

        else:
            logger.error(f"LLM did not return a list for mapping generation: {raw_llm}")
            debug_info["error"].append(f"LLM did not return a list for mapping generation: {raw_llm}")


        self.state["candidates"] = candidates
        self.logs["generate_mappings"] = debug_info

    def validate_mappings(self) -> None:


        logger.info(f"Validating mappings for attribute {self.name}")

        candidates = self.state.get("candidates", []) or []
        json_data = self.input_data.get("json_data")
        ontology_text = self.input_data.get("ontology", "")

        try:
            classes , properties  = self._parse_ontology(ontology_text)
        except Exception as e:
            logger.error(f"Ontology parsing failed: {e}")
            return



        debug_info: Dict[str, Any] = {"per_candidate": []}
        validated: List[Dict[str, Any]] = []
        matrix_rows: List[Dict[str, Any]] = []

        for cand in candidates:
            candidate_str = cand.get("candidate")
            reason = cand.get("reason")

            cand_log = {
                "candidate": candidate_str,
                "reason": reason,
                "rdf_syntax_ok": False,
                "subject_in_ontology": False,
                "predicate_in_ontology": False,
                "predicate_is_datatype_property": False,
                "range_check": None,
                "accepted_by_validator": False,
            }

            if not candidate_str:
                debug_info["per_candidate"].append(cand_log)
                continue

            try:
                subj, pred, obj = self._split_triple(candidate_str)
            except ValueError:

                debug_info["per_candidate"].append(cand_log)
                continue

            cand_log["rdf_syntax_ok"] = True

            subj_s = subj.split(":")[-1]
            pred_s = pred.split(":")[-1]


            if subj_s in classes:
                cand_log["subject_in_ontology"] = True


            prop_info = properties.get(pred_s)

            if prop_info:
                cand_log["predicate_in_ontology"] = True

                # ===== DatatypeProperty =====
                if prop_info.get("type") == "datatype":
                    cand_log["predicate_is_datatype_property"] = True
                    range_iri = prop_info.get("range")


                    try:
                        _, _, obj_literal = self._split_triple(candidate_str)
                    except ValueError:
                        obj_literal = self.name

                    values = self._extract_values_for_attribute(json_data, obj_literal)

                    ok, detail = self.is_reasonable_for_range(values, range_iri)

                    cand_log["range_check"] = {
                        "range_iri": range_iri,
                        "ok": ok,
                        "details": detail,
                    }


                else:
                    cand_log["predicate_is_datatype_property"] = False
                    cand_log["range_check"] = {
                        "range_iri": prop_info.get("range"),
                        "ok": False,
                        "details": "Predicate is owl:ObjectProperty – literals not allowed",
                    }

            else:
                cand_log["predicate_in_ontology"] = False


            range_ok = cand_log.get("range_check", {}).get("ok", False)

            accepted = (
                    cand_log["rdf_syntax_ok"]
                    and cand_log["subject_in_ontology"]
                    and cand_log["predicate_in_ontology"]
                    and cand_log["predicate_is_datatype_property"]
                    and range_ok
            )

            cand_log["accepted_by_validator"] = accepted

            if accepted:
                validated.append(
                    {
                        "candidate": candidate_str,
                        "reason": reason,
                        "validator_meta": cand_log,
                    }
                )

            matrix_rows.append(
                {
                    "candidate": candidate_str,
                    "agents": {
                        "validator": {
                            "accepted": accepted,
                            "reason": cand_log,
                        }
                    },
                }
            )

            debug_info["per_candidate"].append(cand_log)

        self.state["validated_candidates"] = validated
        self._merge_matrix_rows(matrix_rows)
        self.logs["validate_mappings"] = debug_info

    def documentation_reasoning(self, ) -> None:

        logger.info(f"Documentation reasoning for attribute {self.name}")
        documentation = self.input_data["documentation"]
        validated: List[Dict[str, Any]] = self.state.get("validated_candidates", []) or []
        validated_short = self._build_short_candidates()
        debug_info: Dict[str, Any] = {
            "has_documentation": bool(documentation),
            "validated_candidates_count": len(validated),
            "raw_llm_response": None,
        }

        if not documentation or not validated:

            self.logs["documentation_reasoning"] = debug_info
            return

        user_prompt = f"""
You are a mapping expert.
You will be given an json array of json objects, which each contains a ttl triple as the proposed mapping, and other meta data.
Your task is decide if you accepted the a candidate or not , and you give a reason.
A candidate has a ttl triple which as a possible mapping for the attribute (json key PATH, the "." is a divider between levels in a nested structure).
You base this on the given documentation, to see if the proposed candidate is "reasonable" (if data exist, the documentation may be incomplete).
Your respond only in the format of an array.
The array contains json objects with the key "accepted" which will be a boolean if you accepted the corresponding mapping,
and the key "reason" which will be a textual reason for approval or disapproval, in the same array position as the candidate.
You dont use wrappers. No other response, only that. It must be json parsable.

Here is the documentation:
{self.input_data["documentation"]}

Here is the ontology:
{self.input_data["ontology"]}

Your attribute:
{self.name}

The current state of the matrix / array:
{json.dumps(validated_short, indent=4)}
"""

        raw_llm = self._call_llm_as_json(
            user_prompt=user_prompt,
        )
        debug_info["raw_llm_response"] = raw_llm or []

        votes: List[Dict[str, Any]] = []
        if isinstance(raw_llm, list):
            for item in raw_llm:
                if not isinstance(item, dict):
                    continue
                votes.append(
                    {
                        "accepted": bool(item.get("accepted")),
                        "reason": item.get("reason"),
                    }
                )


        for cand, vote in zip(validated, votes):
            cand.setdefault("documentation_vote", vote)

        self._update_matrix_with_agent("documentation", validated, votes)
        self.logs["documentation_reasoning"] = debug_info

    def historical_references_reasoning(
            self,
    ) -> None:

        logger.info(f"Historical references reasoning for attribute {self.name}")
        historical_references = self.input_data["historical_references"]
        validated: List[Dict[str, Any]] = self.state.get("validated_candidates", []) or []
        validated_short = self._build_short_candidates()
        debug_info: Dict[str, Any] = {
            "has_historical_references": bool(historical_references),
            "validated_candidates_count": len(validated),
            "raw_llm_response": None,
        }

        if not historical_references or not validated:
            self.logs["historical_references_reasoning"] = debug_info
            return

        user_prompt = f"""
You are a mapping expert.
You will be given an json array of json objects, which each contains a ttl triple as the proposed mapping on ontology level, and other meta data.
Your task is decide if you accepted the a candidate or not , and you give a reason.
The used historical references are only a subset.
A candidate has a ttl triple which as a possible mapping for the attribute (json key PATH, the "." is a divider between levels in a nested structure).
You base this on the given historical mappings, to see if the proposed candidate is "reasonable", for example through precedents or historical "style" of mapping (if data exist, you may encounter many attributes which are not in the historical mapping).
Your respond only in the format of an array.
The array contains json objects with the key "accepted" which will we a boolean if you accepted the corresponding mapping,
and "reason" which will be a textual reason for approval or disapproval, in the same array position as the candidate.
You dont use wrappers. No other response, only that. It must be json parsable.

Here are the historical data:
{json.dumps(self.input_data["historical_references"], indent=4)}

Your attribute:
{self.name}

The current state of the matrix / array:
{json.dumps(validated_short, indent=4)}
"""

        raw_llm = self._call_llm_as_json(
            user_prompt=user_prompt,
        )
        debug_info["raw_llm_response"] = raw_llm or []

        votes: List[Dict[str, Any]] = []
        if isinstance(raw_llm, list):
            for item in raw_llm:
                if not isinstance(item, dict):
                    continue
                votes.append(
                    {
                        "accepted": bool(item.get("accepted")),
                        "reason": item.get("reason"),
                    }
                )

        for cand, vote in zip(validated, votes):
            cand.setdefault("historical_vote", vote)

        self._update_matrix_with_agent("historical", validated, votes)
        self.logs["historical_references_reasoning"] = debug_info

    def attribute_label_proximity_reasoning(
            self,
    ) -> None:

        logger.info(f"Attribute label proximity reasoning for attribute {self.name}")
        json_data = self.input_data["json_data"]
        validated: List[Dict[str, Any]] = self.state.get("validated_candidates", []) or []
        validated_short = self._build_short_candidates()
        debug_info: Dict[str, Any] = {
            "raw_llm_response": None,
        }

        if not json_data or not validated:
            self.logs["attribute_label_proximity_reasoning"] = debug_info
            return

        user_prompt = f"""
You are a mapping expert.
You will be given an json array of json objects, which each contains a ttl triple as the proposed mapping on ontology level, and other meta data.
Your task is decide if you accepted the a candidate or not , and you give a reason.
A candidate has a ttl triple which as a possible mapping for the attribute (json key PATH, the "." is a divider between levels in a nested structure).
Your base decision on how closely the mapping (the Subject and Predicate of the TTL triple of the candidate) resembles the original Attribute in the json input data, primarily in naming 
(I.e. how closely (in proximity) the Mapping and the Attribute resemble each other in name (or naming function. but the more the actually mapping resembles the Attribute name precise, the more better )).
To be more precise: You evaluate of closely the Combination of Subject and Predicate resemble the Attribute in Name, not in function.
Your respond only in the format of an array.
The array contains json objects with the key "accepted" which will we a boolean if you accepted the corresponding mapping,
a "reason" which will be a textual reason for approval or disapproval, in the same array position as the candidate, 
and a "score", with 3 being the highest, and 0 being the lowest for non acceptance. 
You can give any candidate you deem acceptable a score greater then 1. The higher the score, the better the candidate in context of what you decided.
The scores are basically the confidence ranking if you accept multiple candidates: The one you are most confident receives a score of 3 , the second a score 2 and the last a score of 1.
You can also mention in in your reason, why you gave this this candidate this score (and why you placed the score higher / below another accepted candidate)
You dont use wrappers. No other response, only that. It must be json parsable.

Your attribute:
{self.name}

The current state of the matrix / array:
{json.dumps(validated_short, indent=4)}
"""

        raw_llm = self._call_llm_as_json(
            user_prompt=user_prompt,
        )
        debug_info["raw_llm_response"] = raw_llm or []

        votes: List[Dict[str, Any]] = []
        if isinstance(raw_llm, list):
            for item in raw_llm:
                if not isinstance(item, dict):
                    continue
                votes.append(
                    {
                        "accepted": bool(item.get("accepted")),
                        "reason": item.get("reason"),
                        "score" : int(item.get("score")),
                    }
                )

        for cand, vote in zip(validated, votes):
            cand.setdefault("attribute_name_mapping_proximity_vote", vote)

        self._update_matrix_with_agent("attribute_name_matching_mapping_proximity", validated, votes)
        self.logs["attribute_name_mapping_proximity_reasoning"] = debug_info

    def example_value_reasoning(
            self,
    ) -> None:

        logger.info(f"example_value_reasoning reasoning for attribute {self.name}")
        json_data = self.input_data["json_data"]
        documentation = self.input_data["documentation"]
        validated: List[Dict[str, Any]] = self.state.get("validated_candidates", []) or []
        validated_short = self._build_short_candidates()
        debug_info: Dict[str, Any] = {
            "validated_candidates_count": len(validated),
            "raw_llm_response": None,
        }

        if not json_data or not validated:
            self.logs["example_value_reasoning"] = debug_info
            return

        user_prompt = f"""
You are a mapping expert.
You will be given an json array of json objects, which each contains a ttl triple as the proposed mapping on ontology level, and other meta data.
Your task is decide if you accepted the a candidate or not , and you give a reason.
A candidate has a ttl triple which as a possible mapping for the attribute (json key PATH, the "." is a divider between levels in a nested structure).
You base this decision by inspecting explicit the values for your Attribute in the original json and check, if they make semantic sense / are reasonable for the mapping.
From the validator meta you can infer which range the predicate had.
The Validator was really "generous" in what it accepted: For example values like "y" and "n" where excepted as booleans (are other binary occurring strings), because you can reasonably parse them into the correct xsd type without any important lost of information.
NOTE: At the end of the TTL Triple is always a literal with the Attribute name: This is only a structure placeholder and how no concern for you.
Your respond only in the format of an array.
The array contains json objects with the key "accepted" which will we a boolean if you accepted the corresponding mapping,
and "reason" which will be a textual reason for approval or disapproval, in the same array position as the candidate.
You dont use wrappers. No other response, only that. It must be json parsable.

Here is the original json data:
{json.dumps(json_data, indent=4)}

Here is the documentation:
{documentation}

Your attribute:
{self.name}

The current state of the matrix / array:
{json.dumps(validated_short, indent=4)}
"""

        raw_llm = self._call_llm_as_json(
            user_prompt=user_prompt,
        )
        debug_info["raw_llm_response"] = raw_llm or []

        votes: List[Dict[str, Any]] = []
        if isinstance(raw_llm, list):
            for item in raw_llm:
                if not isinstance(item, dict):
                    continue
                votes.append(
                    {
                        "accepted": bool(item.get("accepted")),
                        "reason": item.get("reason"),
                    }
                )

        for cand, vote in zip(validated, votes):
            cand.setdefault("example_value_vote", vote)

        self._update_matrix_with_agent("example_value", validated, votes)
        self.logs["example_value_reasoning"] = debug_info

    def select_final_mappings(self) -> None:

        logger.info(f"Selecting final mappings for attribute {self.name}")

        validated: List[Dict[str, Any]] = self.state.get("validated_candidates", []) or []
        validated_short = self._build_short_candidates()

        debug_info: Dict[str, Any] = {
            "validated_candidates_count": len(validated_short),
            "raw_llm_response": None,
            "warnings": [],
        }

        if not validated:
            self.state["final_mapping"] = None
            self.logs["select_final_mappings"] = debug_info
            return

        user_prompt = f"""
    You are a mapping expert.
    You will be given a JSON array (“matrix”) of objects, where each entry represents a mapping candidate.
    Each candidate contains:
    - a TTL triple as proposed mapping for the attribute (JSON key PATH, '.' is a divider between levels),
    - and meta data from other agents (validator, documentation, historical).
    - The Attribute name in it of itself should hold no semantic value for mapping. 
        Withing the TTL Triple of a candidate, the Subject and Predicate constitute the mapping, the literal with Attribute name is purely as identifier for processing.
        But the Attribute name MAY can give you clues, on how the Mapping should be / which candidate.
        
    Your task:
    - You must evaluate ALL candidates.
    - For EACH candidate, decide if you accept it as the final mapping or not.
    - You MUST accept exactly ONE candidate (and reject all others).
    - Try to use the reasoning's of the others agents for the candidates in your mind, their decisions and why. This is one of your stronger tools.
    
    Context Data:
    - Historically the hits@1 for your validated candidates is 70% , and your hits@3 is 81% (hits@k meaning: within the first k candidates is the correct candidate).
    This means, the first candidate is with a percentage of 70% correct ON AVERAGE, but to be the best, you have to find the cases where you need to switch of one to the other 2, or want to stay on the first.
    In conclusion: be very biased towards the first candidate, but not to much ,as not soft lock yourself of a maximum score of 70%
    
    -The "attribute_name_mapping_proximity_vote" evaluates, how "exact" (the combination of Subject and Predicate) , match the Attribute in name, not in function.
    
    - The used historical references are only a subset.
    
    Output format:
    - You MUST respond with a JSON array.
    - The array MUST have the same length and order as the list of candidates.
    - Each element MUST be a JSON object with:
      - "accepted": boolean (true if this is the final chosen mapping, false otherwise)
      - "reason": short textual reason why you accept or reject this candidate.

    No extra text, no wrappers, no comments. Only the JSON array.

    Here is the original JSON data:
    {json.dumps(self.input_data["json_data"], indent=4)}

    Here is the documentation:
    {self.input_data["documentation"]}

    Here are the historical data:
    {json.dumps(self.input_data["historical_references"], indent=4)}

    Your attribute:
    {self.name}

    Current evaluation matrix (candidates on top):
    {json.dumps(validated_short, indent=4)}
    """

        raw_llm = self._call_llm_as_json(
            user_prompt=user_prompt,
        )
        debug_info["raw_llm_response"] = raw_llm

        # ---- Votes einlesen ----
        votes: List[Dict[str, Any]] = []
        if isinstance(raw_llm, list):
            for item in raw_llm:
                if not isinstance(item, dict):
                    continue
                votes.append(
                    {
                        "accepted": bool(item.get("accepted")),
                        "reason": item.get("reason"),
                    }
                )
        else:
            debug_info["warnings"].append(
                f"LLM did not return a list for final selection: {raw_llm!r}"
            )


        for cand, vote in zip(validated, votes):
            cand.setdefault("selection_vote", vote)


        self._update_matrix_with_agent("selection", validated, votes)


        final_index = None
        for idx, vote in enumerate(votes):
            if vote.get("accepted"):
                if final_index is None:
                    final_index = idx
                else:

                    debug_info["warnings"].append(
                        f"Multiple candidates marked as accepted. Using first at index {final_index}, ignoring {idx}."
                    )
                    break

        selected = None
        if isinstance(final_index, int) and 0 <= final_index < len(validated):
            selected = {
                "candidate": validated[final_index]["candidate"],
                "score" : 1,
                "meta": validated[final_index],
                "selection_reason": votes[final_index].get("reason") if final_index < len(votes) else None,
                "index": final_index,
            }
        else:
            debug_info["warnings"].append(
                "No candidate was marked as accepted by the selection agent."
            )

        self.state["final_mapping"] = selected
        self.logs["select_final_mappings"] = debug_info


    def _build_literal_for_attribute(self) -> str:

        escaped = self.name.replace('"', '\\"')
        return f'"{escaped}"'

    def _call_llm_as_json(
            self,
            system_prompt: str = None,
            user_prompt: str = None,
            messages_to_send_in=None,
            expected_description: str = "",
    ) -> Any:


        if messages_to_send_in:
            messages = messages_to_send_in
        else:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})

        # API call
        response = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=messages
        )

        content = response.choices[0].message.content

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"LLM returned non-JSON for {expected_description}: {content}")
            return None

        return parsed

    def _split_triple(self, triple: str) -> (str, str, str):

        triple = triple.strip()


        if triple.endswith('.'):
            triple = triple[:-1].strip()

        pattern = r'^(?P<subject>[^\s]+)\s+(?P<predicate>[^\s]+)\s+"(?P<object>.*)"$'
        m = re.match(pattern, triple)

        if not m:
            raise ValueError(f"Invalid triple format: {triple}")

        return m.group("subject"), m.group("predicate"), m.group("object")

    def _parse_ontology(self,ontology_text: str):


        g = Graph()
        g.parse(data=ontology_text, format="turtle")

        classes = set()
        properties: Dict[str, Dict[str, Any]] = {}


        def short(u):
            s = str(u)
            if "#" in s:
                return s.split("#")[-1]
            return s.split("/")[-1]


        for s in g.subjects(RDF.type, OWL.Class):
            classes.add(short(s))


        for s in g.subjects(RDF.type, OWL.DatatypeProperty):
            key = short(s)
            props = properties.setdefault(key, {})
            props["type"] = "datatype"
            rng = next(g.objects(s, RDFS.range), None)
            if rng is not None:
                # Range auf xsd:short bringen
                if str(rng).startswith(str(XSD)):
                    local = short(rng)
                    props["range"] = f"xsd:{local}"
                else:
                    props["range"] = short(rng)


        for s in g.subjects(RDF.type, OWL.ObjectProperty):
            key = short(s)
            props = properties.setdefault(key, {})

            if props.get("type") != "datatype":
                props["type"] = "object"
            rng = next(g.objects(s, RDFS.range), None)
            if rng is not None:
                if str(rng).startswith(str(XSD)):
                    local = short(rng)
                    props["range"] = f"xsd:{local}"
                else:
                    props["range"] = short(rng)

        return classes, properties

    def _extract_values_for_attribute(
            self,
            json_data: Union[List, Dict],
            attribute_path: str,
    ) -> List[Any]:


        def extract(obj, path_parts):
            if not path_parts:
                return [obj]
            key = path_parts[0]
            rest = path_parts[1:]

            if isinstance(obj, list):
                vals = []
                for item in obj:
                    vals.extend(extract(item, path_parts))
                return vals

            if isinstance(obj, dict):
                if key in obj:
                    return extract(obj[key], rest)
                else:
                    return []
            return []

        parts = attribute_path.split(".") if attribute_path else []
        values = extract(json_data, parts)


        flat: List[Any] = []

        def flatten(v):
            if isinstance(v, list):
                for x in v:
                    flatten(x)
            else:
                flat.append(v)

        for v in values:
            flatten(v)
        return flat

    def is_placeholder(self,v: Any) -> bool:
        PLACEHOLDER_VALUES = {
            "", " ", "n/a", "na", "null", "none", "-", "--", "unknown", "unk"
        }
        s = str(v).strip().lower()
        return s in PLACEHOLDER_VALUES

    def is_reasonable_for_range(self, values: List[Any], range_iri: Optional[str]):
        if not range_iri:
            return True, "No range → permissive accept"

        tmp = range_iri
        if "#" in tmp:
            ri = tmp.split("#")[-1].lower().strip()
        else:
            ri = tmp.split(":")[-1].lower().strip()

        # collect non-null non-placeholder
        non_null = [v for v in values if v is not None]
        non_placeholder = [v for v in non_null if not self.is_placeholder(v)]

        if not non_placeholder:
            return True, "Only placeholder/empty values → cannot contradict range"

        def safe(s):
            return str(s).strip()


        def try_parse_date(x):
            try:
                dateparser.parse(safe(x), dayfirst=False)
                return True
            except Exception:
                return False

        def try_parse_time(x):

            s = safe(x)


            try:
                dt = dateparser.parse(s)
                return True
            except:
                pass


            if re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", s):
                return True
            if re.match(r"^\d{3,4}$", s):
                return True
            if re.match(r"(?i)^\s*(am|pm)?\s*\d{1,2}:\d{2}(:\d{2})?\s*(am|pm)?\s*$", s):
                return True

            return False

        def try_parse_int_like(x):
            s = safe(x)
            s = s.replace(",", ".")  # normalize
            try:
                f = float(s)
                return abs(f - round(f)) < 1e-6
            except:
                return False

        def is_multi_int_like(x):


            def is_clean_int(p):
                if not re.match(r"^\d+$", p):
                    return False
                # reject leading zeros EXCEPT the single-digit "0"
                return (p == "0") or (not p.startswith("0"))

            s = str(x).strip().lower()


            delimiters = [",", ";", "/", "&"]

            for d in delimiters:
                if d in s:
                    parts = [p.strip() for p in s.split(d) if p.strip()]
                    return (
                            len(parts) >= 2
                            and all(is_clean_int(p) for p in parts)
                    )

            if "," in s:
                parts = [p.strip() for p in s.split(",") if p.strip()]
                if len(parts) == 2:
                    return all(re.match(r"^-?\d+$", p) for p in parts)
            if "-" in s:
                parts = [p.strip() for p in s.split("-") if p.strip()]
                if len(parts) == 2:
                    return all(re.match(r"^-?\d+$", p) for p in parts)

            # '15 to 55'
            if " to " in s:
                parts = [p.strip() for p in s.split(" to ") if p.strip()]
                if len(parts) == 2:
                    return all(re.match(r"^-?\d+$", p) for p in parts)

            return False

        def try_parse_uri(x):

            s = safe(x)
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:[^\s]+$", s):
                return True
            return True

        def try_parse_year(x):
            s = safe(x)

            if re.match(r"^\d{1,4}$", s):
                return True
            try:
                dt = dateparser.parse(s)
                return True
            except:
                return False

        def try_parse_month(x):
            s = safe(x)
            if s.isdigit():
                return True
            try:
                dt = dateparser.parse("2020-" + s + "-01")
                return True
            except:
                return False

        def try_parse_day(x):
            s = safe(x)
            if s.isdigit():
                return True
            try:
                dt = dateparser.parse(f"2020-01-{s}")
                return True
            except:
                return False

        # ------------------------------------------------------------
        # L E N I E N T   RANGE MATCHING
        # ------------------------------------------------------------

        if ri == "string":
            return True, "Everything is allowed for xsd:string"

        if ri == "int":


            if all(try_parse_int_like(v) for v in non_placeholder):
                return True, "Integer validation applied"


            if any(is_multi_int_like(v) for v in non_placeholder):
                others = [v for v in non_placeholder if not is_multi_int_like(v)]
                if all(try_parse_int_like(v) for v in others):
                    return True, "Multi-int tolerant validation applied"

            return False, "Integer validation failed"

        if ri in ("decimal", "float", "double"):
            try:
                for v in non_placeholder:
                    float(safe(v).replace(",", "."))
                return True, "Lenient float/decimal"
            except:
                return False, "Float parse failed"

        if ri == "date":
            ok = all(try_parse_date(v) for v in non_placeholder)
            return ok, "Lenient date format"

        if ri == "datetime":
            ok = all(try_parse_date(v) or try_parse_time(v) for v in non_placeholder)
            return ok, "Lenient datetime"

        if ri == "datetimestamp":
            ok = all(try_parse_date(v) for v in non_placeholder)
            return ok, "Lenient dateTimeStamp"

        if ri == "time":
            ok = all(try_parse_time(v) for v in non_placeholder)
            return ok, "Lenient time"

        if ri == "gyear":
            ok = all(try_parse_year(v) for v in non_placeholder)
            return ok, "Lenient gYear"

        if ri == "gmonth":
            ok = all(try_parse_month(v) for v in non_placeholder)
            return ok, "Lenient gMonth"

        if ri == "gday":
            ok = all(try_parse_day(v) for v in non_placeholder)
            return ok, "Lenient gDay"

        if ri == "anyuri":
            ok = all(try_parse_uri(v) for v in non_placeholder)
            return ok, "Lenient URI"

        if ri == "anysimpletype":
            return True, "Fully permissive anySimpleType"


        if range_iri.lower().startswith("xsd:"):
            return True, f"Generic XSD datatype accepted ({range_iri})"

        return True, f"Unknown datatype {range_iri} → accepted leniently"

    def _merge_matrix_rows(self, new_rows: List[Dict[str, Any]]) -> None:

        matrix: List[Dict[str, Any]] = self.state.get("matrix", []) or []
        by_candidate = {row["candidate"]: row for row in matrix if "candidate" in row}

        for row in new_rows:
            cand = row.get("candidate")
            if not cand:
                continue

            if cand in by_candidate:
                existing_agents = by_candidate[cand].setdefault("agents", {})
                new_agents = row.get("agents", {})
                existing_agents.update(new_agents)
            else:
                matrix.append(row)
                by_candidate[cand] = row

        self.state["matrix"] = matrix

    def _update_matrix_with_agent(
            self,
            agent_name: str,
            validated_candidates: List[Dict[str, Any]],
            votes: List[Dict[str, Any]],
    ) -> None:

        matrix: List[Dict[str, Any]] = self.state.get("matrix", []) or []
        by_candidate = {row["candidate"]: row for row in matrix if "candidate" in row}

        for cand, vote in zip(validated_candidates, votes):
            c_str = cand.get("candidate")
            if not c_str:
                continue
            row = by_candidate.get(c_str)
            if not row:
                row = {
                    "candidate": c_str,
                    "agents": {},
                }
                matrix.append(row)
                by_candidate[c_str] = row

            agents = row.setdefault("agents", {})
            agents[agent_name] = vote

        self.state["matrix"] = matrix

    def _build_short_candidates(self, first_amount = 0) -> dict:

        if first_amount:
            split = first_amount

        else:
            split = self.first_valid_amount

        vc = self.state.get("validated_candidates") or []
        return vc[:split]
