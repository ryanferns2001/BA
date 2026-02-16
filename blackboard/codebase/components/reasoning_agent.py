import json
from openai import OpenAI
import logging

class ReasoningAgent:

    def __init__(self, api_key: str, gpt_model="gpt-5"):
        self.client = OpenAI(api_key=api_key)
        self.gpt_model = gpt_model

    def determine_discussions(self, attribute_map: dict, original_json_data=None, documentation : str = "", historical_references = None, amount_turns = 3, grouping= None) -> dict:

        if original_json_data is None:
            original_json_data = {}
        user_prompt = f"""
You are a semantic mapping consistency checker.

You will receive a JSON dict where keys are attributes and values contain:
- final selected mapping
- agent evaluation matrix

Your task:
- Identify cases where attributes should discuss for what reason you deem necessary. This can range for many reason like misalignment, wrong choosing of the final mapping (i.e. you don't generally think it fits), or global context of attributes / original data (A mapping is the Subject and Predicate, the Object / literal is only there as identifier, and should hold no semantic value). Remember, because your discussion reason for the Attributes, so that they have a discuss among them self, what good pick / choice another candidate in their validated candidate list , they may change nothing, which is fine.
-  you have the data of all the attributes and they reasoning matrix, how the got the the mapping (Maybe you initial find an "error",
    but see the historical agent found precedents, and as such is correct, and such no discussion is needed, or maybe you think multiple attributes still need to align or similar).
    For example, even if "latitude", "longitude" etc. use the Subject Coordinate_pair, because they are a pair (Similar should be other coordinate pair like (not arrays) stuff),
    "coordinates" which may be an array coordinates may is something like the subject "Coordinates" and the relation coordinate_pair_DP,
    which mock the Coordinate Pair Class as in between step, because you can't use the multiple Relations for the same attribute
     (If coordinates is an array, you can't very well use the relation latitude and longitude, this is why something like ).
- You MUST NOT start or continue discussions ONLY because:
  • different fields contain the same literal values (e.g., coordinates duplicated),
  • the data appears redundant in the original JSON,
  • attributes represent similar concepts at different nesting levels.
- Redundancy in the source data is NORMAL and EXPECTED.  
  DO NOT treat redundancy as a mapping problem.
- Two attributes , or set of attributes, may legitimately map to the same subject or predicate
  without requiring alignment, consolidation, or canonicalization.
- The used historical references are only a subset.
- Historically the hits@1 for validated candidates is 70% , and hits@3 is 81% (hits@k meaning: within the first k candidates is the correct candidate).
    This means, the first candidate is with a percentage of 70% correct ON AVERAGE, but to be the best, there need to be sometimes to be chosen something else.
- Remember that these mapping are the basis for a semantic model, you don't have to assume the final mapping to be extreme flat. 
- Also Remember, because these are individual mappings of ontology basis, things like instances are none of your concern. 
- There may exist attributes with same mapping , but different literal, because they are on different "levels" in the original json (For example , "latitude" and "location.latitude" may very well have in the mapping the same Subject and relation, but "need" respectively their longitude). 
- ONLY start a discussion WITH RELEVANT attributes, (are in a similar group).
- Your discussions reason should possible to realize based on the candidate pool of the Attributes: Don't give impossible to actualize reasons.
- The Attribute name in it of itself should hold no semantic value for mapping. 
        Withing the TTL Triple of a candidate, the Subject and Predicate constitute the mapping, the literal with Attribute name is purely as identifier for processing.
        But the Attribute name MAY can give you clues, on how the Mapping should be / which candidate.
-The "attribute_name_mapping_proximity_vote" evaluates, how "exact" (the combination of Subject and Predicate) , match the Attribute in name, not in function.        
- You have to give each attribute a role: Weak and strong. The Attributes which the weak roles are basically the ones your are not quite certain, there are correct in the global context or at.
    Attributes with the strong role in a discussion basically serve for input and feedback, and should barely, if at all, convinced to be change their mapping. 
    (You don't have to include a "strong" attribute, these are merely optional. For an attribute to be strong, you have to be really confident).

- You set the amount of max turns to {amount_turns}
OUTPUT FORMAT EXACTLY (you may start more then 1 discussion, here is just an example for 1. You may use any number greater then 1 of attributes for the discussion):

{{
  "discussion_1": {{
      "participants": [{{"attribute" : <attr1> , "role" : <weak or strong> }}, {{"attribute" : <attr2> , "role" : <weak or strong> }}],
      "reason": "<why>",
      "max_turns": {amount_turns},
      "discussion_log": [],
      "conclusion": null
  }}
}}


Do not perform discussion, only declare them. Only respond in this json format. No wrappers like '''json or similar.'

Grouping context (blackboard):
{json.dumps(grouping, indent=2) if grouping else "null"}

Rule:
Prefer discussion participants that are in the same grouping. Only propose cross-group discussions if you explicitly justify why.

Attribute Data:
{json.dumps(attribute_map, indent=4)}

Original JSON Data:
{json.dumps(original_json_data, indent=4)}

Documentation Data:
{documentation}

Historical References (subset, not all)::
{json.dumps(historical_references, indent=4)}
"""
        logging.info("Reasoning Agent started.")
        response = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=[{"role": "user", "content": user_prompt}]
        )
        response_clean = response.choices[0].message.content
        json.loads(response_clean)
        try:

            logging.debug(f"Discussion Response from Reasoning Agent:\n{json.dumps(json.loads(response_clean), indent=4)}")
        except Exception as e:
            logging.error(f"Discussion Response from Reasoning Agent:\n{response_clean}")
        return json.loads(response_clean)
