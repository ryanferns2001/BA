import copy
import json
import re
import logging


class DiscussionEngine:

    def __init__(self, api_key, gpt_model="gpt-5"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.gpt_model = gpt_model


    def run_discussion(self, discussion: dict, mappers: dict , original_json = None, documentation = None, historical_references = None, grouping= None) -> dict:


        logging.debug(f"Starting discussion for {json.dumps(discussion, indent=4)}")

        participants_raw = discussion.get("participants", [])

        participants = []
        for p in participants_raw:
            if isinstance(p, str):
                participants.append({"attribute": p, "role": "weak"})
            else:
                participants.append({
                    "attribute": p.get("attribute"),
                    "role": p.get("role", "weak")
                })

        max_turns = discussion.get("max_turns", 4)
        turn_logs = []


        end_flags = {p["attribute"]: False for p in participants}
        changed_any = False


        short_candidates = self._build_short_candidates(mappers)

        for turn in range(1, max_turns + 1):
            turn_entry = {"turn": turn, "log": []}


            attr_context = {}
            for p in participants:
                attr_name = p["attribute"]
                mapper = mappers.get(attr_name)
                if not mapper:
                    continue
                attr_context[attr_name] = {
                    "role": p["role"],
                    "final_mapping": mapper.state.get("final_mapping"),
                    "validated_candidates": short_candidates.get(attr_name, [])
                }

            for p in participants:
                attr_name = p["attribute"]
                role = p["role"]

                if end_flags.get(attr_name, False):

                    continue

                mapper = mappers.get(attr_name)
                if not mapper:
                    continue

                prompt = self._build_prompt_discussion(
                    current_attr=attr_name,
                    role=role,
                    discussion=discussion,
                    attr_context=attr_context,
                    current_turn=turn,
                    original_json_data=original_json,
                    documentation=documentation,
                    historical_mappings=historical_references,
                    grouping=grouping
                )

                response = self._call_llm_as_json(prompt)
                if not isinstance(response, dict):

                    logging.warning(f"DiscussionEngine: non-dict response for {attr_name}: {response}")
                    continue

                turn_entry["log"].append(response)


                if turn >= 2:
                    did_change = self._apply_commands(
                        response=response,
                        mappers=mappers,
                        short_candidates=short_candidates,
                        end_flags=end_flags,
                        call_from="discussion",
                    )
                    if did_change:
                        changed_any = True

            turn_logs.append(turn_entry)

            if all(end_flags.values()):
                break

        if changed_any:
            conclusion = "Correction"
        elif all(end_flags.values()):
            conclusion = "Acceptance"
        else:
            conclusion = "No agreement"

        discussion["turn_logs"] = turn_logs

        discussion["discussion_log"] = []
        discussion["conclusion"] = conclusion

        logging.debug(f"Discussion finished with conclusion={conclusion}")
        return discussion


    def _build_short_candidates(self, mappers: dict) -> dict:

        short = {}
        for attr, mapper in mappers.items():
            vc = mapper.state.get("validated_candidates") or []
            short[attr] = vc[:3]
        return short

    def _call_llm_as_json(self, prompt):

        response = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logging.error(f"DiscussionEngine LLM returned non-JSON: {content}")
            return {}

    def _apply_commands(self, response, mappers: dict, short_candidates: dict,
                        end_flags: dict , call_from = "default") -> bool:

        attr = response.get("attribute")
        if not attr or attr not in mappers:
            return False

        mapper = mappers[attr]
        cmds = response.get("commands", []) or []
        params = response.get("command_parameters", []) or []
        changed = False

        for cmd, param in zip(cmds, params):
            if cmd == "Change:Candidate":
                try:
                    idx = int(param)
                except (ValueError, TypeError):
                    continue

                candidates = short_candidates.get(attr, [])
                if idx < 0 or idx >= len(candidates):
                    continue

                cand_obj = candidates[idx]
                fm = copy.deepcopy( mapper.state.get("final_mapping") or {})
                before = copy.deepcopy(fm)
                mapper.state[f"final_mapping_before_{call_from}"] = before
                fm["candidate"] = cand_obj.get("candidate")
                fm["meta"] = cand_obj
                fm["index"] = idx


                sel_vote = cand_obj.get("selection_vote", {})
                try:
                    fm["selection_reason"] = f"Changed to this candidate after {call_from}"
                    fm["meta"]["selection_vote"]["reason"] = f"Changed to this candidate after {call_from}"
                    fm["meta"]["selection_vote"]["accepted"] = True
                except Exception:
                    print("Error")

                mapper.state["final_mapping"] = fm
                mapper.logs.setdefault(f"{call_from}", []).append(
                    {"action": "change_candidate", "index": idx, "new_candidate": fm["candidate"]}
                )
                changed = True

            elif cmd == "DiscussionState:End":
                end_flags[attr] = True

        return changed

    def _build_prompt_discussion(self, current_attr, role, discussion, attr_context, current_turn, documentation ="", original_json_data = None, historical_mappings = None, grouping= None):

        turn_logs_so_far = discussion.get("turn_logs", [])
        discussion_static = {
            k: v
            for k, v in discussion.items()
            if k != "turn_logs"
        }
        return f"""
You are participating in a semantic mapping  discussion.

ROLE BEHAVIOR:
- As a WEAK role, you are more flexible in changing your mapping to another validated candidate if arguments convince you. Which doesn't mean you can't have strong opinion yourself, if you can back it up / convince the others.
- As a STRONG role, you are more conservative: you mostly provide feedback and only change your mapping if there is a very strong reason.

- YOU DON'T HAVE TO "RESOLVE" THE REASON, but you should really keep it in mind. You don't have to be convinced of the reason for this discussion or even disagree, provided you are confident enough in your opinions.
- Based on historical data: The probability of the correct mapping being the first one is 70%, and that the correct mapping in the array is in the top 3 is 80% (so choose wisely)
    
TURN RULES:
- In TURN 1 you MUST NOT execute any commands. You only explain what you see, how your mapping relates to others, and what you might consider.
- From TURN 2 onwards you MAY:
  - keep your mapping,
  - or switch to another candidate from your top-3 validated candidates,
  - or declare you are done with this discussion.
  - You may use multiple commands (Important: The order of the commands in the commands array, and the parameters for the corresponding command in the parameters array must be identical)

ALLOWED COMMANDS (from TURN 2 onwards):
- "Change:Candidate" with a parameter that is the INDEX (0-based) of the candidate in your validated_candidates list.
- "DiscussionState:End" with parameter "" (empty string) to indicate you are finished with this discussion.
You can also provide no commands at all if you only want to talk.

-Since this a discussion you can voice your opinion. Even more: you see all available candidates from the other attributes in the discussion, and COULD even prompt to things like "if you pick this, I pick that.
    A discussion is a dialog over multiple turns.
- If you dont have anything to add, you can just say for the response part "no comment" or similar, and starting turn 2 use the commend to end the discussion.
In the array, the TTL-Triple of candidate, the Subject and Predicate are the mapping, while the literal with the attribute name is purely as identifier for processing, but MAY give clues for the appropriate candidate.

Here is the full context for the previous generation and discussion upon which the candidates are build and selected:

Grouping context (blackboard):
{json.dumps(grouping, indent=2) if grouping else "null"}

Instruction:
If the participants belong to the same group in the grouping context above, aim for semantically compatible subject/predicate choices across the group.
Only argue for cross-group changes if you explicitly justify why.

Original Json-Data:
{json.dumps(original_json_data, indent=2)}

Documentation:
{documentation}.

Historical Mappings (subset, not all)::
{json.dumps(historical_mappings, indent=2)}

Ontology not provided because of the size. (But you may assume every candidate you see is valid. But as the ontology is pretty open, some candidates MAY be to flat mapped).

DATA YOU SEE (for ALL participants):
{json.dumps(attr_context, indent=4)}

DISCUSSION Static block:
{json.dumps(discussion_static, indent=4)}

Respond EXACTLY in JSON format (no wrappers like ```json):

{{
  "attribute": "{current_attr}",
  "response": "<your textual reasoning for this turn>",
  "commands": ["..."],
  "command_parameters": ["..."]
}}

If you are in TURN 1, leave "commands" as an empty list.

Discussion reason:
{discussion.get("reason", "")}

Your attribute: {current_attr}
Your role: {role.upper()}.
Current turn: {current_turn}

TURN LOGS SO FAR:
{json.dumps(turn_logs_so_far, indent=4)}
"""

    def split_triple(self, triple: str):
        triple = triple.strip()

        if triple.endswith('.'):
            triple = triple[:-1].strip()

        m = re.match(r'^(?P<subject>[^\s]+)\s+(?P<predicate>[^\s]+)\s+"(?P<object>.*)"$', triple)
        if not m:
            raise ValueError(f"Invalid triple format: {triple}")

        return m.group("subject"), m.group("predicate"), m.group("object")
