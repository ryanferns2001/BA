import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class GroupingAgent:
    """
    Grouping Agent for a blackboard-based semantic typing system.

    - run_provisional(): name-based, conservative, pre-candidate grouping
    - run_final(): evidence-based, semantic meaning grouping, post-candidate/discussion refinement

    Writes/updates: blackboard["grouping"]
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5",
        max_iterations: int = 3,
        timeout_seconds: int = 120,
    ) -> None:
        if not api_key:
            raise ValueError("GroupingAgent requires an OpenAI API key.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds

    # -------------------------
    # Public API
    # -------------------------

    def run_provisional(self, blackboard: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provisional grouping: based primarily on attribute names/paths.
        Intended to run before candidate generation.
        """
        attributes = self._extract_attributes(blackboard)
        if not attributes:
            logger.warning("GroupingAgent.run_provisional: no attributes found; skipping.")
            return {}

        prompt = self._build_provisional_prompt(attributes)
        llm_json = self._call_llm_json(prompt)

        grouping = self._normalize_grouping_output(
            llm_json=llm_json,
            status="provisional",
            stage="pre_candidates",
            iteration=self._next_iteration(blackboard),
            attributes=attributes,
            evidence_summary={
                "name_similarity": True,
                "value_patterns": False,
                "documentation": False,
                "candidate_overlap": False,
                "historical_refs": False,
            },
        )

        self._write_grouping(blackboard, grouping)
        return grouping

    def run_final(self, blackboard: Dict[str, Any]) -> Dict[str, Any]:
        """
        Final grouping: semantic meaning-based refinement using richer evidence.
        Intended to run after candidate generation (and optionally during discussion).
        """
        attributes = self._extract_attributes(blackboard)
        if not attributes:
            logger.warning("GroupingAgent.run_final: no attributes found; skipping.")
            return {}

        provisional = self._safe_get(blackboard, ["grouping"])
        candidate_summary = self._extract_candidate_summary(blackboard)
        value_summaries = self._extract_value_summaries(blackboard)
        documentation = self._extract_documentation(blackboard)

        prompt = self._build_final_prompt(
            attributes=attributes,
            provisional_grouping=provisional,
            candidate_summary=candidate_summary,
            value_summaries=value_summaries,
            documentation=documentation,
        )
        llm_json = self._call_llm_json(prompt)

        grouping = self._normalize_grouping_output(
            llm_json=llm_json,
            status="final",
            stage="post_candidates",
            iteration=self._next_iteration(blackboard),
            attributes=attributes,
            evidence_summary={
                "name_similarity": True,
                "value_patterns": bool(value_summaries),
                "documentation": bool(documentation),
                "candidate_overlap": bool(candidate_summary),
                "historical_refs": False,  # add later if you store/access it
            },
        )

        self._write_grouping(blackboard, grouping)
        return grouping

    # -------------------------
    # Prompt builders (LOCKED)
    # -------------------------

    def _build_provisional_prompt(self, attributes: List[str]) -> str:
        system = (
            "You are a Grouping Agent operating within a blackboard-based multi-agent system.\n"
            "Your task is to identify a provisional grouping of attributes that appear\n"
            "semantically related based on their attribute names or paths.\n\n"
            "This grouping is used only as contextual information for downstream agents.\n"
            "You must not assign semantic types, ontology classes, or relations."
        )

        user = (
            "You are given a list of attributes from a structured dataset.\n\n"
            "Your task:\n"
            "- Group attributes that appear semantically related based on their names.\n"
            "- Be conservative: only group attributes when name-based similarity is clear.\n"
            "- Do not force all attributes into groups.\n"
            "- Attributes that do not clearly belong to any group must remain ungrouped.\n\n"
            "IMPORTANT:\n"
            "- This is a PROVISIONAL grouping.\n"
            "- Do NOT use external knowledge.\n"
            "- Do NOT infer meaning from values or candidates.\n"
            '- Do NOT group solely based on shared suffixes such as "_id", "_code", "_flag".\n'
            "- Do NOT output semantic types, ontology terms, or mappings.\n\n"
            "Output format:\n"
            "Return a valid JSON object with the following structure and nothing else:\n\n"
            '{\n'
            '  "status": "provisional",\n'
            '  "groups": [\n'
            "    {\n"
            '      "group_id": "G1",\n'
            '      "label": "<short descriptive label>",\n'
            '      "attributes": ["attr1", "attr2", "..."],\n'
            '      "rationale": "<brief explanation based on name similarity>"\n'
            "    }\n"
            "  ],\n"
            '  "ungrouped_attributes": ["attrX", "attrY"],\n'
            '  "ungrouped_rationales": {\n'
            '    "attrX": "<brief explanation why it was not grouped>",\n'
            '    "attrY": "<brief explanation why it was not grouped>"\n'
            '  },\n'  
            '  "quality_flags": {\n'
            '    "confidence": "<low|medium|high>"\n'
            "  }\n"
            "}\n\n"
            "Attributes:\n"
            f"{json.dumps(attributes, indent=2)}"
        )

        # We pass system+user as a single user message because your codebase uses simple chat.completions.
        return f"SYSTEM:\n{system}\n\nUSER:\n{user}"

    def _build_final_prompt(
        self,
        attributes: List[str],
        provisional_grouping: Optional[Dict[str, Any]],
        candidate_summary: Optional[Dict[str, Any]],
        value_summaries: Optional[Dict[str, Any]],
        documentation: Optional[str],
    ) -> str:
        system = (
            "You are a Grouping Agent operating within a blackboard-based multi-agent system.\n"
            "Your task is to produce a final grouping of attributes based on semantic meaning.\n"
            "This grouping will be used as contextual information by other agents.\n\n"
            "You must not assign semantic types, ontology classes, or relations."
        )

        user = (
            "You are given:\n"
            "- A list of attributes\n"
            "- A provisional grouping (if available)\n"
            "- Additional semantic evidence\n\n"
            "Your task:\n"
            "- Refine the grouping based on semantic meaning.\n"
            "- Attributes may be grouped even if their names differ, if they represent the same concept.\n"
            "- You may merge or split provisional groups if semantic evidence supports it.\n"
            "- Leave attributes ungrouped if evidence is insufficient.\n\n"
            "Semantic evidence may include:\n"
            "- Similar value patterns\n"
            "- Overlapping candidate mappings\n"
            "- Documentation descriptions\n"
            "- Historical references\n\n"
            "IMPORTANT:\n"
            "- This is a FINAL grouping.\n"
            "- Do NOT create groups without clear semantic justification.\n"
            "- Do NOT force all attributes into groups.\n"
            "- Do NOT output ontology classes, relations, or triples.\n\n"
            "Output format:\n"
            "Return a valid JSON object with the following structure and nothing else:\n\n"
            '{\n'
            '  "status": "final",\n'
            '  "groups": [\n'
            "    {\n"
            '      "group_id": "G1",\n'
            '      "label": "<short descriptive label>",\n'
            '      "attributes": ["attr1", "attr2", "..."],\n'
            '      "rationale": "<brief explanation citing semantic evidence>"\n'
            "    }\n"
            "  ],\n"
            '  "ungrouped_attributes": ["attrX", "attrY"],\n'
            '  "ungrouped_rationales": {\n'
            '    "attrX": "<brief explanation why it was not grouped>",\n'
            '    "attrY": "<brief explanation why it was not grouped>"\n'
            '  },\n'  
            '  "quality_flags": {\n'
            '    "confidence": "<low|medium|high>",\n'
            '    "coverage": <number between 0 and 1>\n'
            "  }\n"
            "}\n\n"
            "Attributes:\n"
            f"{json.dumps(attributes, indent=2)}\n\n"
            "Provisional grouping:\n"
            f"{json.dumps(provisional_grouping, indent=2) if provisional_grouping else 'null'}\n\n"
            "Semantic evidence:\n"
            "- Candidate mappings:\n"
            f"{json.dumps(candidate_summary, indent=2) if candidate_summary else 'null'}\n"
            "- Attribute value summaries:\n"
            f"{json.dumps(value_summaries, indent=2) if value_summaries else 'null'}\n"
            "- Documentation:\n"
            f"{documentation if documentation else 'null'}\n"
        )

        return f"SYSTEM:\n{system}\n\nUSER:\n{user}"

    # -------------------------
    # LLM call + JSON parsing
    # -------------------------

    def _call_llm_json(self, prompt: str) -> Dict[str, Any]:
        """
        Calls the model and returns parsed JSON.
        Includes a robust fallback that extracts the first JSON object found.
        """
        logger.info("GroupingAgent: calling LLM (%s)", self.model)

        # NOTE: Keep parameters minimal to avoid model-specific incompatibilities.
        # Some models ignore temperature; omitting it is safe.
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.timeout_seconds,
        )

        content = response.choices[0].message.content or ""
        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract the first JSON object from the response.
            extracted = self._extract_first_json_object(content)
            if extracted is None:
                logger.error("GroupingAgent: failed to parse JSON. Raw output:\n%s", content)
                raise
            return extracted

    def _extract_first_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extracts the first {...} JSON object found in a text blob.
        """
        # Simple heuristic: find first '{' and last '}' and attempt parse.
        # Also handles cases where the model added extra text accidentally.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start : end + 1].strip()

        # Remove trailing commas before closing braces/brackets (common LLM error)
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    # -------------------------
    # Blackboard helpers
    # -------------------------

    def _write_grouping(self, blackboard: Dict[str, Any], grouping: Dict[str, Any]) -> None:
        blackboard["grouping"] = grouping
        logger.info(
            "GroupingAgent: wrote grouping (%s/%s) with %d groups",
            grouping.get("status"),
            grouping.get("stage"),
            len(grouping.get("groups", [])),
        )

    def _next_iteration(self, blackboard: Dict[str, Any]) -> int:
        existing = blackboard.get("grouping")
        if not isinstance(existing, dict):
            return 1
        prev = existing.get("iteration")
        if isinstance(prev, int):
            return prev + 1
        return 1

    def _extract_attributes(self, blackboard: Dict[str, Any]) -> List[str]:
        """
        Tries multiple common locations for attribute lists, to minimize coupling.
        Adjust if your blackboard stores attributes under a specific key.
        """
        # Common possibilities (you can refine based on your actual blackboard structure):
        candidates = [
            self._safe_get(blackboard, ["attributes"]),
            self._safe_get(blackboard, ["state", "attributes"]),
            self._safe_get(blackboard, ["json_data", "attributes"]),
        ]

        for c in candidates:
            if isinstance(c, list) and all(isinstance(x, str) for x in c):
                return c

        # If your VC-SLAM JSON is stored on the blackboard, you may need to derive attribute paths.
        # For now, we keep this minimal and require explicit attribute list injection.
        return []

    def _extract_candidate_summary(self, blackboard: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Returns a compact summary of candidates if present.
        Expected shape depends on your pipeline; we keep this resilient.
        """
        # Example locations you might use; adapt once we inspect your pipeline outputs:
        for path in (["candidates"], ["state", "candidates"], ["mapping_candidates"]):
            c = self._safe_get(blackboard, path)
            if isinstance(c, dict):
                return c
        return None

    def _extract_value_summaries(self, blackboard: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for path in (["value_summaries"], ["state", "value_summaries"], ["attribute_value_summaries"]):
            v = self._safe_get(blackboard, path)
            if isinstance(v, dict):
                return v
        return None

    def _extract_documentation(self, blackboard: Dict[str, Any]) -> Optional[str]:
        for path in (["documentation"], ["state", "documentation"]):
            d = self._safe_get(blackboard, path)
            if isinstance(d, str) and d.strip():
                return d.strip()
        return None

    def _safe_get(self, obj: Dict[str, Any], path: List[str]) -> Any:
        cur: Any = obj
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        return cur

    # -------------------------
    # Output normalization
    # -------------------------

    def _normalize_grouping_output(
        self,
        llm_json: Dict[str, Any],
        status: str,
        stage: str,
        iteration: int,
        attributes: List[str],
        evidence_summary: Dict[str, bool],
    ) -> Dict[str, Any]:
        """
        Ensures the grouping output is structurally usable and contains required metadata.
        """
        groups = llm_json.get("groups", [])
        ungrouped = llm_json.get("ungrouped_attributes", [])
        ungrouped_rationales = llm_json.get("ungrouped_rationales", {})
        qf = llm_json.get("quality_flags", {})

        if not isinstance(groups, list):
            groups = []
        if not isinstance(ungrouped, list):
            ungrouped = []
        if not isinstance(ungrouped_rationales, dict):
            ungrouped_rationales = {}    

        # Ensure each group has required fields
        normalized_groups = []
        gid_counter = 1
        seen_attrs = set()

        for g in groups:
            if not isinstance(g, dict):
                continue
            g_attrs = g.get("attributes", [])
            if not isinstance(g_attrs, list) or not all(isinstance(x, str) for x in g_attrs):
                continue

            # Remove attributes not in the provided attribute list
            g_attrs = [a for a in g_attrs if a in attributes]

            # Keep groups with at least 2 attributes (conservative)
            if len(g_attrs) < 2:
                continue

            label = g.get("label")
            rationale = g.get("rationale")
            if not isinstance(label, str) or not label.strip():
                label = f"group_{gid_counter}"
            if not isinstance(rationale, str) or not rationale.strip():
                rationale = "Grouped based on available evidence."

            group_id = g.get("group_id")
            if not isinstance(group_id, str) or not group_id.strip():
                group_id = f"G{gid_counter}"

            normalized_groups.append(
                {
                    "group_id": group_id,
                    "label": label.strip(),
                    "attributes": g_attrs,
                    "rationale": rationale.strip(),
                }
            )
            gid_counter += 1
            seen_attrs.update(g_attrs)

        # Compute ungrouped attributes conservatively
        remaining = [a for a in attributes if a not in seen_attrs]
        # Merge model-provided ungrouped (filtered to valid attributes)
        model_ungrouped = [a for a in ungrouped if isinstance(a, str) and a in attributes]
        ungrouped_final = sorted(set(remaining).union(model_ungrouped), key=lambda x: attributes.index(x))

        # Normalize ungrouped rationales (one short reason per ungrouped attribute)
        DEFAULT_UNGROUPED_REASON = "No sufficiently strong evidence to assign this attribute to a group."
        normalized_ungrouped_rationales = {}

        for a in ungrouped_final:
            r = ungrouped_rationales.get(a)
            if isinstance(r, str) and r.strip():
                normalized_ungrouped_rationales[a] = r.strip()
            else:
                normalized_ungrouped_rationales[a] = DEFAULT_UNGROUPED_REASON

        # Quality flags (diagnostic only)
        coverage = 0.0
        if attributes:
            coverage = 1.0 - (len(ungrouped_final) / float(len(attributes)))

        confidence = "medium"
        if isinstance(qf, dict) and isinstance(qf.get("confidence"), str):
            confidence = qf["confidence"].strip().lower()
            if confidence not in ("low", "medium", "high"):
                confidence = "medium"

        normalized_qf: Dict[str, Any] = {
            "confidence": confidence,
            "coverage": round(coverage, 4),
            "has_overlaps": False,  # we enforce non-overlap by construction
            "has_singletons": False,  # we dropped <2-size groups above
        }

        now = datetime.now(timezone.utc).isoformat()

        return {
            "version": "1.0",
            "status": status,
            "stage": stage,
            "iteration": iteration,
            "max_iterations": self.max_iterations,
            "created_at": now,
            "source_agent": "Grouping Agent",
            "evidence_summary": evidence_summary,
            "groups": normalized_groups,
            "ungrouped_attributes": ungrouped_final,
            "ungrouped_rationales": normalized_ungrouped_rationales,
            "quality_flags": normalized_qf,
            "debug": {
                "raw_model": self.model,
                "prompt_version": "grouping-v1",
            },
        }
