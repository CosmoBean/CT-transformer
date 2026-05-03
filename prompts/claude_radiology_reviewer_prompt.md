Review the chest X-ray image together with the structured model evidence.

Allowed image-level labels:
{{ALLOWED_LABELS}}

Allowed derived global buckets:
{{ALLOWED_GLOBAL_BUCKETS}}

Return exactly one JSON object that matches the schema. Do not add markdown or extra commentary.

Rules:
- Use only the provided image and case packet evidence.
- Do not invent labels outside the allowed label set.
- Keep `supported_findings` limited to findings with meaningful evidence.
- Put borderline or conflicting findings in `uncertain_findings`.
- Use `review_recommendation` as one of: `supported`, `uncertain`, `needs_human_review`.
- Use `confidence_band` as one of: `high`, `moderate`, `low`.
- Keep the `findings_section` and `impression_section` concise and clinically cautious.
- Always include the provided safety framing in the `safety_note`.

Case packet:
{{CASE_PACKET_JSON}}
