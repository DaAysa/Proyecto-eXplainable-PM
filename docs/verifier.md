# Independent response verification

PMAx runs Engineer → Analyst → Verifier before presenting the final response. Verifier makes one logical model request through the existing provider connection, with its own system instructions and no correction loop.

Its input contains only the latest question, the structured Analyst report (with existing artifact IDs), and bounded computed-data previews. Engineer captures the artifacts created or updated by the successful execution attempt. Artifacts left by failed attempts or previous questions remain available to the legacy workflow but are excluded from Verifier evidence. When accepted provenance cannot be established, the review can evaluate coverage but cannot confirm those artifacts as support. Event logs, agent histories, generated code, error traces, and image bytes are not sent to Verifier.

The structured review separates:

- `answers_question`: whether all essential requested points were addressed.
- `requirements`: each requested point and its coverage.
- `claims`: assertions labeled `supported`, `contradicted`, or `not_verifiable`, with existing evidence IDs and brief explanations.
- `suggestions`: up to three brief suggestions, without rewriting the report.

Metric, value, unit, activities, population, period, and filters must agree with evidence. Missing evidence is not a contradiction. This is an AI review of available outputs, not an independent recomputation or a guarantee of truth. Previews are limited to 20 artifacts, 4,000 characters per preview and 24,000 preview characters overall; partial or unavailable data cannot support assertions about unshown values.

The interface preserves the Analyst report and stores its review in a separate block, under the initially collapsed **Verificación de la respuesta** panel. The panel includes coverage, claims, associated evidence previews, and suggestions. An invalid response or provider error produces **No se pudo completar la verificación**, rather than a false coverage verdict. The structured review is also saved separately in the session's `reports` directory with a manifest entry. PDF export continues to export the original analysis report.

## Checking the implementation

Run the targeted tests from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_verifier -v
```

These tests use model doubles for complete and incomplete answers, a claim contradicted by a metric, and a claim without evidence. They also cover provenance across retries and questions, input minimization, invalid output, provider failure, workflow ordering, persistence, and Streamlit rendering. They do not establish the semantic accuracy of a live model.

For a manual check, launch the app normally, configure a provider, and submit a question that produces a metric. After the Analyst and Verifier finish, open **Verificación de la respuesta** and compare the verdict, requested points, and source previews. A new question gets a fresh review; rerendering the conversation does not call Verifier again. Automatic feedback or retries to Engineer/Analyst are intentionally outside this version.
