# Prompt Authoring Guide For Pair-Wise Parameter Analysis

This guide explains how to write prompts for each pair:
- criterion (parameter)
- prompting technique

It also explains how architecture description and architecture diagram are passed to the model.

## 1) How Inputs Are Injected Automatically

You do not need to manually paste project data every time.
The pipeline already injects project context before your pair prompt:
- architecture description text
- strategy name and directive
- architecture diagram as image context (if available)

Code path:
- Base context prompt construction: src/parameter_pipeline.py
- Pair prompt execution: src/parameter_pipeline.py

## 2) Placeholders You Can Use In Prompt Cells

You can use these tokens in any prompt cell. They are replaced automatically at runtime.

- {parameter}: current criterion text
- {criterion}: same as {parameter}
- {criteria}: same as {parameter}
- {technique}: current technique name
- {prompt_technique}: current technique name
- {project_title}: current project title
- {architecture_description}: full architecture description text
- {architecture_text}: same as {architecture_description}
- {has_diagram}: yes or no
- {diagram_available}: yes or no
- {diagram_note}: sentence saying whether diagram is attached

Tip:
Use {has_diagram} or {diagram_note} to define fallback behavior when image is missing.

## 3) Multi-Step Prompts In One Cell

If you want step-by-step execution in one cell, separate steps using one of:
- ||
- line with ---
- line with ===
- Step 1:, Step 2:, ...
- Prompt 1:, Prompt 2:, ...
- JSON array string

Example:
Step 1: Extract diagram components for {parameter} || Step 2: Extract text components for {parameter} || Step 3: Compare and output JSON

## 4) Recommended Prompt Structure (Any Technique)

Use this structure for best quality:
1. Evidence extraction instruction
2. Comparison or reasoning rule
3. Decision rule (Detected vs Not Detected)
4. Strict output rule

Example final line in each prompt:
Output ONLY strict JSON for this parameter with value Detected or Not Detected.

## 5) Technique-Specific Writing Patterns

### Monolithic
Use one direct instruction with explicit decision criteria.

Template:
Check {parameter} using architecture text and diagram evidence. If explicit matching evidence exists, Detected; otherwise Not Detected. Output strict JSON only.

### CoT
Use 3 to 4 steps in one cell.

Template:
Step 1: Extract component evidence from diagram for {parameter}. || Step 2: Extract component evidence from architecture text for {parameter}. || Step 3: Compare both evidence sets and decide Detected/Not Detected for {parameter}. || Step 4: Output strict JSON only.

### CoVe
Force self-verification after a draft decision.

Template:
Step 1: Draft decision for {parameter}. || Step 2: Ask 3 verification questions that could flip the label. || Step 3: Re-check evidence and output corrected strict JSON.

### RCoT
Start from required evidence, then compare with actual evidence.

Template:
Step 1: Define mandatory evidence required to detect {parameter}. || Step 2: Check if that evidence exists in text and diagram. || Step 3: Output strict JSON.

### Expert
Use strict architecture-review language and ambiguity penalties.

Template:
As an expert architect, evaluate whether {parameter} is explicitly represented, technically consistent, and unambiguous across text and diagram. If ambiguous or contradictory, mark Not Detected. Output strict JSON only.

### Two-Model
Best as a single prompt cell if you want full generator-judge-refine behavior.

Template:
Generate an initial label for {parameter} from architecture evidence, then critique the label for false positives/negatives, then output final corrected strict JSON.

Note:
If you provide multiple steps in one cell for two_model, current runtime uses sequential chat flow.

## 6) Output Discipline

Keep prompt endings strict to reduce parse errors:
- Use only one key: current parameter
- Use only values: Detected or Not Detected
- No markdown
- No extra explanation

Good output example:
{"Is the architecture diagram complete and accurate?": "Detected"}

## 7) Common Mistakes To Avoid

- Mixing multiple parameters in one pair prompt
- Asking for long explanations in final output
- Leaving decision criteria vague
- Not defining fallback when diagram is missing
- Using synonyms like Present/Absent without normalization intent

## 8) Practical Workflow

1. Fill long-format or wide-format matrix.
2. Start with monolithic + cot only.
3. Validate outputs on 3 to 5 sample projects.
4. Tune decision rules for false positives.
5. Add cove/rcot/expert/two_model after baseline stabilizes.
