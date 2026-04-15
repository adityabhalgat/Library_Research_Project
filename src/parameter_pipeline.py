import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fpdf import FPDF
from rich.console import Console
from rich.table import Table

from src.input_handlers.csv_handler import CSVHandler
from src.input_handlers.image_handler import ImageHandler
from src.input_handlers.prompt_table_handler import PromptMatrix, PromptTableHandler, SUPPORTED_TECHNIQUES
from src.llm import OllamaClient
from src.llm.base import ChatMessage, LLMResponse
from src.output.run_logger import RunTextLogger
from src.prompts import get_prompt_strategy, ProjectInputs
from config import settings

console = Console()


class ParameterAnalysisPipeline:
    def __init__(self):
        self.csv_handler = CSVHandler()
        self.output_dir = Path(settings.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm_client = OllamaClient(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature
        )
        # Initialize judge client for two_model strategy
        self.judge_client = OllamaClient(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature
        )
        self.prompt_matrix = self._load_prompt_matrix()

    def load_parameters(self, path: str = "parameters.csv") -> List[str]:
        """Load parameters from prompt matrix if available, else from CSV."""
        if self.prompt_matrix and self.prompt_matrix.criteria:
            return list(self.prompt_matrix.criteria)

        params: List[str] = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and str(row[0]).strip():
                    params.append(str(row[0]).strip())

        if params and self._canonical_key(params[0]) in {"parameter", "parameters", "criterion", "criteria"}:
            params = params[1:]

        return params

    def load_projects(self, path: str = "projects.csv") -> List[str]:
        """Load project IDs from CSV."""
        projects: List[str] = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header "project_id"
            for row in reader:
                if row:
                    projects.append(row[0])
        return projects

    def _resolve_prompt_table_path(self) -> Path | None:
        configured_path = Path(settings.parameter_prompt_table_path)
        if configured_path.exists():
            return configured_path

        # Fallback: use the first .xlsx in workspace if an explicit path is not present.
        xlsx_candidates = sorted(Path.cwd().glob("*.xlsx"))
        if xlsx_candidates:
            return xlsx_candidates[0]

        return None

    def _load_prompt_matrix(self) -> PromptMatrix | None:
        prompt_table_path = self._resolve_prompt_table_path()
        if not prompt_table_path:
            return None

        try:
            matrix = PromptTableHandler(
                table_path=str(prompt_table_path),
                sheet_name=settings.parameter_prompt_table_sheet or None,
            ).load()
            if matrix.criteria:
                console.print(
                    f"[green]Loaded prompt matrix from {prompt_table_path} "
                    f"({len(matrix.criteria)} criteria, {len(matrix.available_techniques)} techniques).[/green]"
                )
            return matrix
        except Exception as exc:
            console.print(
                f"[yellow]Could not load prompt matrix from {prompt_table_path}: {exc}. "
                "Falling back to parameters.csv workflow.[/yellow]"
            )
            return None

    def _resolve_strategies(self) -> List[str]:
        if self.prompt_matrix and self.prompt_matrix.available_techniques:
            return list(self.prompt_matrix.available_techniques)
        return list(SUPPORTED_TECHNIQUES)

    def analyze(self):
        parameters = self.load_parameters()
        project_ids = self.load_projects()
        strategies = self._resolve_strategies()

        started_at = datetime.now()
        run_ts = started_at.strftime("%Y%m%d_%H%M%S")
        run_logger = RunTextLogger(
            run_type="parameter_analysis",
            logs_dir="ogs",
            header_context={
                "Model": settings.ollama_model,
                "Projects": len(project_ids),
                "Parameters": len(parameters),
            },
        )

        # Results table: strategy -> parameter -> count
        results = {s: {p: 0 for p in parameters} for s in strategies}
        detailed_results: List[Dict[str, Any]] = []

        console.print(f"[bold green]Starting Analysis for {len(project_ids)} projects against {len(parameters)} parameters using {len(strategies)} strategies...[/bold green]")

        for pid in project_ids:
            project_data = self.csv_handler.get_group_data(pid)
            if not project_data:
                console.print(f"[red]Project {pid} not found in dataset.[/red]")
                continue
            
            project_inputs = ProjectInputs(
                project_title=project_data.get('project_title'),
                abstract=project_data.get('abstract'),
                architecture_description=project_data.get('architecture_description')
            )

            # Load Architecture Image
            architecture_image_path = project_data.get('architecture_image_path')
            image_base64 = None
            image_media_type = "image/jpeg"

            if architecture_image_path and os.path.exists(architecture_image_path):
                try:
                    handler = ImageHandler(architecture_image_path)
                    image_base64 = handler.to_base64()
                    image_media_type = handler.media_type
                    console.print(f"[green]Loaded architecture image: {architecture_image_path}[/green]")
                except Exception as e:
                    console.print(f"[yellow]Could not load image {architecture_image_path}: {e}[/yellow]")
            else:
                if architecture_image_path:
                     console.print(f"[yellow]Image path not found: {architecture_image_path}[/yellow]")

            console.print(f"\n[bold blue]Analyzing Project {pid}: {project_inputs.project_title}[/bold blue]")
            run_logger.log_project_header(project_id=pid, project_title=project_inputs.project_title)

            for strategy_name in strategies:
                strategy = get_prompt_strategy(strategy_name)
                try:
                    analysis: Dict[str, str] = {}

                    for parameter in parameters:
                        response, executed_prompt = self._run_parameter_pair(
                            strategy=strategy,
                            strategy_name=strategy_name,
                            project_inputs=project_inputs,
                            parameter=parameter,
                            image_base64=image_base64,
                            image_media_type=image_media_type,
                        )

                        value = self._parse_single_parameter_response(response.content, parameter)
                        analysis[parameter] = value

                        run_logger.log_llm_exchange(
                            project_id=pid,
                            project_title=project_inputs.project_title,
                            strategy_name=strategy_name,
                            phase_name=f"parameter::{self._canonical_key(parameter)}",
                            prompt_text=executed_prompt,
                            response_text=response.content,
                            has_image=bool(image_base64),
                        )

                    detailed_results.append(
                        {
                            "project_id": pid,
                            "project_title": project_inputs.project_title,
                            "strategy": strategy_name,
                            "analysis": analysis,
                        }
                    )

                    # Print project-specific report
                    console.print(f"  Strategy: [cyan]{strategy_name}[/cyan]")
                    console.print(json.dumps(analysis, indent=4))

                    # Update counts
                    for param in parameters:
                        value = analysis.get(param, "Not Detected")
                        if self._is_detected(value):
                            results[strategy_name][param] += 1

                except Exception as e:
                    console.print(f"[red]Error executing strategy {strategy_name}: {e}[/red]")
                    run_logger.log_note(
                        f"Strategy {strategy_name} failed for {pid}: {e}"
                    )

        # Display final table
        self._display_table(results, parameters)
        table_text = self._build_table_text(results, parameters)

        run_logger.close(
            summary={
                "Run Started": started_at.isoformat(),
                "Run Ended": datetime.now().isoformat(),
                "Strategies": len(strategies),
            }
        )

        output_paths = self._save_parameter_outputs(
            run_ts=run_ts,
            parameters=parameters,
            results=results,
            detailed_results=detailed_results,
            strategies=strategies,
            table_text=table_text,
            source_log_path=run_logger.file_path,
        )
        console.print("\n[bold green]Saved parameter analysis artifacts:[/bold green]")
        for label, path in output_paths.items():
            console.print(f"- {label}: [cyan]{path}[/cyan]")

    def _run_parameter_pair(
        self,
        *,
        strategy,
        strategy_name: str,
        project_inputs: ProjectInputs,
        parameter: str,
        image_base64: str | None,
        image_media_type: str,
    ) -> tuple[LLMResponse, str]:
        base_prompt = self._construct_prompt(project_inputs, strategy_name)
        pair_prompts = self._get_pair_prompts(parameter=parameter, strategy_name=strategy_name)

        if not pair_prompts:
            instruction_override = self._get_instruction_override([parameter])
            response = strategy._run_technique(
                llm=self.llm_client,
                prompt=base_prompt,
                image_base64=image_base64,
                image_media_type=image_media_type,
                judge_llm=self.judge_client,
                instruction_override=instruction_override,
            )
            combined_prompt = f"{base_prompt}\n\n{instruction_override}".strip()
            return response, combined_prompt

        rendered_prompts = [
            self._render_pair_prompt(
                step,
                parameter=parameter,
                strategy_name=strategy_name,
                project_inputs=project_inputs,
                has_diagram=bool(image_base64),
            )
            for step in pair_prompts
        ]

        if len(rendered_prompts) == 1:
            instruction_override = self._get_instruction_override([parameter])
            combined_prompt = (
                f"{base_prompt}\n\n"
                f"PAIR-SPECIFIC PROMPT FOR '{parameter}':\n{rendered_prompts[0]}\n\n"
                f"{instruction_override}"
            )
            response = strategy._run_technique(
                llm=self.llm_client,
                prompt=combined_prompt,
                image_base64=image_base64,
                image_media_type=image_media_type,
                judge_llm=self.judge_client,
                instruction_override=None,
            )
            return response, combined_prompt

        response, executed_prompt = self._run_multi_step_pair_prompt(
            base_prompt=base_prompt,
            strategy_name=strategy_name,
            parameter=parameter,
            steps=rendered_prompts,
            image_base64=image_base64,
            image_media_type=image_media_type,
        )
        return response, executed_prompt

    def _run_multi_step_pair_prompt(
        self,
        *,
        base_prompt: str,
        strategy_name: str,
        parameter: str,
        steps: List[str],
        image_base64: str | None,
        image_media_type: str,
    ) -> tuple[LLMResponse, str]:
        """Execute a multi-step prompt cell sequentially for one parameter/technique pair."""
        opening_prompt = (
            f"{base_prompt}\n\n"
            f"PAIR CONTEXT:\n"
            f"- Prompt technique: {strategy_name}\n"
            f"- Criterion/Parameter: {parameter}\n"
            "You will receive sequential step prompts. Follow them in order."
        )

        messages: List[ChatMessage] = [
            ChatMessage(
                role="user",
                content=opening_prompt,
                image_base64=image_base64,
                image_media_type=image_media_type or "image/jpeg",
            )
        ]

        prompt_log_parts = [opening_prompt]
        last_response: LLMResponse | None = None

        for idx, step in enumerate(steps, start=1):
            is_last = idx == len(steps)
            step_prompt = step
            if is_last:
                step_prompt = (
                    f"{step_prompt}\n\n"
                    f"{self._get_instruction_override([parameter])}"
                )

            prompt_log_parts.append(f"STEP {idx}:\n{step_prompt}")
            messages.append(ChatMessage(role="user", content=step_prompt))

            last_response = self.llm_client.chat(messages)
            messages.append(ChatMessage(role="assistant", content=last_response.content))

        if last_response is None:
            raise RuntimeError("No response produced for multi-step pair prompt.")

        return last_response, "\n\n".join(prompt_log_parts)

    def _get_pair_prompts(self, *, parameter: str, strategy_name: str) -> List[str]:
        if not self.prompt_matrix:
            return []
        return self.prompt_matrix.get_prompts(parameter, strategy_name)

    def _render_pair_prompt(
        self,
        template: str,
        *,
        parameter: str,
        strategy_name: str,
        project_inputs: ProjectInputs,
        has_diagram: bool,
    ) -> str:
        rendered = str(template)
        architecture_description = (project_inputs.architecture_description or "").strip()
        if not architecture_description:
            architecture_description = "No architecture description provided."

        diagram_available = "yes" if has_diagram else "no"
        diagram_note = (
            "Architecture diagram is attached in image context."
            if has_diagram
            else "No architecture diagram is available."
        )

        replacements = {
            "{parameter}": parameter,
            "{criterion}": parameter,
            "{criteria}": parameter,
            "{technique}": strategy_name,
            "{prompt_technique}": strategy_name,
            "{project_title}": project_inputs.project_title or "",
            "{architecture_description}": architecture_description,
            "{architecture_text}": architecture_description,
            "{has_diagram}": diagram_available,
            "{diagram_available}": diagram_available,
            "{diagram_note}": diagram_note,
        }
        for token, replacement in replacements.items():
            rendered = rendered.replace(token, replacement)
        return rendered

    def _parse_single_parameter_response(self, content: str, parameter: str) -> str:
        parsed = self._parse_llm_response(content, [parameter]).get(parameter, "Not Detected")
        if parsed != "Error decoding response":
            return parsed
        return self._normalize_detection_value(content)

    def _construct_prompt(self, inputs: ProjectInputs, strategy_name: str) -> str:
        architecture_description = (inputs.architecture_description or "").strip()
        if not architecture_description:
            architecture_description = (
                "No system architecture description was provided. Base your detections only on the"
                " architecture diagram if attached; if neither text nor diagram is available, state"
                " that the information is unavailable."
            )

        strategy_directives = {
            "monolithic": "One-pass review: inspect description and diagram once, then produce final labels.",
            "cot": "Chain-of-Thought: collect evidence first, then decide labels in a second pass.",
            "expert": "Expert architect lens: prioritize consistency gaps, missing components, and technical ambiguity.",
            "cove": "Chain-of-Verification: draft labels, verify each against evidence, then output corrected labels.",
            "rcot": "Reverse reasoning: start from each required parameter, then back-track evidence in inputs.",
            "two_model": "Two-model flow: generator drafts labels, judge validates, generator finalizes.",
        }
        strategy_directive = strategy_directives.get(strategy_name, "Review inputs and decide labels.")

        prompt_text = f"""
    ANALYZE THE FOLLOWING PROJECT DETAILS FOR PARAMETER DETECTION.

    STRATEGY: {strategy_name}
    STRATEGY DIRECTIVE: {strategy_directive}

    **System Architecture Description:**
    {architecture_description}

    **Architecture Diagram:**
    (See attached image if provided)

    IMPORTANT: Use only the architecture description and architecture image as evidence.
    """
        return prompt_text

    def _get_instruction_override(self, parameters: List[str]) -> str:
        param_list_str = "\n".join([f"- {p}" for p in parameters])
        return f"""
YOUR TASK is to detect if the following parameters are present using ONLY the System Architecture Description text and the Architecture Diagram image.

PARAMETERS:
{param_list_str}

REQUIREMENTS:
1. Review only the Architecture Description text and the Architecture Diagram image. Ignore titles and abstracts.
2. If the architecture description text is missing, explicitly note that in your reasoning and rely solely on the image (if present). If both are missing, mark all parameters as "Not Detected".
3. For each parameter, output ONLY "Detected" or "Not Detected".
4. Provide the output in STRICT JSON format.

JSON FORMAT:
{{
    "Parameter Name 1": "Detected",
    "Parameter Name 2": "Not Detected",
    ...
}}
Do NOT include any other text, markdown blocks, or explanations. ONLY the JSON.
Use EXACTLY the parameter strings above as JSON keys. Do not rename, summarize, or paraphrase keys.
"""

    def _parse_llm_response(self, content: str, parameters: List[str]) -> Dict[str, str]:
        # Simple JSON parsing, cleaning markdown
        content = str(content)  # Ensure string
        # Clean markdown wrappers
        content = content.replace("```json", "").replace("```", "").strip()
        # Find start and end of JSON object
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end != -1:
            content = content[start:end]

        try:
            data = json.loads(content)
            normalized_data = {p: "Not Detected" for p in parameters}
            canonical_map = {self._canonical_key(p): p for p in parameters}

            for k, v in data.items():
                if not isinstance(k, str):
                    continue

                matched_key = None
                if k in normalized_data:
                    matched_key = k
                else:
                    matched_key = canonical_map.get(self._canonical_key(k))

                if matched_key:
                    normalized_data[matched_key] = self._normalize_detection_value(v)

            return normalized_data
        except json.JSONDecodeError:
            return {p: "Error decoding response" for p in parameters}

    def _canonical_key(self, text: str) -> str:
        return " ".join(str(text).strip().lower().split())

    def _normalize_detection_value(self, value: Any) -> str:
        text = str(value).strip().lower()
        if text in {"yes", "true", "1", "present"}:
            return "Detected"
        if text in {"no", "false", "0", "absent"}:
            return "Not Detected"
        if "not detected" in text:
            return "Not Detected"
        if "detected" in text:
            return "Detected"
        return "Not Detected"

    def _is_detected(self, value: str) -> bool:
        if isinstance(value, str):
            return "detected" in value.lower() and "not detected" not in value.lower()
        return False

    def _display_table(self, results: Dict[str, Dict[str, int]], parameters: List[str]):
        table = Table(title="Parameter Detection by Strategy")
        
        table.add_column("Strategy", justify="left", style="cyan", no_wrap=True)
        for param in parameters:
            table.add_column(param, justify="center")
            
        for strategy, counts in results.items():
            row = [strategy]
            for param in parameters:
                row.append(str(counts.get(param, 0)))
            table.add_row(*row)
            
        console.print(table)

    def _build_table_text(self, results: Dict[str, Dict[str, int]], parameters: List[str]) -> str:
        lines = []
        header = ["strategy", *parameters]
        lines.append(" | ".join(header))
        lines.append("-" * max(40, len(lines[0]) + 5))

        for strategy, counts in results.items():
            row = [strategy]
            for param in parameters:
                row.append(str(counts.get(param, 0)))
            lines.append(" | ".join(row))

        return "\n".join(lines)

    def _save_parameter_outputs(
        self,
        *,
        run_ts: str,
        parameters: List[str],
        results: Dict[str, Dict[str, int]],
        detailed_results: List[Dict[str, Any]],
        strategies: List[str],
        table_text: str,
        source_log_path: Path,
    ) -> Dict[str, str]:
        base_name = f"{run_ts}_parameter_analysis"
        json_path = self.output_dir / f"{base_name}.json"
        txt_path = self.output_dir / f"{base_name}.txt"
        pdf_path = self.output_dir / f"{base_name}.pdf"
        copied_log_path = self.output_dir / f"{base_name}_llm.log"

        payload = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "model": settings.ollama_model,
                "strategies": strategies,
            },
            "parameters": parameters,
            "aggregated_counts": results,
            "detailed_results": detailed_results,
            "table": table_text,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Parameter Analysis Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(table_text)
            f.write("\n\n")
            f.write("Detailed Results\n")
            f.write("-" * 60 + "\n")
            for item in detailed_results:
                f.write(f"Project: {item.get('project_id')} | Strategy: {item.get('strategy')}\n")
                f.write(json.dumps(item.get("analysis", {}), ensure_ascii=False, indent=2))
                f.write("\n\n")

        self._generate_parameter_pdf(pdf_path=pdf_path, table_text=table_text, detailed_results=detailed_results)
        shutil.copyfile(source_log_path, copied_log_path)

        return {
            "summary_json": str(json_path),
            "summary_txt": str(txt_path),
            "summary_pdf": str(pdf_path),
            "llm_log_copy": str(copied_log_path),
            "llm_log_original": str(source_log_path),
        }

    def _generate_parameter_pdf(
        self,
        *,
        pdf_path: Path,
        table_text: str,
        detailed_results: List[Dict[str, Any]],
    ) -> None:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, "Parameter Analysis Report", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("helvetica", "", 10)
        pdf.multi_cell(0, 6, table_text)

        pdf.ln(4)
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, "Detailed Results", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("helvetica", "", 9)
        for item in detailed_results:
            line = f"Project: {item.get('project_id')} | Strategy: {item.get('strategy')}"
            pdf.multi_cell(0, 5, line)
            body = json.dumps(item.get("analysis", {}), ensure_ascii=False)
            pdf.multi_cell(0, 5, body)
            pdf.ln(2)

        pdf.output(str(pdf_path))

if __name__ == "__main__":
    pipeline = ParameterAnalysisPipeline()
    pipeline.analyze()
