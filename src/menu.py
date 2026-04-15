#!/usr/bin/env python3
"""Interactive menu-driven program for BE Project Critique Pipeline."""

import os
import sys
import glob

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

from config.settings import settings
from src.pipeline import CritiquePipeline
from src.input_handlers.csv_handler import CSVHandler
from src.parameter_pipeline import ParameterAnalysisPipeline


console = Console()


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """Print the application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║           BE PROJECT CRITIQUE PIPELINE                       ║
║           Local LLM-Powered Project Analysis (Ollama)        ║
╚══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold blue"))


def show_main_menu():
    """Display the main menu and return user choice."""
    console.print("\n[bold cyan]═══ MAIN MENU ═══[/bold cyan]\n")
    
    menu_items = [
        ("1", "Generate Critique (Manual Input)"),
        ("2", "Generate Critique (CSV Dataset)"),
        ("3", "View Ollama Status & Models"),
        ("4", "View Prompting Techniques"),
        ("5", "List Saved Critiques"),
        ("6", "View a Saved Critique"),
        ("7", "Run Parameter Analysis (Batch)"),
        ("8", "Exit")
    ]
    
    table = Table(show_header=False, box=None)
    table.add_column("Option", style="cyan", width=4)
    table.add_column("Description", style="white")
    
    for opt, desc in menu_items:
        table.add_row(f"[{opt}]", desc)
    
    console.print(table)
    console.print()
    
    choice = Prompt.ask(
        "[bold yellow]Enter your choice[/bold yellow]",
        choices=[str(i) for i in range(1, len(menu_items) + 1)],
        default="2"
    )
    return choice


def select_model(title: str = "SELECT MODEL") -> str:
    """Let user select an Ollama model."""
    console.print(f"\n[bold cyan]═══ {title} ═══[/bold cyan]\n")
    
    available_models = settings.get_available_models()
    default_model = settings.ollama_model
    
    if available_models:
        table = Table(title="Available Ollama Models")
        table.add_column("Option", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Default", style="yellow")
        
        for idx, model in enumerate(available_models, 1):
            is_default = "★ Default" if model == default_model or model.startswith(default_model + ":") else ""
            table.add_row(str(idx), model, is_default)
        
        console.print(table)
        console.print()
        
        choice = Prompt.ask(
            "[bold yellow]Select model number (or press Enter for default)[/bold yellow]",
            default="1"
        )
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available_models):
                return available_models[idx]
        except ValueError:
            pass
        
        return default_model
    else:
        console.print(f"[yellow]No models found. Using default: {default_model}[/yellow]")
        console.print("[dim]Make sure Ollama is running and models are pulled: ollama pull llava[/dim]")
        return default_model


def view_ollama_status():
    """View Ollama server status and available models."""
    console.print("\n[bold cyan]═══ OLLAMA STATUS ═══[/bold cyan]\n")
    
    is_available = settings.is_ollama_available()
    status = "[green]✓ CONNECTED[/green]" if is_available else "[red]✗ NOT RUNNING[/red]"
    
    console.print(f"Server URL:  {settings.ollama_base_url}")
    console.print(f"Status:      {status}")
    console.print(f"Default Model: {settings.ollama_model}")
    
    if is_available:
        models = settings.get_available_models()
        if models:
            console.print(f"\n[green]Installed Models ({len(models)}):[/green]")
            for m in models:
                console.print(f"  • {m}")
        else:
            console.print("\n[yellow]No models installed. Run: ollama pull llava[/yellow]")
    else:
        console.print("\n[red]Start Ollama with: ollama serve[/red]")
    
    input("\nPress Enter...")


def view_prompt_techniques():
    """View details about prompting techniques."""
    console.print("\n[bold cyan]═══ PROMPTING TECHNIQUES ═══[/bold cyan]\n")
    
    techs = {
        "expert": "Specific persona: Expert Software Architect (20 years exp). Focuses on inconsistencies, bottlenecks, overlaps, and clarity. [Recommended for this project]",
        "monolithic": "Single comprehensive prompt covering all aspects in one pass.",
        "cot": "Chain-of-Thought: Breaks down analysis into explicit reasoning steps (Title -> Architecture -> Description -> Feasibility).",
        "cove": "Chain-of-Verification: Generates a baseline critique, identifies potential inaccuracies, verifies them, and produces a corrected final response.",
        "rcot": "Reverse Chain-of-Thought: Analyzes the project, then works backward from potential problems to their root causes.",
        "two_model": "Two-Model Feedback: One model generates the critique, and a second 'Judge' model reviews and refines it for higher quality."
    }
    
    for name, desc in techs.items():
        console.print(Panel(desc, title=f"[bold]{name.upper()}[/bold]", border_style="green"))
        
    input("\nPress Enter...")


def list_saved_critiques():
    """List saved critique files."""
    console.print("\n[bold cyan]═══ SAVED CRITIQUES ═══[/bold cyan]\n")
    files = glob.glob("outputs/*.json")
    if not files:
        console.print("[yellow]No saved critiques found.[/yellow]")
    else:
        for f in sorted(files, reverse=True):
            console.print(f"- {f}")
    input("\nPress Enter...")


def view_unique_critique():
    """View content of a saved critique."""
    console.print("\n[bold cyan]═══ VIEW CRITIQUE ═══[/bold cyan]\n")
    files = sorted(glob.glob("outputs/*.json"), reverse=True)
    
    if not files:
        console.print("[yellow]No files found.[/yellow]")
        input("Press Enter...")
        return
        
    for idx, f in enumerate(files):
        console.print(f"[{idx+1}] {f}")
        
    choice = IntPrompt.ask("Select file number", default=1)
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            with open(files[idx], 'r') as f:
                content = f.read()
            console.print(Syntax(content, "json", theme="monokai", line_numbers=True))
        else:
            console.print("[red]Invalid selection[/red]")
    except:
        console.print("[red]Invalid input[/red]")
        
    input("\nPress Enter...")


def select_prompt_technique():
    """Let user select a prompting technique."""
    console.print("\n[bold cyan]═══ SELECT PROMPTING TECHNIQUE ═══[/bold cyan]\n")
    
    techniques = [
        ("1", "expert", "Expert Architect (Recommended)", "Expert persona, specific critique criteria"),
        ("2", "monolithic", "Monolithic (Baseline)", "Single comprehensive prompt"),
        ("3", "cot", "Chain-of-Thought", "Step-by-step reasoning"),
        ("4", "cove", "Chain-of-Verification", "Critique with self-correction"),
        ("5", "rcot", "Reverse Chain-of-Thought", "Problem-to-root reasoning"),
        ("6", "two_model", "Two-Model Strategy", "Generator + Reviewer judge")
    ]
    
    table = Table(title="Prompting Techniques")
    table.add_column("Option", style="cyan")
    table.add_column("Technique", style="green")
    table.add_column("Title", style="white")
    table.add_column("Description", style="dim")
    
    for opt, tech, title, desc in techniques:
        table.add_row(opt, tech, title, desc)
    
    console.print(table)
    console.print()
    
    choice = Prompt.ask(
        "[bold yellow]Select technique[/bold yellow]",
        choices=[str(i) for i in range(1, len(techniques) + 1)],
        default="1"
    )
    
    tech_map = {
        "1": "expert", "2": "monolithic", "3": "cot", 
        "4": "cove", "5": "rcot", "6": "two_model"
    }
    return tech_map[choice]


def get_manual_inputs():
    """Get project inputs manually."""
    console.print("\n[bold cyan]═══ ENTER PROJECT DETAILS ═══[/bold cyan]\n")
    inputs = {}
    
    title = Prompt.ask("[bold]Project Title[/bold]")
    if title: inputs["project_title"] = title
    
    abstract = Prompt.ask("\n[bold]Abstract[/bold] (project summary/description)")
    if abstract: inputs["abstract"] = abstract
    
    arch_desc = Prompt.ask("\n[bold]System Architecture Description[/bold] (optional, file path or text)", default="")
    if arch_desc: inputs["architecture_description"] = arch_desc
    
    img = Prompt.ask("\n[bold]Architecture Image[/bold] (path, optional)", default="")
    if img:
        if os.path.exists(img):
            inputs["architecture_image"] = img
        else:
            console.print(f"[yellow]Warning: File not found: {img}[/yellow]")
            
    return inputs


def generate_critique(mode="manual"):
    """Handle the critique generation flow."""
    clear_screen()
    print_banner()
    
    # Check Ollama availability
    if not settings.is_ollama_available():
        console.print("[red]Error: Ollama is not running![/red]")
        console.print("[yellow]Start it with: ollama serve[/yellow]")
        console.print("[yellow]Then pull a model: ollama pull llava[/yellow]")
        input("\nPress Enter...")
        return
    
    # 1. Select Model
    model = select_model("SELECT OLLAMA MODEL")

    # 2. Select Technique
    technique = select_prompt_technique()
    judge_model = None
    
    # special handling for two_model needing a judge
    if technique == "two_model":
        judge_model = select_model("SELECT JUDGE MODEL")

    # 3. Get Inputs
    inputs = {}
    normalized_id = None
    custom_filename = None
    
    if mode == "csv":
        # CSV Flow
        csv_path = Prompt.ask("Enter CSV Path", default="be_project_dataset.csv")
        try:
            handler = CSVHandler(csv_path)
        except Exception as e:
            console.print(f"[red]Error loading CSV: {e}[/red]")
            return

        group_id = Prompt.ask("Enter Group Number/ID (e.g. 1)")
        
        try:
            group_data = handler.get_group_data(group_id)
            if not group_data:
                console.print(f"[red]Group '{group_id}' not found in CSV![/red]")
                input("Press Enter...")
                return
            
            # Show summary
            console.print(f"\n[green]Found Group: {group_data.get('group_id')}[/green]")
            console.print(f"Title: {group_data.get('project_title', 'N/A')}")
            console.print(f"Image Path: {group_data.get('architecture_image_path', 'None')}")
            
            inputs["project_title"] = group_data["project_title"]
            inputs["abstract"] = group_data["abstract"]
            inputs["architecture_description"] = group_data["architecture_description"]
            inputs["architecture_image"] = group_data["architecture_image_path"]
            
            # Custom filename
            normalized_id = group_data.get('group_id', 'group_unknown')
            custom_filename = f"{normalized_id}.json"
            
        except Exception as e:
            console.print(f"[red]Error processing group data: {e}[/red]")
            input("Press Enter...")
            return
            
    else:
        # Manual Flow
        inputs = get_manual_inputs()
        if not inputs:
            console.print("[red]No inputs provided![/red]")
            return

    # 4. Confirm
    console.print("\n[bold cyan]═══ CONFIRMATION ═══[/bold cyan]\n")
    console.print(f"Mode:          [cyan]{mode.upper()}[/cyan]")
    console.print(f"Model:         [cyan]{model}[/cyan]")
    console.print(f"Technique:     [cyan]{technique}[/cyan]")
    if mode == "csv":
        console.print(f"Output File:   [cyan]{custom_filename}[/cyan]")
    
    if not Confirm.ask("\n[bold yellow]Proceed?[/bold yellow]"):
        return
        
    # 5. Execute
    pipeline = CritiquePipeline()
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task(f"Generating critique with {model}...", total=None)
            result = pipeline.generate_critique(
                prompt_technique=technique,
                model=model,
                judge_model=judge_model,
                project_id=normalized_id,
                **inputs
            )
            
            progress.update(task, completed=True)

        # Handle custom filename for CSV mode
        if mode == "csv" and normalized_id:
            if result.output_path and os.path.exists(result.output_path):
                safe_model = result.model.replace("/", "-").replace(" ", "_")
                new_filename = f"{normalized_id}_{technique}.json"
                    
                new_path = os.path.join(os.path.dirname(result.output_path), new_filename)
                try:
                    os.rename(result.output_path, new_path)
                    result.output_path = new_path
                except OSError as e:
                    console.print(f"[yellow]Warning: Could not rename file: {e}[/yellow]")
                
        # Display Results
        console.print("\n[bold green]✓ Done![/bold green]\n")
        
        output_txt = f"Model: {result.model}\nTechnique: {result.prompt_technique}\nTime: {result.execution_time:.2f}s"
        if result.output_path:
            output_txt += f"\nJSON Saved to: [green]{result.output_path}[/green]"
        if getattr(result, 'log_output_path', None):
            output_txt += f"\nLLM Log Saved to: [yellow]{result.log_output_path}[/yellow]"
        if getattr(result, 'pdf_output_path', None):
            output_txt += f"\nPDF Saved to: [cyan]{result.pdf_output_path}[/cyan]"
            
        console.print(Panel(
            output_txt,
            title="Result"
        ))

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    
    input("\nPress Enter...")


def main():
    """Main loop."""
    try:
        while True:
            clear_screen()
            print_banner()
            choice = show_main_menu()
            
            if choice == "1":
                generate_critique("manual")
            elif choice == "2":
                generate_critique("csv")
            elif choice == "3":
                view_ollama_status()
            elif choice == "4":
                view_prompt_techniques()
            elif choice == "5":
                list_saved_critiques()
            elif choice == "6":
                view_unique_critique()
            elif choice == "7":
                ParameterAnalysisPipeline().analyze()
                input("\nPress Enter to continue...")
            elif choice == "8":
                console.print("\n[yellow]Goodbye![/yellow]")
                break
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye![/yellow]")
    except Exception as e:
        console.print(f"\n[red]Unexpected Error: {e}[/red]")


if __name__ == "__main__":
    main()
