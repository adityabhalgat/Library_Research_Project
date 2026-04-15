"""Command Line Interface for BE Project Critique Pipeline."""

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.pipeline import CritiquePipeline
from config import settings

console = Console()

@click.group()
@click.version_option()
def cli():
    """BE Project Critique Pipeline - Local LLM-powered project analysis tool.
    
    Generate comprehensive critiques of Bachelor of Engineering project papers
    using Ollama (local open-source LLMs) and various prompting techniques.
    """
    pass


@cli.command()
@click.option(
    "--prompt-technique", "-p",
    type=click.Choice(["monolithic", "cot", "expert", "cove", "rcot", "two_model"]),
    required=True,
    help="Prompting technique to use"
)
@click.option(
    "--model", "-m",
    type=str,
    default=None,
    help="Ollama model to use (default: from .env, e.g. llava)"
)
@click.option(
    "--judge-model",
    type=str,
    default=None,
    help="Judge model for two-model strategy (default: same as main model)"
)
@click.option(
    "--architecture-image", "-i",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to architecture diagram image (JPG/PNG)"
)
@click.option(
    "--architecture-networkx", "-n",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to NetworkX graph pickle file"
)
@click.option(
    "--project-title", "-t",
    help="Project title (file path or inline text)"
)
@click.option(
    "--abstract", "-a",
    help="Project abstract (file path or inline text)"
)
@click.option(
    "--architecture-description", "-d",
    help="System architecture description (file path or inline text)"
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, writable=True),
    help="Output directory for critique files"
)
@click.option(
    "--no-save",
    is_flag=True,
    help="Don't save output to file (just display)"
)

def critique(
    prompt_technique, model, judge_model, architecture_image, architecture_networkx,
    project_title, abstract, architecture_description, output_dir, no_save
):
    """Generate a critique for a BE project.
    
    Examples:
    
        # Basic usage with Chain-of-Thought
        python -m src.cli critique -p cot -t "Library Management System" -a "This project aims to..."
        
        # With architecture image
        python -m src.cli critique -p expert -t "Smart Parking" -i ./arch.jpg
        
        # Using a specific model
        python -m src.cli critique -p monolithic -m llava-llama3 -t "IoT System"
    """
    # Validation
    if not any([project_title, abstract, architecture_description, architecture_image, architecture_networkx]):
        console.print("[red]Error: At least one input (project title, abstract, architecture description, or architecture image) must be provided.[/red]")
        return

    # Check Ollama availability
    if not settings.is_ollama_available():
        console.print("[red]Error: Ollama is not running! Start it with: ollama serve[/red]")
        return

    # Initialize pipeline
    pipeline = CritiquePipeline(output_dir=output_dir)
    
    # Run critique
    with console.status("[bold green]Generating critique with Ollama...[/bold green]"):
        try:
            result = pipeline.generate_critique(
                prompt_technique=prompt_technique,
                model=model,
                judge_model=judge_model,
                architecture_image=architecture_image,
                architecture_networkx=architecture_networkx,
                project_title=project_title,
                abstract=abstract,
                architecture_description=architecture_description,
                save=not no_save
            )
            
            # Display result
            output_txt = (
                f"[bold]Model:[/bold] {result.model}\n"
                f"[bold]Technique:[/bold] {result.prompt_technique}\n"
                f"[bold]Execution Time:[/bold] {result.execution_time:.2f}s\n"
                f"[bold]JSON Output:[/bold] {result.output_path or 'Not saved'}\n"
                f"[bold]Run Log:[/bold] {getattr(result, 'log_output_path', None) or 'Not saved'}"
            )
            if getattr(result, 'pdf_output_path', None):
                output_txt += f"\n[bold]PDF Output:[/bold] {result.pdf_output_path}"
                
            console.print(Panel(
                output_txt,
                title=f"Critique Result - Ollama ({result.model})",
                border_style="green"
            ))
            
            if result.output_path:
                console.print(f"Critique JSON saved to: [link=file://{result.output_path}]{result.output_path}[/link]")
            if getattr(result, 'log_output_path', None):
                console.print(f"Run log saved to: [link=file://{result.log_output_path}]{result.log_output_path}[/link]")
            if getattr(result, 'pdf_output_path', None):
                console.print(f"Critique PDF saved to: [link=file://{result.pdf_output_path}]{result.pdf_output_path}[/link]")
                    
        except Exception as e:
            console.print(f"[bold red]Error generating critique:[/bold red] {str(e)}")


@cli.command()
def list():
    """List all saved critiques."""
    pipeline = CritiquePipeline()
    critiques = pipeline.list_saved_critiques()
    
    if not critiques:
        console.print("[yellow]No saved critiques found.[/yellow]")
        return
        
    table = Table(title="Saved Critiques")
    table.add_column("Filename", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Technique", style="blue")
    table.add_column("Date", style="magenta")
    
    for c in critiques:
        if "error" in c:
            continue
        table.add_row(
            c.get("filename", "unknown"),
            c.get("model", "unknown"),
            c.get("prompt_technique", "unknown"),
            c.get("timestamp", "unknown")
        )
        
    console.print(table)


@cli.command()
def status():
    """Show Ollama status and available models."""
    # Ollama Status
    is_available = settings.is_ollama_available()
    status_text = "[green]✓ Running[/green]" if is_available else "[red]✗ Not running[/red]"
    
    console.print(Panel(
        f"[bold]Ollama Server:[/bold] {settings.ollama_base_url}\n"
        f"[bold]Status:[/bold] {status_text}\n"
        f"[bold]Default Model:[/bold] {settings.ollama_model}",
        title="Ollama Status",
        border_style="cyan"
    ))
    
    if is_available:
        models = settings.get_available_models()
        if models:
            table = Table(title="Installed Models")
            table.add_column("Model", style="green")
            for m in models:
                table.add_row(m)
            console.print(table)
        else:
            console.print("[yellow]No models installed. Run: ollama pull llava[/yellow]")
    
    console.print()
    
    # Techniques
    table_tech = Table(title="Available Prompting Techniques")
    table_tech.add_column("Technique", style="cyan")
    table_tech.add_column("Description")
    
    techniques = [
        ("monolithic", "Single comprehensive prompt covering all aspects"),
        ("cot", "Chain-of-Thought (Step-by-step reasoning)"),
        ("expert", "Expert Architect (Specific persona and criteria)"),
        ("cove", "Chain-of-Verification (Generate -> Verify -> Finalize)"),
        ("rcot", "Reverse Chain-of-Thought (Generate -> Abstract -> Compare -> Refine)"),
        ("two_model", "Two-Model (Primary Generator -> Judge -> Refiner)"),
    ]
    
    for tech, desc in techniques:
        table_tech.add_row(tech, desc)
        
    console.print(table_tech)


@cli.command()
@click.argument('filename')
def view(filename):
    """View a saved critique."""
    pipeline = CritiquePipeline()
    try:
        data = pipeline.load_critique(filename)
        console.print_json(data=data)
    except Exception as e:
        console.print(f"[red]Error loading critique: {e}[/red]")


if __name__ == "__main__":
    cli()
