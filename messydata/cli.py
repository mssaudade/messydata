import json
import os
import sys

import click

from .schema import DatasetSchema, MultiDatasetSchema, load_config


@click.group()
def main():
    """MessyData — synthetic dirty data generator."""


@main.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--rows", "-n", default=1000, show_default=True, help="Number of rows (or rows per day in date mode).")
@click.option("--seed", "-s", default=42, show_default=True, help="Random seed.")
@click.option("--output", "-o", default=None, help="Output file path. Defaults to stdout (CSV).")
@click.option(
    "--format", "-f", "fmt",
    type=click.Choice(["csv", "parquet", "json", "jsonl"]),
    default=None,
    help="Output format. Inferred from --output extension if not set; defaults to csv.",
)
@click.option("--start-date", default=None, help="Generate for this date only, or start of a range (YYYY-MM-DD).")
@click.option("--end-date", default=None, help="End of date range (YYYY-MM-DD). Requires --start-date.")
def generate(config, rows, seed, output, fmt, start_date, end_date):
    """Generate a dataset from CONFIG and write it to a file or stdout.

    Use --start-date / --end-date to generate date-aware data. --rows becomes rows per day.
    """
    from .pipeline import MultiPipeline, Pipeline

    if end_date and not start_date:
        raise click.UsageError("--end-date requires --start-date.")

    # Infer format from output extension if not explicitly set
    if fmt is None:
        if output is None or os.path.splitext(output)[1] == "":
            fmt = "csv"
        elif output.endswith(".parquet"):
            fmt = "parquet"
        elif output.endswith(".json"):
            fmt = "json"
        elif output.endswith(".jsonl"):
            fmt = "jsonl"
        else:
            fmt = "csv"

    if fmt == "parquet" and output is None:
        raise click.UsageError("parquet format requires --output (cannot write to stdout).")

    pipeline = Pipeline.from_config(config)

    if isinstance(pipeline, MultiPipeline):
        # Several datasets: --output is a directory, one file per dataset.
        if output is None:
            raise click.UsageError("a multi-dataset config needs --output <directory>.")
        if start_date and end_date:
            tables = pipeline.run_date_range(start_date, end_date, rows_per_day=rows, seed=seed)
        elif start_date:
            raise click.UsageError("a multi-dataset config takes --start-date with --end-date.")
        else:
            tables = pipeline.run(n_rows=rows, seed=seed)
        os.makedirs(output, exist_ok=True)
        ext = fmt if fmt in ("parquet", "json", "jsonl") else "csv"
        for name, df in tables.items():
            path = os.path.join(output, f"{name}.{ext}")
            _write(df, ext, path)
            click.echo(f"Wrote {len(df)} rows to {path}", err=True)
        return

    if start_date and end_date:
        df = pipeline.run_date_range(start_date, end_date, rows_per_day=rows, seed=seed)
    elif start_date:
        df = pipeline.run_for_date(start_date, n_rows=rows, seed=seed)
    else:
        df = pipeline.run(n_rows=rows, seed=seed)

    if fmt == "csv":
        text = df.to_csv(index=False)
        if output:
            with open(output, "w") as f:
                f.write(text)
        else:
            click.echo(text, nl=False)

    elif fmt == "json":
        text = df.to_json(orient="records", indent=2)
        if output:
            with open(output, "w") as f:
                f.write(text)
        else:
            click.echo(text)

    elif fmt == "jsonl":
        text = df.to_json(orient="records", lines=True)
        if output:
            with open(output, "w") as f:
                f.write(text)
        else:
            click.echo(text, nl=False)

    elif fmt == "parquet":
        df.to_parquet(output, index=False)

    if output:
        click.echo(f"Wrote {len(df)} rows to {output}", err=True)


@main.command()
@click.argument("config", type=click.Path(exists=True))
def validate(config):
    """Validate CONFIG and report any errors."""
    try:
        schema = load_config(config)
        if isinstance(schema, MultiDatasetSchema):
            click.echo(f"✓ Valid — {len(schema.datasets)} datasets", err=False)
        else:
            click.echo(
                f"✓ Valid — {len(schema.fields)} fields, {len(schema.anomalies)} anomalies",
                err=False,
            )
        sys.exit(0)
    except Exception as e:
        click.echo(f"✗ Invalid: {e}", err=True)
        sys.exit(1)


@main.command()
def schema():
    """Print the JSON schema for the MessyData config format."""
    from .schema import DatasetSchema

    click.echo(json.dumps(DatasetSchema.model_json_schema(), indent=2))


def _write(df, fmt, path):
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "json":
        df.to_json(path, orient="records", indent=2)
    elif fmt == "jsonl":
        df.to_json(path, orient="records", lines=True)
    else:
        df.to_csv(path, index=False)
