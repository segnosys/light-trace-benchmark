#!/usr/bin/env python3
"""
Live dashboard for simulation benchmark visualization
"""

import json
import argparse
import statistics
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
from dash import Dash, html, dcc, callback, Input, Output
import plotly.graph_objects as go


# Module-level so closures don't capture loop-local rebindings (B023). 10 colors
# x 4 shapes lets us label up to 40 distinct sessions before symbols repeat.
SESSION_PALETTE_COLORS = [
    "#2ecc71",  # Green
    "#3498db",  # Blue
    "#9b59b6",  # Purple
    "#f39c12",  # Orange
    "#1abc9c",  # Teal
    "#e91e63",  # Pink
    "#00bcd4",  # Cyan
    "#8bc34a",  # Light green
    "#ff9800",  # Amber
    "#673ab7",  # Deep purple
]
SESSION_PALETTE_SHAPES = ["circle", "square", "diamond", "triangle-up"]


def session_style(idx: int):
    """Stable color + shape for a given session index, paged through the palette."""
    color = SESSION_PALETTE_COLORS[idx % len(SESSION_PALETTE_COLORS)]
    shape = SESSION_PALETTE_SHAPES[(idx // len(SESSION_PALETTE_COLORS)) % len(SESSION_PALETTE_SHAPES)]
    return color, shape


@dataclass
class MetricPoint:
    """Single data point from metrics.jsonl"""
    timestamp: float
    elapsed_seconds: float
    prefill_tps: float = 0  # Prefill tokens/sec (1s window)
    prefill_tps_window: float = 0  # Prefill tokens/sec (configurable window)
    prefill_tpm_per_gpu: float = 0  # Prefill TPM per GPU
    generation_tps: float = 0  # Generation TPS (MTP compensated)
    cache_hit_rate: float = 0  # Cache hit rate (0-1)
    ideal_cache_hit_rate: float = 0  # Ideal cache hit rate (assuming no eviction)
    requests_completed: int = 0
    requests_sent: int = 0
    errors: int = 0
    in_flight: int = 0
    num_sessions_active: int = 0
    num_sessions_retired: int = 0
    num_sessions_abandoned: int = 0
    num_sessions_total: int = 0
    sessions_created_by_rate: int = 0
    sessions_abandoned_by_rate: int = 0
    gpus: int = 1
    window_size: float = 15.0
    new_session_times: List[float] = None  # Timestamps of natural new session requests
    forced_session_times: List[float] = None  # Timestamps of forced new session requests (via keypress)
    existing_session_requests: List = None  # [[time, session_id], ...] for coloring by session
    new_planned_prompt_lengths: List[int] = None  # New planned prompt lengths this interval (SEND order)
    new_planned_ideal_cache_hit_rates: List[float] = None  # New planned ideal cache hit rates (SEND order)
    new_prompt_lengths: List[int] = None  # New prompt lengths this interval (COMPLETION order)
    new_generation_lengths: List[int] = None  # New generation lengths this interval
    new_cache_hit_rates: List[float] = None  # New per-request cache hit rates this interval
    new_ideal_cache_hit_rates: List[float] = None  # New ideal cache hit rates this interval
    new_inter_arrival_times: List[float] = None  # New inter-arrival times this interval
    new_ttfts: List[float] = None  # New TTFT values this interval (seconds)
    new_acceptance_lengths: List[float] = None  # New per-request acceptance lengths
    new_acceptance_rates: List[float] = None  # New per-request acceptance rates


@dataclass
class BenchmarkTrace:
    """Data for one line on the graph"""
    label: str              # e.g., "my-test"
    benchmark_name: str     # e.g., "my-test/2025-01-19-14-23-45"
    metrics: List[MetricPoint]
    file_path: Path
    last_position: int = 0  # For incremental reading
    all_new_session_times: List[float] = None  # Accumulated natural new session timestamps
    all_forced_session_times: List[float] = None  # Accumulated forced session timestamps
    all_existing_session_requests: List = None  # Accumulated [[time, session_id], ...]
    all_planned_prompt_lengths: List[int] = None  # Accumulated planned prompt lengths (SEND order)
    all_planned_ideal_cache_hit_rates: List[float] = None  # Accumulated planned ideal cache hit rates (SEND order)
    all_prompt_lengths: List[int] = None  # Accumulated prompt lengths (COMPLETION order)
    all_generation_lengths: List[int] = None  # Accumulated generation lengths
    all_cache_hit_rates: List[float] = None  # Accumulated per-request cache hit rates
    all_ideal_cache_hit_rates: List[float] = None  # Accumulated ideal cache hit rates
    all_inter_arrival_times: List[float] = None  # Accumulated inter-arrival times
    metadata: Dict = None  # Config from metadata.json
    log_analysis: Dict = None  # Log analysis data from log_analysis.json


def load_metadata(benchmark_dir: Path) -> Dict:
    """Load metadata.json from benchmark directory"""
    metadata_file = benchmark_dir / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def load_log_analysis(benchmark_dir: Path) -> Optional[Dict]:
    """Load log_analysis.json from benchmark directory if it exists"""
    analysis_file = benchmark_dir / "log_analysis.json"
    if analysis_file.exists():
        try:
            with open(analysis_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def format_num(val, decimals=1):
    """Format large numbers with K/M suffix"""
    if val >= 1_000_000:
        return f"{val/1_000_000:.{decimals}f}M"
    elif val >= 1_000:
        return f"{val/1_000:.{decimals}f}K"
    else:
        return f"{val:.{decimals}f}"


def percentile(data, p):
    """Calculate percentile of data"""
    if not data:
        return 0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def load_metrics(metrics_file: Path, start_position: int = 0) -> tuple[List[MetricPoint], int]:
    """Parse metrics.jsonl file incrementally"""
    metrics = []

    if not metrics_file.exists():
        return metrics, start_position

    with open(metrics_file, 'r') as f:
        f.seek(start_position)
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)

                    metric = MetricPoint(
                        timestamp=data.get("timestamp", 0),
                        elapsed_seconds=data.get("elapsed_seconds", 0),
                        prefill_tps=data.get("prefill_tps", 0),
                        prefill_tps_window=data.get("prefill_tps_window", 0),
                        prefill_tpm_per_gpu=data.get("prefill_tpm_per_gpu", 0),
                        generation_tps=data.get("generation_tps", 0),
                        cache_hit_rate=data.get("cache_hit_rate", 0),
                        ideal_cache_hit_rate=data.get("ideal_cache_hit_rate", 0),
                        requests_completed=data.get("requests_completed", 0),
                        requests_sent=data.get("requests_sent", 0),
                        errors=data.get("errors", 0),
                        in_flight=data.get("in_flight", 0),
                        num_sessions_active=data.get("num_sessions_active", 0),
                        num_sessions_retired=data.get("num_sessions_retired", 0),
                        num_sessions_abandoned=data.get("num_sessions_abandoned", 0),
                        num_sessions_total=data.get("num_sessions_total", 0),
                        sessions_created_by_rate=data.get("sessions_created_by_rate", 0),
                        sessions_abandoned_by_rate=data.get("sessions_abandoned_by_rate", 0),
                        gpus=data.get("gpus", 1),
                        window_size=data.get("window_size", 15.0),
                        new_session_times=data.get("new_session_times", []),
                        forced_session_times=data.get("forced_session_times", []),
                        existing_session_requests=data.get("existing_session_requests", []),
                        new_planned_prompt_lengths=data.get("new_planned_prompt_lengths", []),
                        new_planned_ideal_cache_hit_rates=data.get("new_planned_ideal_cache_hit_rates", []),
                        new_prompt_lengths=data.get("new_prompt_lengths", []),
                        new_generation_lengths=data.get("new_generation_lengths", []),
                        new_cache_hit_rates=data.get("new_cache_hit_rates", []),
                        new_ideal_cache_hit_rates=data.get("new_ideal_cache_hit_rates", []),
                        new_inter_arrival_times=data.get("new_inter_arrival_times", []),
                        new_ttfts=data.get("new_ttfts", []),
                        new_acceptance_lengths=data.get("new_acceptance_lengths", []),
                        new_acceptance_rates=data.get("new_acceptance_rates", []),
                    )
                    metrics.append(metric)
                except json.JSONDecodeError:
                    pass
        new_position = f.tell()

    return metrics, new_position


def scan_benchmarks(root_dir: Path) -> Dict[str, BenchmarkTrace]:
    """Discover all benchmark data"""
    traces = {}

    if not root_dir.exists():
        return traces

    # Find all benchmark directories
    for name_dir in root_dir.iterdir():
        if not name_dir.is_dir():
            continue

        for timestamp_dir in name_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue

            benchmark_name = f"{name_dir.name}/{timestamp_dir.name}"

            # Look for metrics.jsonl in this directory
            metrics_file = timestamp_dir / "metrics.jsonl"
            if metrics_file.exists():
                trace_id = benchmark_name
                label = name_dir.name

                traces[trace_id] = BenchmarkTrace(
                    label=label,
                    benchmark_name=benchmark_name,
                    metrics=[],
                    file_path=metrics_file,
                    all_new_session_times=[],
                    all_forced_session_times=[],
                    all_existing_session_requests=[],
                    all_planned_prompt_lengths=[],
                    all_planned_ideal_cache_hit_rates=[],
                    all_prompt_lengths=[],
                    all_generation_lengths=[],
                    all_cache_hit_rates=[],
                    all_ideal_cache_hit_rates=[],
                    all_inter_arrival_times=[],
                    metadata=load_metadata(timestamp_dir),
                    log_analysis=load_log_analysis(timestamp_dir)
                )

    return traces


def create_dash_app(data_dir: Path = Path("benchmarks")) -> Dash:
    """Initialize dashboard application"""
    app = Dash(__name__)

    # Store for benchmark traces (will be updated dynamically)
    benchmark_traces = {}

    # Store data directory for rescanning
    app.data_dir = data_dir

    # Flask route on the underlying app.server for CSV export. URL:
    #   /export/<bench_id>.csv?fields=elapsed_seconds,generation_tps[,...]
    # bench_id is `<name>/<timestamp>` URL-encoded (forward slash OK in flask
    # path with <path:bench_id>). Default field set = the most-asked metrics.
    DEFAULT_EXPORT_FIELDS = [
        "elapsed_seconds", "prefill_tps", "generation_tps",
        "cache_hit_rate", "in_flight", "requests_completed", "errors",
    ]

    @app.server.route("/export/<path:bench_id>.csv")
    def export_csv(bench_id):
        from flask import abort, Response, request as flask_request

        # Fresh scan so the route works even before the user opens the UI.
        traces = scan_benchmarks(app.data_dir)
        if bench_id not in traces:
            abort(404, description=f"benchmark not found: {bench_id}")

        # Optional ?fields= filter; whitelist against MetricPoint dataclass
        # fields to prevent attribute traversal abuse.
        from dataclasses import fields as _fields
        allowed = {f.name for f in _fields(MetricPoint)}
        requested = (
            flask_request.args.get("fields", "").split(",")
            if flask_request.args.get("fields") else DEFAULT_EXPORT_FIELDS
        )
        cols = [c for c in requested if c in allowed] or DEFAULT_EXPORT_FIELDS

        points, _ = load_metrics(traces[bench_id].file_path)
        # CSV emit (no quoting needed — all values are numeric or short ints)
        rows = [",".join(cols)]
        for p in points:
            rows.append(",".join(str(getattr(p, c, "")) for c in cols))
        body = "\n".join(rows) + "\n"
        safe_filename = bench_id.replace("/", "__")
        return Response(
            body,
            mimetype="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{safe_filename}.csv"',
            },
        )

    # Custom CSS styling
    app.index_string = '''<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f5f5;
            margin: 0;
            padding: 20px;
            font-size: 16px;
        }
        .header {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
        }
        .graph-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .benchmark-browser {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-height: 400px;
            overflow-y: auto;
        }
        #benchmark-selector label {
            font-size: 16px;
            margin-bottom: 6px;
            display: block;
        }
        #benchmark-selector input[type="checkbox"] {
            margin-right: 8px;
        }
        .stats-panel {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
        }
        .stats-section {
            background: #f8f9fa;
            border-radius: 6px;
            padding: 15px;
        }
        .stats-section h4 {
            margin: 0 0 10px 0;
            color: #495057;
            font-size: 16px;
            font-weight: 600;
            border-bottom: 2px solid #dee2e6;
            padding-bottom: 5px;
        }
        .stats-row {
            display: flex;
            justify-content: space-between;
            margin: 4px 0;
            font-size: 15px;
        }
        .stats-label {
            color: #6c757d;
        }
        .stats-value {
            font-weight: 500;
            color: #212529;
        }
        .stats-value.highlight {
            color: #0d6efd;
            font-weight: 600;
        }
        /* Plotly chart fonts */
        .js-plotly-plot .plotly .gtitle { font-size: 18px !important; }
        .js-plotly-plot .plotly .xtitle, .js-plotly-plot .plotly .ytitle { font-size: 14px !important; }
        .js-plotly-plot .plotly .xtick text, .js-plotly-plot .plotly .ytick text { font-size: 13px !important; }
        .js-plotly-plot .plotly .legend text { font-size: 13px !important; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>'''

    # Create layout with combined metrics graph and in-flight graph
    app.layout = html.Div([
        html.H1("Peak Throughput Dashboard", className="header"),

        html.Div(id='status-message', style={'textAlign': 'center', 'marginBottom': '10px'}),

        html.Div([
            html.H3("Select Benchmarks"),
            dcc.Checklist(
                id='benchmark-selector',
                options=[],
                value=[],
                style={'columnCount': 3}
            ),
        ], className='benchmark-browser'),

        # Stats Panel
        html.Div(id='stats-panel', className='stats-panel'),

        # Selection Stats Panel (shown when user selects a range on the graph)
        html.Div(id='selection-stats', className='stats-panel', style={'display': 'none'}),

        # Averaging Window Slider
        html.Div([
            html.Label("Averaging Window: ", style={'marginRight': '10px', 'fontWeight': 'bold', 'whiteSpace': 'nowrap'}),
            html.Span(id='window-size-display', children='1s', style={'marginRight': '20px', 'minWidth': '50px', 'fontWeight': 'bold'}),
            html.Div([
                dcc.Slider(
                    id='window-size-slider',
                    min=1,
                    max=120,
                    step=1,
                    value=1,
                    marks={1: 'None', 10: '10s', 30: '30s', 60: '1m', 120: '2m'},
                    tooltip={'placement': 'bottom', 'always_visible': False}
                )
            ], style={'flex': '1', 'minWidth': '400px'})
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '15px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px'}),

        html.Div([
            dcc.Graph(id='combined-graph', figure=go.Figure()),
            html.Div(id='session-legend', style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'gap': '8px',
                'padding': '10px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '4px',
                'marginTop': '10px'
            })
        ], className='graph-container'),

        html.Div([
            html.Div([
                dcc.Graph(id="inflight-graph", figure=go.Figure())
            ], style={"width": "50%", "display": "inline-block"}),
            html.Div([
                dcc.Graph(id="new-sessions-rate-graph", figure=go.Figure())
            ], style={"width": "50%", "display": "inline-block"}),
        ], className="graph-container"),

        html.Div([
            dcc.Graph(id="sessions-graph", figure=go.Figure())
        ], className='graph-container'),

        html.Div([
            html.Div([
                dcc.Graph(id='acc-len-graph', figure=go.Figure())
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='acc-rate-graph', figure=go.Figure())
            ], style={'width': '50%', 'display': 'inline-block'}),
        ], className='graph-container'),

        html.Div([
            html.Div([
                dcc.Graph(id='qps-graph', figure=go.Figure())
            ], style={'width': '33.33%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='ttft-graph', figure=go.Figure())
            ], style={'width': '33.33%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='prompt-length-graph', figure=go.Figure())
            ], style={'width': '33.33%', 'display': 'inline-block'}),
        ], className='graph-container'),

        # Distribution histograms in a row
        html.Div([
            html.Div([
                dcc.Graph(id='prompt-length-hist', figure=go.Figure())
            ], style={'width': '25%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='generation-length-hist', figure=go.Figure())
            ], style={'width': '25%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='cache-hit-rate-hist', figure=go.Figure())
            ], style={'width': '25%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='inter-arrival-hist', figure=go.Figure())
            ], style={'width': '25%', 'display': 'inline-block'}),
        ], className='graph-container'),

        # Log Analysis Section (only shown if log_analysis.json exists)
        html.Div(id='log-analysis-section', children=[
            html.H3("Log Analysis", style={'marginBottom': '15px', 'color': '#2c3e50'}),
            html.Div([
                html.Div([
                    dcc.Graph(id='batch-size-dist-graph', figure=go.Figure())
                ], style={'width': '33.33%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(id='scheduling-latency-time-graph', figure=go.Figure())
                ], style={'width': '33.33%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(id='batch-size-time-graph', figure=go.Figure())
                ], style={'width': '33.33%', 'display': 'inline-block'}),
            ]),
        ], className='graph-container', style={'display': 'none'}),

        dcc.Interval(
            id='interval-component',
            interval=1000,  # 1 second
            n_intervals=0
        ),

        html.Div(id='trace-storage', style={'display': 'none'})
    ])

    @callback(
        [Output('combined-graph', 'figure'),
         Output('inflight-graph', 'figure'),
         Output('new-sessions-rate-graph', 'figure'),
         Output('sessions-graph', 'figure'),
         Output('acc-len-graph', 'figure'),
         Output('acc-rate-graph', 'figure'),
         Output('qps-graph', 'figure'),
         Output('ttft-graph', 'figure'),
         Output('prompt-length-graph', 'figure'),
         Output('prompt-length-hist', 'figure'),
         Output('generation-length-hist', 'figure'),
         Output('cache-hit-rate-hist', 'figure'),
         Output('inter-arrival-hist', 'figure'),
         Output('benchmark-selector', 'options'),
         Output('status-message', 'children'),
         Output('stats-panel', 'children'),
         Output('session-legend', 'children'),
         Output('selection-stats', 'children'),
         Output('selection-stats', 'style'),
         Output('window-size-display', 'children'),
         Output('log-analysis-section', 'style'),
         Output('batch-size-dist-graph', 'figure'),
         Output('scheduling-latency-time-graph', 'figure'),
         Output('batch-size-time-graph', 'figure')],
        [Input('benchmark-selector', 'value'),
         Input('interval-component', 'n_intervals'),
         Input('window-size-slider', 'value'),
         Input('combined-graph', 'relayoutData'),
         Input('inflight-graph', 'relayoutData'),
         Input('new-sessions-rate-graph', 'relayoutData'),
         Input('sessions-graph', 'relayoutData'),
         Input('acc-len-graph', 'relayoutData'),
         Input('acc-rate-graph', 'relayoutData'),
         Input('qps-graph', 'relayoutData'),
         Input('ttft-graph', 'relayoutData'),
         Input('prompt-length-graph', 'relayoutData')]
    )
    def update_graphs(selected_traces, n_intervals, window_size_slider, *relayout_datas):
        """Update all graphs with latest data and rescan for new benchmarks"""
        nonlocal benchmark_traces

        # Rescan for new benchmarks
        new_traces = scan_benchmarks(app.data_dir)

        # Count new benchmarks
        new_count = sum(1 for trace_id in new_traces if trace_id not in benchmark_traces)

        # Merge new traces with existing
        for trace_id, new_trace in new_traces.items():
            if trace_id not in benchmark_traces:
                benchmark_traces[trace_id] = new_trace

        # Create options for selector
        options = []
        def sort_key(item):
            trace_id, trace = item
            timestamp_num = 0
            if '/' in trace.benchmark_name:
                timestamp = trace.benchmark_name.split('/')[-1]
                try:
                    timestamp_num = int(timestamp.replace('-', ''))
                except ValueError:
                    timestamp_num = 0
            return (-timestamp_num, trace.label.lower())

        sorted_traces = sorted(benchmark_traces.items(), key=sort_key)
        for trace_id, trace in sorted_traces:
            parts = trace.benchmark_name.split('/')
            if len(parts) == 2:
                timestamp_str = parts[1]
                try:
                    dt = datetime.strptime(timestamp_str, "%Y-%m-%d-%H-%M-%S")
                    time_display = dt.strftime("%b %d, %H:%M")
                    label = f"{trace.label}  -  {time_display}"
                except ValueError:
                    label = trace.label
            else:
                label = trace.label
            options.append({'label': label, 'value': trace_id})

        if not benchmark_traces:
            # Empty-state UX: instead of a blank dashboard, tell the user
            # where the viewer is looking and what shape it expects.
            _data_dir_str = str(app.data_dir)
            status = html.Div([
                html.Strong("No benchmark runs found in "),
                html.Code(_data_dir_str),
                html.Br(),
                html.Span(
                    f"Launch a run with `lightrace-agent --data-dir {_data_dir_str} ...` "
                    "and the dashboard will pick it up automatically. The viewer scans for "
                ),
                html.Code("<name>/<YYYY-MM-DD-HH-MM-SS>/metrics.jsonl"),
                html.Span(" beneath the configured --data-dir."),
            ], style={"color": "#6c757d", "padding": "10px", "lineHeight": "1.6"})
        else:
            status = f"Found {len(benchmark_traces)} benchmark runs"
            if new_count > 0:
                status += f" (+{new_count} new)"

        # Update traces with latest data
        selected_traces = selected_traces or []
        for trace_id in selected_traces:
            if trace_id in benchmark_traces:
                trace = benchmark_traces[trace_id]
                new_metrics, new_pos = load_metrics(trace.file_path, trace.last_position)
                trace.metrics.extend(new_metrics)
                trace.last_position = new_pos

                # Accumulate session timeline data
                for m in new_metrics:
                    if m.new_session_times:
                        trace.all_new_session_times.extend(m.new_session_times)
                    if m.forced_session_times:
                        trace.all_forced_session_times.extend(m.forced_session_times)
                    if m.existing_session_requests:
                        trace.all_existing_session_requests.extend(m.existing_session_requests)
                    if m.new_planned_prompt_lengths:
                        trace.all_planned_prompt_lengths.extend(m.new_planned_prompt_lengths)
                    if m.new_planned_ideal_cache_hit_rates:
                        trace.all_planned_ideal_cache_hit_rates.extend(m.new_planned_ideal_cache_hit_rates)
                    if m.new_prompt_lengths:
                        trace.all_prompt_lengths.extend(m.new_prompt_lengths)
                    if m.new_generation_lengths:
                        trace.all_generation_lengths.extend(m.new_generation_lengths)
                    if m.new_cache_hit_rates:
                        trace.all_cache_hit_rates.extend(m.new_cache_hit_rates)
                    if m.new_ideal_cache_hit_rates:
                        trace.all_ideal_cache_hit_rates.extend(m.new_ideal_cache_hit_rates)
                    if m.new_inter_arrival_times:
                        trace.all_inter_arrival_times.extend(m.new_inter_arrival_times)

        # Track session colors for legend (across all traces)
        all_session_colors = {}  # session_num -> color

        # Calculate shared x-axis range from all selected traces
        max_elapsed = 0
        for trace_id in selected_traces:
            if trace_id in benchmark_traces:
                trace = benchmark_traces[trace_id]
                if trace.metrics:
                    trace_max = max(m.elapsed_seconds for m in trace.metrics)
                    max_elapsed = max(max_elapsed, trace_max)
        x_range = [0, max_elapsed * 1.02] if max_elapsed > 0 else None  # 2% padding
        x_range_key = "full"  # Used for uirevision to sync all graphs
        # Preserve user zoom selection from any time-series graph
        for relayout_data in relayout_datas:
            if relayout_data:
                if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
                    x_range = [relayout_data["xaxis.range[0]"], relayout_data["xaxis.range[1]"]]
                    x_range_key = f"{x_range[0]:.2f}-{x_range[1]:.2f}"
                    break
                elif "xaxis.range" in relayout_data:
                    x_range = list(relayout_data["xaxis.range"])
                    x_range_key = f"{x_range[0]:.2f}-{x_range[1]:.2f}"
                    break
                elif "xaxis.autorange" in relayout_data:
                    # User double-clicked to reset - use full range
                    x_range = [0, max_elapsed * 1.02] if max_elapsed > 0 else None
                    x_range_key = "full"
                    break

        # Helper function for moving average smoothing
        def smooth(values, window=5):
            if len(values) < window:
                return values
            smoothed = []
            for i in range(len(values)):
                start = max(0, i - window // 2)
                end = min(len(values), i + window // 2 + 1)
                smoothed.append(sum(values[start:end]) / (end - start))
            return smoothed

        # Helper function to bucket-average time series data
        def bucket_average(x_vals, y_vals, bucket_size):
            if not x_vals or not y_vals or bucket_size <= 1:
                return x_vals, y_vals  # No bucketing for 1s or less
            if len(x_vals) != len(y_vals):
                return x_vals, y_vals

            # Group by bucket
            buckets = {}
            for x, y in zip(x_vals, y_vals):
                bucket_idx = int(x // bucket_size)
                if bucket_idx not in buckets:
                    buckets[bucket_idx] = []
                buckets[bucket_idx].append((x, y))

            # Average each bucket
            x_out = []
            y_out = []
            for bucket_idx in sorted(buckets.keys()):
                points = buckets[bucket_idx]
                avg_x = sum(p[0] for p in points) / len(points)
                avg_y = sum(p[1] for p in points) / len(points)
                x_out.append(avg_x)
                y_out.append(avg_y)

            return x_out, y_out

        # Create combined graph with 3 y-axes using plain go.Figure()
        combined_fig = go.Figure()

        # Colors: Blue for TPM/GPU, Green for TPS, Red for Cache Hit Rate
        # Line thickness for A/B comparison: thick (2) for first, thin (1) for second

        for i, trace_id in enumerate(selected_traces):
            if trace_id not in benchmark_traces:
                continue

            trace = benchmark_traces[trace_id]
            if not trace.metrics:
                continue

            x_raw = [m.elapsed_seconds for m in trace.metrics]

            # Suffix for multiple benchmarks
            suffix = f" ({trace.label})" if len(selected_traces) > 1 else ""
            # Line width based on trace index: first trace thick, others thin
            line_width = 3 if i == 0 else 2

            # TPM/GPU - Blue (y1, left axis)
            y_tpm_raw = [m.prefill_tpm_per_gpu for m in trace.metrics]
            x, y_tpm = bucket_average(x_raw, y_tpm_raw, window_size_slider)
            combined_fig.add_trace(
                go.Scatter(
                    x=x, y=y_tpm,
                    mode='lines',
                    name=f"TPM/GPU{suffix}",
                    line=dict(color='#636EFA', width=line_width),
                    yaxis='y1'
                )
            )

            # Generation TPS - Green (y2, first right axis)
            y_tps_raw = [m.generation_tps for m in trace.metrics]
            x, y_tps = bucket_average(x_raw, y_tps_raw, window_size_slider)
            combined_fig.add_trace(
                go.Scatter(
                    x=x, y=y_tps,
                    mode='lines',
                    name=f"Gen TPS{suffix}",
                    line=dict(color='#00CC96', width=line_width),
                    yaxis='y2'
                )
            )

            # Cache Hit Rate - Red dashed (y3, second right axis, 0-100%)
            y_cache_raw = [m.cache_hit_rate * 100 for m in trace.metrics]
            x, y_cache = bucket_average(x_raw, y_cache_raw, window_size_slider)
            combined_fig.add_trace(
                go.Scatter(
                    x=x, y=y_cache,
                    mode='lines',
                    name=f"Actual Cache %{suffix}",
                    line=dict(color='#EF553B', width=line_width, dash='dot'),
                    yaxis='y3'
                )
            )

            # Ideal Cache Hit Rate - Orange (y3, same axis as actual)
            # Hidden by default - click legend to show
            # Uses completion-time windowed data (same samples as actual, guarantees ideal >= actual)
            y_ideal_cache_raw = [m.ideal_cache_hit_rate * 100 for m in trace.metrics]
            _, y_ideal_cache = bucket_average(x_raw, y_ideal_cache_raw, window_size_slider)
            combined_fig.add_trace(
                go.Scatter(
                    x=x, y=y_ideal_cache,
                    mode='lines',
                    name=f"Ideal Cache %{suffix}",
                    line=dict(color='#FFA500', width=line_width),
                    yaxis='y3',
                    visible='legendonly'
                )
            )
            # Session timeline dots (on y4 axis, normalized 0-1)
            # Color existing session requests by session_id
            if trace.all_existing_session_requests:
                # Group by session_id, preserving order of first appearance
                session_requests = {}
                session_order = []  # Track order of first appearance
                for req in trace.all_existing_session_requests:
                    time_val, session_id = req[0], req[1]
                    if session_id not in session_requests:
                        session_requests[session_id] = []
                        session_order.append(session_id)
                    session_requests[session_id].append(time_val)

                # Use module-level palette + helper so we don't capture
                # loop-local rebindings (was a B023 lint hit) and so the
                # palette lists aren't rebuilt for every trace iteration.
                get_session_style = session_style

                # Build session style map for legend and store globally
                for idx, session_id in enumerate(session_order):
                    color, shape = get_session_style(idx)
                    all_session_colors[idx] = (color, shape)

                # Plot each session with its own color+shape
                for idx, session_id in enumerate(session_order):
                    times = session_requests[session_id]
                    color, shape = all_session_colors[idx]
                    combined_fig.add_trace(
                        go.Scatter(
                            x=times,
                            y=[0.5] * len(times),
                            mode='markers',
                            name=f"Session {idx}",
                            showlegend=False,
                            marker=dict(color=color, size=8, opacity=0.8, symbol=shape),
                            yaxis='y4',
                            hovertemplate=f"Session {idx}<br>t=%{{x:.1f}}s<extra></extra>"
                        )
                    )

            # Natural new session requests - yellow stars
            if trace.all_new_session_times:
                combined_fig.add_trace(
                    go.Scatter(
                        x=trace.all_new_session_times,
                        y=[0.5] * len(trace.all_new_session_times),
                        mode='markers',
                        name=f"New Session{suffix}",
                        marker=dict(color='#FFD700', size=14, opacity=1.0, symbol='star', line=dict(color='black', width=1.5)),
                        yaxis='y4',
                        hoverinfo='x'
                    )
                )

            # Forced new session requests (via keypress) - red stars
            if trace.all_forced_session_times:
                combined_fig.add_trace(
                    go.Scatter(
                        x=trace.all_forced_session_times,
                        y=[0.5] * len(trace.all_forced_session_times),
                        mode='markers',
                        name=f"Forced Session{suffix}",
                        marker=dict(color='#FF0000', size=16, opacity=1.0, symbol='star', line=dict(color='black', width=1.5)),
                        yaxis='y4',
                        hoverinfo='x'
                    )
                )

        # Add toggleable horizontal lines for average TPM/GPU and TPS (sustain period only, per trace)
        for i, trace_id in enumerate(selected_traces):
            if trace_id not in benchmark_traces:
                continue
            trace = benchmark_traces[trace_id]
            if trace.metrics and trace.metadata:
                # Get sustain period bounds
                ramp_end = trace.metadata.get('ramp_duration_secs', 0)
                sustain_dur = trace.metadata.get('sustain_duration_secs', 0)
                sustain_end = ramp_end + sustain_dur if sustain_dur > 0 else trace.metrics[-1].elapsed_seconds
                
                # Filter to sustain period only
                sustain_metrics = [m for m in trace.metrics if ramp_end <= m.elapsed_seconds <= sustain_end]
                
                if sustain_metrics:
                    tpm_values = [m.prefill_tpm_per_gpu for m in sustain_metrics if m.prefill_tpm_per_gpu > 0]
                    tps_values = [m.generation_tps for m in sustain_metrics if m.generation_tps > 0]
                    
                    # X range for horizontal lines
                    x_min = trace.metrics[0].elapsed_seconds
                    x_max = trace.metrics[-1].elapsed_seconds
                    
                    # Suffix for multiple benchmarks
                    suffix = f" ({trace.label})" if len(selected_traces) > 1 else ""
                    
                    if tpm_values:
                        avg_tpm = sum(tpm_values) / len(tpm_values)
                        combined_fig.add_trace(go.Scatter(
                            x=[x_min, x_max],
                            y=[avg_tpm, avg_tpm],
                            mode='lines',
                            name=f'Avg TPM: {avg_tpm/1000:.0f}k{suffix}',
                            line=dict(color='#636EFA', width=1, dash='dot'),
                            yaxis='y1',
                            visible='legendonly'
                        ))
                    
                    if tps_values:
                        avg_tps = sum(tps_values) / len(tps_values)
                        combined_fig.add_trace(go.Scatter(
                            x=[x_min, x_max],
                            y=[avg_tps, avg_tps],
                            mode='lines',
                            name=f'Avg TPS: {avg_tps:.0f}{suffix}',
                            line=dict(color='#00CC96', width=1, dash='dot'),
                            yaxis='y2',
                            visible='legendonly'
                        ))

        # Update layout with four y-axes
        combined_fig.update_layout(
            uirevision=x_range_key,  # Preserve zoom/pan/visibility across updates
            title=dict(text="Throughput Metrics", y=0.98, x=0.5, xanchor="center"),
            margin=dict(t=100),  # Extra top margin for legend
            xaxis=dict(
                title="Time (seconds)",
                domain=[0.06, 0.88],  # Leave room for axes on both sides
                range=x_range
            ),
            yaxis=dict(
                title="TPM/GPU",
                title_font=dict(color='#636EFA'),
                tickfont=dict(color='#636EFA'),
                side='left',
                position=0.0
            ),
            yaxis2=dict(
                title="Generation TPS",
                title_font=dict(color='#00CC96'),
                tickfont=dict(color='#00CC96'),
                side='right',
                overlaying='y',
                anchor='x'
            ),
            yaxis3=dict(
                title="Cache %",
                title_font=dict(color='#EF553B'),
                tickfont=dict(color='#EF553B'),
                side='right',
                overlaying='y',
                anchor='free',
                position=0.94,
                range=[0, 100]
            ),
            yaxis4=dict(
                # Hidden axis for session dots at bottom of chart
                overlaying='y',
                range=[0, 10],  # Dots at y=0.5, so they appear at ~5% from bottom
                fixedrange=True,  # Prevent autoscale from moving dots
                showticklabels=False,
                showgrid=False,
                zeroline=False,
            ),
            hovermode='x unified',
            height=500,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.08,
                xanchor='center',
                x=0.5,
                itemclick='toggle',
                itemdoubleclick='toggleothers'
            )
        )

        # Create In-Flight Requests graph
        inflight_fig = go.Figure()
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

        for i, trace_id in enumerate(selected_traces):
            if trace_id not in benchmark_traces:
                continue

            trace = benchmark_traces[trace_id]
            if trace.metrics:
                x_raw = [m.elapsed_seconds for m in trace.metrics]
                y_raw = [m.in_flight for m in trace.metrics]
                x, y = bucket_average(x_raw, y_raw, window_size_slider)
                inflight_fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='lines',
                    name=trace.label,
                    line=dict(color=colors[i % len(colors)])
                ))

        inflight_fig.update_layout(
            uirevision=x_range_key,
            title="In-Flight Requests",
            xaxis=dict(
                title="Time (seconds)",
                range=x_range
            ),
            yaxis_title="Concurrent Requests",
            hovermode='x unified',
            height=300
        )

        # Create New Session Ratio graph (rolling window)
        new_sessions_rate_fig = go.Figure()
        for i, trace_id in enumerate(selected_traces):
            if trace_id not in benchmark_traces:
                continue

            trace = benchmark_traces[trace_id]
            if trace.metrics:
                # Combine natural and forced new session times
                all_new_times = []
                if trace.all_new_session_times:
                    all_new_times.extend(trace.all_new_session_times)
                if trace.all_forced_session_times:
                    all_new_times.extend(trace.all_forced_session_times)
                all_new_times.sort()
                
                # Get existing session request times
                existing_times = []
                if trace.all_existing_session_requests:
                    existing_times = sorted([req[0] for req in trace.all_existing_session_requests])
                
                if all_new_times or existing_times:
                    window_size = max(30, window_size_slider)  # min 30s for this graph
                    x_vals = []
                    y_vals = []
                    
                    for m in trace.metrics:
                        t = m.elapsed_seconds
                        window_start = max(0, t - window_size)
                        # Count new sessions in window
                        new_count = sum(1 for st in all_new_times if window_start <= st <= t)
                        # Count existing session requests in window
                        existing_count = sum(1 for st in existing_times if window_start <= st <= t)
                        # Calculate ratio
                        total = new_count + existing_count
                        ratio = (new_count / total * 100) if total > 0 else 0
                        x_vals.append(t)
                        y_vals.append(ratio)
                    
                    if x_vals:
                        new_sessions_rate_fig.add_trace(go.Scatter(
                            x=x_vals, y=y_vals,
                            mode="lines",
                            name=trace.label,
                            line=dict(color=colors[i % len(colors)])
                        ))

        new_sessions_rate_fig.update_layout(
            uirevision=x_range_key,
            title="New Session Ratio",
            xaxis=dict(
                title="Time (seconds)",
                range=x_range
            ),
            yaxis_title="New Session %",
            hovermode="x unified",
            height=300
        )

        # Create Active Sessions graph
        sessions_fig = go.Figure()
        for i, trace_id in enumerate(selected_traces):
            if trace_id not in benchmark_traces:
                continue

            trace = benchmark_traces[trace_id]
            if trace.metrics:
                x_raw = [m.elapsed_seconds for m in trace.metrics]
                y_active_raw = [m.num_sessions_active for m in trace.metrics]
                y_total_raw = [m.num_sessions_total for m in trace.metrics]
                y_exited_raw = [m.num_sessions_retired + m.num_sessions_abandoned for m in trace.metrics]
                x, y_active = bucket_average(x_raw, y_active_raw, window_size_slider)
                _, y_total = bucket_average(x_raw, y_total_raw, window_size_slider)
                _, y_exited = bucket_average(x_raw, y_exited_raw, window_size_slider)
                sessions_fig.add_trace(go.Scatter(
                    x=x, y=y_active,
                    mode='lines',
                    name=f"Active ({trace.label})" if len(selected_traces) > 1 else "Active",
                    line=dict(color=colors[i % len(colors)])
                ))
                sessions_fig.add_trace(go.Scatter(
                    x=x, y=y_total,
                    mode='lines',
                    name=f"Total ({trace.label})" if len(selected_traces) > 1 else "Total",
                    line=dict(color=colors[i % len(colors)], dash='dot')
                ))
                sessions_fig.add_trace(go.Scatter(
                    x=x, y=y_exited,
                    mode='lines',
                    name=f"Exited ({trace.label})" if len(selected_traces) > 1 else "Exited (retired+abandoned)",
                    line=dict(color=colors[i % len(colors)], dash='dash')
                ))

        sessions_fig.update_layout(
            uirevision=x_range_key,
            title="Sessions Over Time",
            xaxis=dict(
                title="Time (seconds)",
                range=x_range
            ),
            yaxis_title="Number of Sessions",
            hovermode='x unified',
            height=300
        )

        # Create Acceptance Length over time graph
        acc_len_fig = go.Figure()
        for i, trace_id in enumerate(selected_traces):
            if trace_id not in benchmark_traces:
                continue

            trace = benchmark_traces[trace_id]
            if trace.metrics:
                # Collect individual data points
                scatter_x = []
                scatter_y = []
                for m in trace.metrics:
                    if m.new_acceptance_lengths:
                        for acc_len in m.new_acceptance_lengths:
                            scatter_x.append(m.elapsed_seconds)
                            scatter_y.append(acc_len)

                if scatter_x:
                    # Add scatter plot of individual points
                    acc_len_fig.add_trace(go.Scatter(
                        x=scatter_x, y=scatter_y,
                        mode='markers',
                        name=trace.label,
                        marker=dict(color=colors[i % len(colors)], size=4, opacity=0.5)
                    ))

                    # Add smoothed line
                    if len(scatter_y) >= 5:
                        y_smooth = smooth(scatter_y, window=max(5, window_size_slider // 3))
                        acc_len_fig.add_trace(go.Scatter(
                            x=scatter_x, y=y_smooth,
                            mode='lines',
                            name=f'{trace.label} avg',
                            line=dict(color=colors[i % len(colors)], width=2)
                        ))

        # Add horizontal average lines for each trace
        for i, trace_id in enumerate(selected_traces):
            if trace_id not in benchmark_traces:
                continue
            trace = benchmark_traces[trace_id]
            if trace.metrics:
                all_acc_lens = []
                for m in trace.metrics:
                    if m.new_acceptance_lengths:
                        all_acc_lens.extend(m.new_acceptance_lengths)
                if all_acc_lens:
                    avg_acc_len = sum(all_acc_lens) / len(all_acc_lens)
                    x_min = trace.metrics[0].elapsed_seconds
                    x_max = trace.metrics[-1].elapsed_seconds
                    suffix = f" ({trace.label})" if len(selected_traces) > 1 else ""
                    acc_len_fig.add_trace(go.Scatter(
                        x=[x_min, x_max],
                        y=[avg_acc_len, avg_acc_len],
                        mode='lines',
                        name=f'Avg: {avg_acc_len:.2f}{suffix}',
                        line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
                        visible=True
                    ))

        if selected_traces:
            first_trace_id = selected_traces[0]
            if first_trace_id in benchmark_traces:
                first_trace = benchmark_traces[first_trace_id]
                if first_trace.metadata and 'ramp_duration_secs' in first_trace.metadata:
                    ramp_end = first_trace.metadata['ramp_duration_secs']
                    acc_len_fig.add_vline(x=ramp_end, line_dash='dash', line_color='gray', line_width=1)

        acc_len_fig.update_layout(
            uirevision=x_range_key,
            title='MTP Acceptance Length Over Time',
            xaxis=dict(title='Time (seconds)', range=x_range),
            yaxis_title='Tokens per Step',
            hovermode='x unified',
            height=300
        )

        # Create Acceptance Rate over time graph
        acc_rate_fig = go.Figure()
        for i, trace_id in enumerate(selected_traces):
            if trace_id not in benchmark_traces:
                continue

            trace = benchmark_traces[trace_id]
            if trace.metrics:
                # Collect individual data points
                scatter_x = []
                scatter_y = []
                for m in trace.metrics:
                    if m.new_acceptance_rates:
                        for acc_rate in m.new_acceptance_rates:
                            scatter_x.append(m.elapsed_seconds)
                            scatter_y.append(acc_rate)

                if scatter_x:
                    # Add scatter plot of individual points
                    acc_rate_fig.add_trace(go.Scatter(
                        x=scatter_x, y=scatter_y,
                        mode='markers',
                        name=trace.label,
                        marker=dict(color=colors[i % len(colors)], size=4, opacity=0.5)
                    ))

                    # Add smoothed line
                    if len(scatter_y) >= 5:
                        y_smooth = smooth(scatter_y, window=max(5, window_size_slider // 3))
                        acc_rate_fig.add_trace(go.Scatter(
                            x=scatter_x, y=y_smooth,
                            mode='lines',
                            name=f'{trace.label} avg',
                            line=dict(color=colors[i % len(colors)], width=2)
                        ))

        # Add horizontal average lines for each trace
        for i, trace_id in enumerate(selected_traces):
            if trace_id not in benchmark_traces:
                continue
            trace = benchmark_traces[trace_id]
            if trace.metrics:
                all_acc_rates = []
                for m in trace.metrics:
                    if m.new_acceptance_rates:
                        all_acc_rates.extend(m.new_acceptance_rates)
                if all_acc_rates:
                    avg_acc_rate = sum(all_acc_rates) / len(all_acc_rates)
                    x_min = trace.metrics[0].elapsed_seconds
                    x_max = trace.metrics[-1].elapsed_seconds
                    suffix = f" ({trace.label})" if len(selected_traces) > 1 else ""
                    acc_rate_fig.add_trace(go.Scatter(
                        x=[x_min, x_max],
                        y=[avg_acc_rate, avg_acc_rate],
                        mode='lines',
                        name=f'Avg: {avg_acc_rate:.2f}{suffix}',
                        line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
                        visible=True
                    ))

        if selected_traces:
            first_trace_id = selected_traces[0]
            if first_trace_id in benchmark_traces:
                first_trace = benchmark_traces[first_trace_id]
                if first_trace.metadata and 'ramp_duration_secs' in first_trace.metadata:
                    ramp_end = first_trace.metadata['ramp_duration_secs']
                    acc_rate_fig.add_vline(x=ramp_end, line_dash='dash', line_color='gray', line_width=1)

        acc_rate_fig.update_layout(
            uirevision=x_range_key,
            title='MTP Acceptance Rate Over Time',
            xaxis=dict(title='Time (seconds)', range=x_range),
            yaxis_title='Acceptance Rate',
            hovermode='x unified',
            height=300
        )

        # Create QPS over time graph
        qps_fig = go.Figure()
        for i, trace_id in enumerate(selected_traces):
            if trace_id not in benchmark_traces:
                continue

            trace = benchmark_traces[trace_id]
            if len(trace.metrics) >= 2:
                x_raw = []
                y_qps_raw = []
                # Calculate instantaneous send rate from consecutive metrics (SEND order - deterministic)
                for j in range(1, len(trace.metrics)):
                    prev = trace.metrics[j-1]
                    curr = trace.metrics[j]
                    dt = curr.elapsed_seconds - prev.elapsed_seconds
                    if dt > 0:
                        # Use requests_sent for deterministic send rate (not completion rate)
                        requests_delta = curr.requests_sent - prev.requests_sent
                        send_rate = requests_delta / dt
                        x_raw.append(curr.elapsed_seconds)
                        y_qps_raw.append(send_rate)

                if x_raw:
                    # Apply bucket averaging
                    x, y_qps = bucket_average(x_raw, y_qps_raw, max(10, window_size_slider))
                    qps_fig.add_trace(go.Scatter(
                        x=x, y=y_qps,
                        mode='lines',
                        name=trace.label,
                        line=dict(color=colors[i % len(colors)])
                    ))

        # Add vertical line at ramp->sustain transition (use first trace's metadata)
        if selected_traces:
            first_trace_id = selected_traces[0]
            if first_trace_id in benchmark_traces:
                first_trace = benchmark_traces[first_trace_id]
                if first_trace.metadata and 'ramp_duration_secs' in first_trace.metadata:
                    ramp_end = first_trace.metadata['ramp_duration_secs']
                    qps_fig.add_vline(
                        x=ramp_end,
                        line_dash='dash',
                        line_color='gray',
                        line_width=1,
                        annotation_text=' ← ramp | sustain →',
                        annotation_position='top'
                    )

        # Add toggleable horizontal lines for average QPS (sustain period only, per trace)
        for i, trace_id in enumerate(selected_traces):
            if trace_id not in benchmark_traces:
                continue
            trace = benchmark_traces[trace_id]
            if trace.metrics and len(trace.metrics) >= 2 and trace.metadata:
                # Get sustain period bounds
                ramp_end = trace.metadata.get('ramp_duration_secs', 0)
                sustain_dur = trace.metadata.get('sustain_duration_secs', 0)
                sustain_end = ramp_end + sustain_dur if sustain_dur > 0 else trace.metrics[-1].elapsed_seconds
                
                # Find metrics at sustain boundaries
                sustain_start_metric = None
                sustain_end_metric = None
                for m in trace.metrics:
                    if m.elapsed_seconds >= ramp_end and sustain_start_metric is None:
                        sustain_start_metric = m
                    if m.elapsed_seconds <= sustain_end:
                        sustain_end_metric = m
                
                if sustain_start_metric and sustain_end_metric:
                    requests_in_sustain = sustain_end_metric.requests_sent - sustain_start_metric.requests_sent
                    time_in_sustain = sustain_end_metric.elapsed_seconds - sustain_start_metric.elapsed_seconds
                    if time_in_sustain > 0:
                        avg_qps = requests_in_sustain / time_in_sustain
                        x_min = trace.metrics[0].elapsed_seconds
                        x_max = trace.metrics[-1].elapsed_seconds
                        suffix = f" ({trace.label})" if len(selected_traces) > 1 else ""
                        qps_fig.add_trace(go.Scatter(
                            x=[x_min, x_max],
                            y=[avg_qps, avg_qps],
                            mode='lines',
                            name=f'Avg: {avg_qps:.2f}{suffix}',
                            line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
                            visible='legendonly'
                        ))

        qps_fig.update_layout(
            uirevision=x_range_key,  # Preserve visibility across updates
            title="QPS Over Time",
            xaxis=dict(title="Time (seconds)", range=x_range),
            yaxis_title="Requests/sec",
            hovermode='x unified',
            height=300
        )


        # Create TTFT over time graph
        ttft_fig = go.Figure()
        for i, trace_id in enumerate(selected_traces):
            if trace_id not in benchmark_traces:
                continue

            trace = benchmark_traces[trace_id]
            if trace.metrics:
                # Collect all TTFT values with their timestamps
                ttft_times = []
                ttft_values = []
                for m in trace.metrics:
                    if m.new_ttfts:
                        for ttft in m.new_ttfts:
                            ttft_times.append(m.elapsed_seconds)
                            ttft_values.append(ttft)  # Keep as seconds

                if ttft_times:
                    # Add scatter plot of individual TTFTs
                    ttft_fig.add_trace(go.Scatter(
                        x=ttft_times, y=ttft_values,
                        mode='markers',
                        name=trace.label,
                        marker=dict(color=colors[i % len(colors)], size=4, opacity=0.5)
                    ))

                    # Add rolling P50 line (window of 10 points)
                    if len(ttft_values) >= 5:
                        window = 10
                        p50_x = []
                        p50_y = []
                        for j in range(len(ttft_values)):
                            start = max(0, j - window // 2)
                            end = min(len(ttft_values), j + window // 2 + 1)
                            window_vals = sorted(ttft_values[start:end])
                            p50_x.append(ttft_times[j])
                            p50_y.append(window_vals[len(window_vals) // 2])

                        ttft_fig.add_trace(go.Scatter(
                            x=p50_x, y=p50_y,
                            mode='lines',
                            name=f"{trace.label} P50",
                            line=dict(color=colors[i % len(colors)], width=2)
                        ))

        # Add vertical line at ramp->sustain transition
        if selected_traces:
            first_trace_id = selected_traces[0]
            if first_trace_id in benchmark_traces:
                first_trace = benchmark_traces[first_trace_id]
                if first_trace.metadata and "ramp_duration_secs" in first_trace.metadata:
                    ramp_end = first_trace.metadata["ramp_duration_secs"]
                    ttft_fig.add_vline(
                        x=ramp_end,
                        line_dash="dash",
                        line_color="gray",
                        line_width=1
                    )

        ttft_fig.update_layout(
            uirevision=x_range_key,
            title="TTFT Over Time",
            xaxis=dict(title="Time (seconds)", range=x_range),
            yaxis_title="TTFT (s)",
            hovermode="x unified",
            height=300
        )

        # Create Average Prompt Length over time graph (using SEND order - deterministic)
        prompt_length_fig = go.Figure()
        for i, trace_id in enumerate(selected_traces):
            if trace_id not in benchmark_traces:
                continue

            trace = benchmark_traces[trace_id]
            if trace.metrics:
                x_raw = []
                y_prompt_len_raw = []
                for m in trace.metrics:
                    # Use planned prompt lengths (SEND order) for deterministic visualization
                    planned_lengths = m.new_planned_prompt_lengths if m.new_planned_prompt_lengths else m.new_prompt_lengths
                    if planned_lengths:
                        avg_len = statistics.mean(planned_lengths)
                        x_raw.append(m.elapsed_seconds)
                        y_prompt_len_raw.append(avg_len)

                if x_raw:
                    # Apply bucket averaging
                    x, y_prompt_len = bucket_average(x_raw, y_prompt_len_raw, max(10, window_size_slider))
                    prompt_length_fig.add_trace(go.Scatter(
                        x=x, y=y_prompt_len,
                        mode='lines',
                        name=trace.label,
                        line=dict(color=colors[i % len(colors)])
                    ))

        prompt_length_fig.update_layout(
            uirevision=x_range_key,
            title="Avg Prompt Length Over Time",
            xaxis=dict(title="Time (seconds)", range=x_range),
            yaxis_title="Tokens",
            hovermode='x unified',
            height=300
        )

        # Create distribution histograms
        # Collect all data for mean/median calculation
        # Use planned (SEND order) data for deterministic visualizations
        all_prompt_data = []
        all_gen_data = []
        all_cache_data = []
        all_ideal_cache_data = []

        for trace_id in selected_traces:
            if trace_id in benchmark_traces:
                trace = benchmark_traces[trace_id]
                # Use planned prompt lengths if available (SEND order - deterministic)
                prompt_data = trace.all_planned_prompt_lengths if trace.all_planned_prompt_lengths else trace.all_prompt_lengths
                all_prompt_data.extend(prompt_data or [])
                all_gen_data.extend(trace.all_generation_lengths or [])
                all_cache_data.extend([r * 100 for r in (trace.all_cache_hit_rates or [])])
                # Use planned ideal cache hit rates if available (SEND order - deterministic)
                ideal_cache_data = trace.all_planned_ideal_cache_hit_rates if trace.all_planned_ideal_cache_hit_rates else trace.all_ideal_cache_hit_rates
                all_ideal_cache_data.extend([r * 100 for r in (ideal_cache_data or [])])

        # Prompt Length Histogram - Blue (#636EFA) - using SEND order data
        prompt_hist_fig = go.Figure()
        for i, trace_id in enumerate(selected_traces):
            if trace_id not in benchmark_traces:
                continue
            trace = benchmark_traces[trace_id]
            # Use planned prompt lengths if available (SEND order - deterministic)
            prompt_data = trace.all_planned_prompt_lengths if trace.all_planned_prompt_lengths else trace.all_prompt_lengths
            if prompt_data:
                prompt_hist_fig.add_trace(go.Histogram(
                    x=prompt_data,
                    name=trace.label,
                    opacity=0.7,
                    marker_color='#636EFA',  # Blue
                    showlegend=False,
                    nbinsx=40,  # Finer binning
                ))
        # Add mean/median lines for prompt length
        if all_prompt_data:
            mean_val = statistics.mean(all_prompt_data)
            median_val = statistics.median(all_prompt_data)
            # Add mean line (red dashed)
            prompt_hist_fig.add_vline(x=mean_val, line_dash="dash", line_color="red", line_width=2)
            # Add median line (orange dashed)
            prompt_hist_fig.add_vline(x=median_val, line_dash="dash", line_color="orange", line_width=2)
            # Add dummy traces for legend
            prompt_hist_fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                name=f'Mean: {mean_val:,.0f}',
                line=dict(color='red', width=2, dash='dash')
            ))
            prompt_hist_fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                name=f'Median: {median_val:,.0f}',
                line=dict(color='orange', width=2, dash='dash')
            ))
        # Set prompt length x-axis range: 0 to max (show entire context length)
        prompt_x_max = max(all_prompt_data) * 1.05 if all_prompt_data else 200000  # 5% padding
        prompt_hist_fig.update_layout(
            title="Prompt Length Distribution",
            xaxis_title="Tokens",
            yaxis_title="Count",
            barmode='overlay',
            height=300,
            margin=dict(l=50, r=20, t=40, b=40),
            legend=dict(x=0.7, y=0.95),
            xaxis=dict(range=[0, prompt_x_max])
        )

        # Generation Length Histogram - Green (#00CC96)
        gen_hist_fig = go.Figure()
        # Calculate bin size based on data range (target ~40 bins)
        gen_max_val = max(all_gen_data) if all_gen_data else 4000
        gen_bin_size = max(gen_max_val / 40, 50)  # At least 50 tokens per bin
        for i, trace_id in enumerate(selected_traces):
            if trace_id not in benchmark_traces:
                continue
            trace = benchmark_traces[trace_id]
            if trace.all_generation_lengths:
                gen_hist_fig.add_trace(go.Histogram(
                    x=trace.all_generation_lengths,
                    name=trace.label,
                    opacity=0.7,
                    marker_color='#00CC96',  # Green
                    showlegend=False,
                    xbins=dict(start=0, size=gen_bin_size),
                ))
        # Add mean/median lines for generation length
        if all_gen_data:
            mean_val = statistics.mean(all_gen_data)
            median_val = statistics.median(all_gen_data)
            # Add mean line (red dashed)
            gen_hist_fig.add_vline(x=mean_val, line_dash="dash", line_color="red", line_width=2)
            # Add median line (orange dashed)
            gen_hist_fig.add_vline(x=median_val, line_dash="dash", line_color="orange", line_width=2)
            # Add dummy traces for legend
            gen_hist_fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                name=f'Mean: {mean_val:,.0f}',
                line=dict(color='red', width=2, dash='dash')
            ))
            gen_hist_fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                name=f'Median: {median_val:,.0f}',
                line=dict(color='orange', width=2, dash='dash')
            ))
        # Set generation length x-axis range: 0-4000 unless data exceeds 4000
        gen_x_max = 4000
        if all_gen_data and max(all_gen_data) > 4000:
            gen_x_max = max(all_gen_data) * 1.05  # 5% padding
        gen_hist_fig.update_layout(
            title="Generation Length Distribution",
            xaxis_title="Tokens",
            yaxis_title="Count",
            barmode='overlay',
            height=300,
            margin=dict(l=50, r=20, t=40, b=40),
            legend=dict(x=0.7, y=0.95),
            xaxis=dict(range=[0, gen_x_max])
        )

        # Cache Hit Rate Histogram - Red (#EF553B)
        cache_hist_fig = go.Figure()
        for i, trace_id in enumerate(selected_traces):
            if trace_id not in benchmark_traces:
                continue
            trace = benchmark_traces[trace_id]
            if trace.all_cache_hit_rates:
                # Convert to percentage
                cache_pcts = [r * 100 for r in trace.all_cache_hit_rates]
                cache_hist_fig.add_trace(go.Histogram(
                    x=cache_pcts,
                    name=trace.label,
                    opacity=0.7,
                    marker_color='#EF553B',  # Red
                    xbins=dict(start=0, end=100, size=2),  # Finer 2% bins
                    showlegend=False
                ))
        # Add mean/median lines for cache hit rate
        if all_cache_data:
            mean_val = statistics.mean(all_cache_data)
            median_val = statistics.median(all_cache_data)
            # Add mean line (red dashed)
            cache_hist_fig.add_vline(x=mean_val, line_dash="dash", line_color="red", line_width=2)
            # Add median line (orange dashed)
            cache_hist_fig.add_vline(x=median_val, line_dash="dash", line_color="orange", line_width=2)
            # Add dummy traces for legend
            cache_hist_fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                name=f'Mean: {mean_val:.1f}%',
                line=dict(color='red', width=2, dash='dash')
            ))
            cache_hist_fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                name=f'Median: {median_val:.1f}%',
                line=dict(color='orange', width=2, dash='dash')
            ))
        cache_hist_fig.update_layout(
            title="Cache Hit Rate Distribution",
            xaxis_title="Cache Hit Rate (%)",
            yaxis_title="Count",
            xaxis=dict(range=[0, 100]),
            barmode='overlay',
            height=300,
            margin=dict(l=50, r=20, t=40, b=40),
            legend=dict(x=0.02, y=0.95)
        )


        # Inter-Arrival Time Histogram - Purple (#AB63FA)
        inter_arrival_hist_fig = go.Figure()
        all_inter_arrival_data = []
        for i, trace_id in enumerate(selected_traces):
            if trace_id not in benchmark_traces:
                continue
            trace = benchmark_traces[trace_id]
            if trace.all_inter_arrival_times:
                # Use raw values in seconds
                inter_arrival_s = [t for t in trace.all_inter_arrival_times]
                all_inter_arrival_data.extend(inter_arrival_s)
                inter_arrival_hist_fig.add_trace(go.Histogram(
                    x=inter_arrival_s,
                    name=trace.label,
                    opacity=0.7,
                    marker_color='#AB63FA',  # Purple
                    nbinsx=40,  # Finer binning
                    showlegend=False
                ))
        # Add mean/median lines for inter-arrival times
        if all_inter_arrival_data:
            mean_val = statistics.mean(all_inter_arrival_data)
            median_val = statistics.median(all_inter_arrival_data)
            # Add mean line (purple dashed)
            inter_arrival_hist_fig.add_vline(x=mean_val, line_dash="dash", line_color="purple", line_width=2)
            # Add median line (magenta dashed)
            inter_arrival_hist_fig.add_vline(x=median_val, line_dash="dash", line_color="magenta", line_width=2)
            # Add dummy traces for legend
            inter_arrival_hist_fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                name=f'Mean: {mean_val:.3f}s',
                line=dict(color='purple', width=2, dash='dash')
            ))
            inter_arrival_hist_fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                name=f'Median: {median_val:.3f}s',
                line=dict(color='magenta', width=2, dash='dash')
            ))
        inter_arrival_hist_fig.update_layout(
            title="Inter-Arrival Time Distribution",
            xaxis_title="Inter-Arrival Time (s)",
            yaxis_title="Count",
            barmode='overlay',
            height=300,
            margin=dict(l=50, r=20, t=40, b=40),
            legend=dict(x=0.7, y=0.95)
        )

        # Build stats panel
        stats_panel_content = []
        if selected_traces:
            # Get first selected trace for config (or merge if multiple)
            first_trace_id = selected_traces[0]
            if first_trace_id in benchmark_traces:
                trace = benchmark_traces[first_trace_id]
                meta = trace.metadata or {}
                latest = trace.metrics[-1] if trace.metrics else None

                # Calculate dynamic stats
                total_requests = latest.requests_completed if latest else 0
                total_errors = latest.errors if latest else 0
                success_rate = ((total_requests - total_errors) / total_requests * 100) if total_requests > 0 else 0
                elapsed = latest.elapsed_seconds if latest else 0

                # Distribution stats
                prompt_mean = statistics.mean(all_prompt_data) if all_prompt_data else 0
                prompt_p50 = percentile(all_prompt_data, 50) if all_prompt_data else 0
                prompt_p90 = percentile(all_prompt_data, 90) if all_prompt_data else 0

                gen_mean = statistics.mean(all_gen_data) if all_gen_data else 0
                gen_p50 = percentile(all_gen_data, 50) if all_gen_data else 0
                gen_p90 = percentile(all_gen_data, 90) if all_gen_data else 0

                cache_mean = statistics.mean(all_cache_data) if all_cache_data else 0
                cache_p50 = percentile(all_cache_data, 50) if all_cache_data else 0
                
                # Calculate eviction stats
                ideal_cache_mean = statistics.mean(all_ideal_cache_data) if all_ideal_cache_data else 0
                # Eviction rate: percentage of expected cache that was evicted
                eviction_loss = max(0, ideal_cache_mean - cache_mean)  # Percentage points lost to eviction
                cache_efficiency = (cache_mean / ideal_cache_mean * 100) if ideal_cache_mean > 0 else 0

                # Calculate average throughput from SUSTAIN PERIOD ONLY (exclude ramp up/down)
                ramp_duration = meta.get('ramp_duration_secs', 0)
                sustain_duration = meta.get('sustain_duration_secs', 0)
                sustain_start = ramp_duration
                sustain_end = ramp_duration + sustain_duration if sustain_duration > 0 else elapsed

                if trace.metrics:
                    # Filter to sustain period only
                    sustain_metrics = [m for m in trace.metrics if sustain_start <= m.elapsed_seconds <= sustain_end]
                    valid_tpm = [m.prefill_tpm_per_gpu for m in sustain_metrics if m.prefill_tpm_per_gpu > 0]
                    valid_tps = [m.generation_tps for m in sustain_metrics if m.generation_tps > 0]
                    avg_tpm = statistics.mean(valid_tpm) if valid_tpm else 0
                    avg_tps = statistics.mean(valid_tps) if valid_tps else 0
                else:
                    avg_tpm = 0
                    avg_tps = 0

                num_gpus = meta.get('num_gpus', 1)

                # Config section
                config_rows = []
                
                # Mode detection
                mode = meta.get('mode', 'traffic-replay')
                is_realistic = mode == 'realistic'
                
                # Show mode at top of config
                mode_display = "Realistic" if is_realistic else "Traffic Replay"
                mode_color = '#00d4aa' if is_realistic else '#636EFA'
                config_rows.append(html.Div([
                    html.Span("Mode", className='stats-label'),
                    html.Span(mode_display, className='stats-value', style={'fontWeight': 'bold', 'color': mode_color})
                ], className='stats-row'))
                if num_gpus > 1:
                    config_rows.append(html.Div([
                        html.Span("GPUs", className='stats-label'),
                        html.Span(str(num_gpus), className='stats-value')
                    ], className='stats-row'))

                initial_qps = meta.get('initial_qps', 0)
                max_qps = meta.get('max_qps', 0)
                # Calculate actual QPS from data
                actual_qps = total_requests / elapsed if elapsed > 0 else 0
                config_rows.append(html.Div([
                    html.Span("QPS", className='stats-label'),
                    html.Span(f"{initial_qps:.2f} → {max_qps:.2f} (actual: {actual_qps:.2f})", className='stats-value')
                ], className='stats-row'))

                max_inflight = meta.get('max_inflight')
                if max_inflight:
                    config_rows.append(html.Div([
                        html.Span("Max in-flight", className='stats-label'),
                        html.Span(str(max_inflight), className='stats-value')
                    ], className='stats-row'))

                ramp = meta.get('ramp_duration_secs', 0)
                sustain = meta.get('sustain_duration_secs', 0)
                config_rows.append(html.Div([
                    html.Span("Ramp / Sustain", className='stats-label'),
                    html.Span(f"{ramp:.0f}s / {sustain:.0f}s", className='stats-value')
                ], className='stats-row'))

                # Mode-specific config
                if is_realistic:
                    # Realistic mode: show think time and session lifetime
                    think_mean = meta.get('think_time_mean', 0)
                    think_median = meta.get('think_time_median', 0)
                    if think_mean > 0:
                        config_rows.append(html.Div([
                            html.Span("Think time", className='stats-label'),
                            html.Span(f"μ={think_mean:.1f}s, med={think_median:.1f}s", className='stats-value')
                        ], className='stats-row'))
                    
                    lifetime_mean = meta.get('session_lifetime_mean', 0)
                    lifetime_median = meta.get('session_lifetime_median', 0)
                    if lifetime_mean > 0:
                        config_rows.append(html.Div([
                            html.Span("Session lifetime", className='stats-label'),
                            html.Span(f"μ={lifetime_mean:.0f}s, med={lifetime_median:.0f}s", className='stats-value')
                        ], className='stats-row'))
                    
                    max_sessions_cfg = meta.get('max_sessions', 0)
                    if max_sessions_cfg > 0:
                        config_rows.append(html.Div([
                            html.Span("Max sessions", className='stats-label'),
                            html.Span(str(max_sessions_cfg), className='stats-value')
                        ], className='stats-row'))

                # Traffic replay mode: show arrival pattern
                use_poisson = meta.get("use_poisson", False)
                poisson_shape = meta.get("poisson_shape", 1.0)
                if not is_realistic:
                    config_rows.append(html.Div([
                        html.Span("Arrival", className='stats-label'),
                        html.Span(f"Gamma(k={poisson_shape})" if use_poisson else "Uniform", className="stats-value")
                    ], className='stats-row'))

                # Calculate actual new session rate from data
                num_natural_new = len(trace.all_new_session_times) if trace.all_new_session_times else 0
                num_forced_new = len(trace.all_forced_session_times) if trace.all_forced_session_times else 0
                num_new_sessions = num_natural_new + num_forced_new
                num_existing_sessions = len(trace.all_existing_session_requests) if trace.all_existing_session_requests else 0
                total_session_requests = num_new_sessions + num_existing_sessions
                actual_new_session_rate = num_new_sessions / total_session_requests if total_session_requests > 0 else 0

                configured_rate = meta.get('new_session_rate', 0)
                if is_realistic:
                    # Realistic mode: new_session_rate is probability per second
                    config_rows.append(html.Div([
                        html.Span("New session rate", className='stats-label'),
                        html.Span(f"{configured_rate*100:.1f}%/sec", className='stats-value')
                    ], className='stats-row'))
                    # Session abandon rate (realistic mode only)
                    abandon_rate = meta.get('session_abandon_rate', 0)
                    config_rows.append(html.Div([
                        html.Span("Abandon rate", className='stats-label'),
                        html.Span(f"{abandon_rate*100:.1f}%/req", className='stats-value')
                    ], className='stats-row'))
                else:
                    # Traffic replay mode: new_session_rate is probability per request
                    config_rows.append(html.Div([
                        html.Span("New session rate", className='stats-label'),
                        html.Span(f"{configured_rate*100:.0f}% (actual: {actual_new_session_rate*100:.1f}%)", className='stats-value')
                    ], className='stats-row'))

                initial_sessions = meta.get('num_initial_sessions', 0)
                if initial_sessions:
                    config_rows.append(html.Div([
                        html.Span("Initial sessions", className='stats-label'),
                        html.Span(str(initial_sessions), className='stats-value')
                    ], className='stats-row'))

                mtp_overhead = meta.get('mtp_overhead_factor', 1.0)
                if mtp_overhead != 1.0:
                    config_rows.append(html.Div([
                        html.Span("MTP overhead", className='stats-label'),
                        html.Span(f"{mtp_overhead:.2f}", className='stats-value')
                    ], className='stats-row'))

                acc_len = meta.get('acc_len')
                if acc_len is not None:
                    config_rows.append(html.Div([
                        html.Span("Acc len", className='stats-label'),
                        html.Span(f"{acc_len:.2f}", className='stats-value')
                    ], className='stats-row'))

                # Results section
                results_rows = [
                    html.Div([
                        html.Span("Requests", className='stats-label'),
                        html.Span(f"{total_requests:,}", className='stats-value highlight')
                    ], className='stats-row'),
                    html.Div([
                        html.Span("Errors", className='stats-label'),
                        html.Span(f"{total_errors}", className='stats-value')
                    ], className='stats-row'),
                    html.Div([
                        html.Span("Success rate", className='stats-label'),
                        html.Span(f"{success_rate:.1f}%", className='stats-value')
                    ], className='stats-row'),
                    html.Div([
                        html.Span("Elapsed", className='stats-label'),
                        html.Span(f"{elapsed:.0f}s", className='stats-value')
                    ], className='stats-row'),
                ]

                # Sessions
                num_active = latest.num_sessions_active if latest else 0
                num_abandoned = latest.num_sessions_abandoned if latest else 0
                num_total = latest.num_sessions_total if latest else 0
                created_by_rate = latest.sessions_created_by_rate if latest else 0
                results_rows.append(html.Div([
                    html.Span("Sessions", className='stats-label'),
                    html.Span(f"{num_active}/{num_total} (+{created_by_rate} -{num_abandoned})", className='stats-value')
                ], className='stats-row'))

                # Distributions section
                dist_rows = [
                    html.Div([
                        html.Span("Prompt", className='stats-label'),
                        html.Span(f"μ={format_num(prompt_mean)}, p50={format_num(prompt_p50)}, p90={format_num(prompt_p90)}", className='stats-value')
                    ], className='stats-row'),
                    html.Div([
                        html.Span("Generation", className='stats-label'),
                        html.Span(f"μ={gen_mean:.0f}, p50={gen_p50:.0f}, p90={gen_p90:.0f}", className='stats-value')
                    ], className='stats-row'),
                    html.Div([
                        html.Span("Cache hit", className='stats-label'),
                        html.Span(f"μ={cache_mean:.1f}%, p50={cache_p50:.1f}%", className='stats-value')
                    ], className='stats-row'),
                    html.Div([
                        html.Span("Ideal cache", className='stats-label'),
                        html.Span(f"μ={ideal_cache_mean:.1f}%", className='stats-value')
                    ], className='stats-row'),
                    html.Div([
                        html.Span("Eviction loss", className='stats-label'),
                        html.Span(f"{eviction_loss:.1f}pp ({100-cache_efficiency:.1f}% of ideal)", className='stats-value')
                    ], className='stats-row'),
                ]

                # Throughput section
                throughput_rows = [
                    html.Div([
                        html.Span(f"Context {'TPM/GPU' if num_gpus > 1 else 'TPM'}", className='stats-label'),
                        html.Span(f"{format_num(avg_tpm)}", className='stats-value highlight')
                    ], className='stats-row'),
                    html.Div([
                        html.Span("Generation TPS", className='stats-label'),
                        html.Span(f"{avg_tps:.1f}", className='stats-value highlight')
                    ], className='stats-row'),
                ]

                stats_panel_content = html.Div([
                    html.Div([
                        html.H4("Config"),
                        *config_rows
                    ], className='stats-section'),
                    html.Div([
                        html.H4("Results"),
                        *results_rows
                    ], className='stats-section'),
                    html.Div([
                        html.H4("Distributions"),
                        *dist_rows
                    ], className='stats-section'),
                    html.Div([
                        html.H4("Avg Throughput"),
                        *throughput_rows
                    ], className='stats-section'),
                ], className='stats-grid')

        # Build session legend
        session_legend_items = []
        if all_session_colors:
            # Shape to CSS mapping
            shape_styles = {
                'circle': {'borderRadius': '50%'},
                'square': {'borderRadius': '0'},
                'diamond': {'borderRadius': '0', 'transform': 'rotate(45deg)'},
                'triangle-up': {'borderRadius': '0', 'clipPath': 'polygon(50% 0%, 0% 100%, 100% 100%)'},
            }

            # Add "New Session" entry first (gold star with black outline)
            session_legend_items.append(
                html.Div([
                    html.Span('★', style={
                        'fontSize': '16px',
                        'color': '#FFD700',
                        'textShadow': '-1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000',
                        'marginRight': '4px'
                    }),
                    html.Span('New', style={'fontSize': '12px', 'color': '#666', 'fontWeight': 'bold'})
                ], style={'display': 'flex', 'alignItems': 'center', 'padding': '2px 8px'})
            )
            # Add "Forced Session" entry (red star - triggered via keypress)
            session_legend_items.append(
                html.Div([
                    html.Span('★', style={
                        'fontSize': '18px',
                        'color': '#FF0000',
                        'textShadow': '-1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000',
                        'marginRight': '4px'
                    }),
                    html.Span('Forced', style={'fontSize': '12px', 'color': '#666', 'fontWeight': 'bold'})
                ], style={'display': 'flex', 'alignItems': 'center', 'padding': '2px 8px'})
            )
            # Add existing sessions with color + shape
            for session_num in sorted(all_session_colors.keys()):
                color, shape = all_session_colors[session_num]
                shape_style = shape_styles.get(shape, {})
                marker_style = {
                    'display': 'inline-block',
                    'width': '10px',
                    'height': '10px',
                    'backgroundColor': color,
                    'marginRight': '4px',
                    **shape_style
                }
                session_legend_items.append(
                    html.Div([
                        html.Span(style=marker_style),
                        html.Span(f'{session_num}', style={'fontSize': '12px', 'color': '#666'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'padding': '2px 6px'})
                )

        # Calculate selection stats if user has selected a range
        selection_stats_content = []
        selection_stats_style = {'display': 'none'}

        # Check if we have a valid x-axis range selection
        x_range_start = None
        x_range_end = None

        if relayout_data:
            # Handle both zoom and autorange reset
            if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
                x_range_start = relayout_data['xaxis.range[0]']
                x_range_end = relayout_data['xaxis.range[1]']
            elif 'xaxis.range' in relayout_data:
                x_range_start, x_range_end = relayout_data['xaxis.range']

        if x_range_start is not None and x_range_end is not None and selected_traces:
            # Calculate stats for the selected range
            selection_stats_style = {'display': 'block', 'marginTop': '10px'}

            # Collect metrics within the selected range
            range_tpm_values = []
            range_tps_values = []
            range_cache_values = []
            range_ideal_cache_values = []
            range_requests = 0

            for trace_id in selected_traces:
                if trace_id not in benchmark_traces:
                    continue
                trace = benchmark_traces[trace_id]

                for m in trace.metrics:
                    if x_range_start <= m.elapsed_seconds <= x_range_end:
                        if m.prefill_tpm_per_gpu > 0:
                            range_tpm_values.append(m.prefill_tpm_per_gpu)
                        if m.generation_tps > 0:
                            range_tps_values.append(m.generation_tps)
                        range_cache_values.append(m.cache_hit_rate * 100)
                        range_ideal_cache_values.append(m.ideal_cache_hit_rate * 100)
                        range_requests += 1

            if range_requests > 0:
                avg_tpm = statistics.mean(range_tpm_values) if range_tpm_values else 0
                avg_tps = statistics.mean(range_tps_values) if range_tps_values else 0
                avg_cache = statistics.mean(range_cache_values) if range_cache_values else 0
                avg_ideal_cache = statistics.mean(range_ideal_cache_values) if range_ideal_cache_values else 0
                eviction_loss = max(0, avg_ideal_cache - avg_cache)

                selection_stats_content = html.Div([
                    html.H4(f"Selection Stats ({x_range_start:.1f}s - {x_range_end:.1f}s)",
                            style={'margin': '0 0 10px 0', 'color': '#0d6efd'}),
                    html.Div([
                        html.Div([
                            html.Span("Avg TPM/GPU", className='stats-label'),
                            html.Span(f"{format_num(avg_tpm)}", className='stats-value highlight')
                        ], className='stats-row'),
                        html.Div([
                            html.Span("Avg Gen TPS", className='stats-label'),
                            html.Span(f"{avg_tps:.1f}", className='stats-value highlight')
                        ], className='stats-row'),
                        html.Div([
                            html.Span("Avg Cache Hit", className='stats-label'),
                            html.Span(f"{avg_cache:.1f}%", className='stats-value')
                        ], className='stats-row'),
                        html.Div([
                            html.Span("Avg Ideal Cache", className='stats-label'),
                            html.Span(f"{avg_ideal_cache:.1f}%", className='stats-value')
                        ], className='stats-row'),
                        html.Div([
                            html.Span("Eviction Loss", className='stats-label'),
                            html.Span(f"{eviction_loss:.1f}pp", className='stats-value')
                        ], className='stats-row'),
                        html.Div([
                            html.Span("Data Points", className='stats-label'),
                            html.Span(f"{range_requests}", className='stats-value')
                        ], className='stats-row'),
                    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '10px'})
                ])

        # Log Analysis Section - only show if any selected trace has log_analysis data
        log_analysis_style = {'display': 'none'}
        batch_size_dist_fig = go.Figure()
        scheduling_latency_time_fig = go.Figure()
        batch_size_time_fig = go.Figure()

        # Check if any selected trace has log analysis data
        log_analysis_data = None
        for trace_id in selected_traces:
            if trace_id in benchmark_traces:
                trace = benchmark_traces[trace_id]
                if trace.log_analysis:
                    log_analysis_data = trace.log_analysis
                    break

        if log_analysis_data:
            log_analysis_style = {'display': 'block'}

            # Batch Size Distribution
            batch_sizes_data = log_analysis_data.get('batch_sizes', [])
            if batch_sizes_data:
                batch_sizes = [d['batch_size'] for d in batch_sizes_data]
                batch_size_dist_fig.add_trace(go.Histogram(
                    x=batch_sizes,
                    marker_color='#3498db',
                    opacity=0.7,
                    name='Batch Size'
                ))
                if batch_sizes:
                    mean_val = sum(batch_sizes) / len(batch_sizes)
                    batch_size_dist_fig.add_vline(x=mean_val, line_dash="dash", line_color="red", line_width=2)
                batch_size_dist_fig.update_layout(
                    title=f"Batch Size Distribution (n={len(batch_sizes)})",
                    xaxis_title="Batch Size",
                    yaxis_title="Count",
                    height=300
                )

            # Scheduling Latency Breakdown Over Time (stacked area)
            sched_data = log_analysis_data.get('scheduling_breakdown', [])
            if sched_data:
                sched_with_ts = [d for d in sched_data if 'timestamp' in d]
                if sched_with_ts:
                    from datetime import datetime as dt
                    timestamps = []
                    for d in sched_with_ts:
                        try:
                            timestamps.append(dt.fromisoformat(d['timestamp']))
                        except (ValueError, TypeError, KeyError):
                            pass

                    if timestamps:
                        start_time = min(timestamps)
                        elapsed = [(t - start_time).total_seconds() for t in timestamps]

                        # Define phases and colors
                        phases = ['pre_tokenization', 'tokenization', 'enqueue', 'queue_wait', 'batch_setup']
                        phase_labels = ['Pre-tokenization', 'Tokenization', 'Enqueue', 'Queue Wait', 'Batch Setup']
                        phase_colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

                        # Extract and smooth each phase
                        window = 20
                        for phase, label, color in zip(phases, phase_labels, phase_colors):
                            values = [d.get(phase, 0) / 1000.0 for d in sched_with_ts]  # ms to seconds
                            if len(values) > window:
                                smoothed = smooth(values, window)
                            else:
                                smoothed = values

                            scheduling_latency_time_fig.add_trace(go.Scatter(
                                x=elapsed, y=smoothed,
                                mode='lines',
                                name=label,
                                line=dict(color=color, width=2),
                                stackgroup='one'  # Stacked area
                            ))

                scheduling_latency_time_fig.update_layout(
                    title="Scheduling Latency Breakdown Over Time",
                    xaxis_title="Time (s)",
                    yaxis_title="Latency (s)",
                    height=300,
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                )

            # Batch Size Over Time
            if batch_sizes_data:
                batch_with_ts = [d for d in batch_sizes_data if 'timestamp' in d]
                if batch_with_ts:
                    from datetime import datetime as dt
                    timestamps = []
                    for d in batch_with_ts:
                        try:
                            timestamps.append(dt.fromisoformat(d['timestamp']))
                        except (ValueError, TypeError, KeyError):
                            pass

                    if timestamps:
                        batch_sizes_ts = [d['batch_size'] for d in batch_with_ts]
                        start_time = min(timestamps)
                        elapsed = [(t - start_time).total_seconds() for t in timestamps]

                        batch_size_time_fig.add_trace(go.Scatter(
                            x=elapsed, y=batch_sizes_ts,
                            mode='markers',
                            marker=dict(color='#3498db', size=3, opacity=0.3),
                            name='Per-iteration'
                        ))

                        # Rolling average
                        if len(batch_sizes_ts) > 50:
                            window = 50
                            rolling_avg = smooth(batch_sizes_ts, window)
                            batch_size_time_fig.add_trace(go.Scatter(
                                x=elapsed, y=rolling_avg,
                                mode='lines',
                                line=dict(color='#2980b9', width=2),
                                name=f'Rolling avg (n={window})'
                            ))

                batch_size_time_fig.update_layout(
                    title="Batch Size Over Time",
                    xaxis_title="Time (s)",
                    yaxis_title="Batch Size",
                    height=300,
                    hovermode='x unified'
                )

        return combined_fig, inflight_fig, new_sessions_rate_fig, sessions_fig, acc_len_fig, acc_rate_fig, qps_fig, ttft_fig, prompt_length_fig, prompt_hist_fig, gen_hist_fig, cache_hist_fig, inter_arrival_hist_fig, options, status, stats_panel_content, session_legend_items, selection_stats_content, selection_stats_style, f'{window_size_slider}s', log_analysis_style, batch_size_dist_fig, scheduling_latency_time_fig, batch_size_time_fig


    return app


def main():
    """Run the dashboard server"""
    parser = argparse.ArgumentParser(
        description="Live dashboard for throughput benchmark visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--data-dir", type=str, default="benchmarks",
                       help="Directory containing benchmark data (default: benchmarks)")
    parser.add_argument("--port", type=int, default=8050,
                       help="Port to run the dashboard on (default: 8050)")
    parser.add_argument("--debug", action="store_true",
                       help="Run in debug mode")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    app = create_dash_app(data_dir)

    print(f"Starting dashboard at http://localhost:{args.port}")
    print(f"Reading benchmarks from: {data_dir.absolute()}")

    app.run(debug=args.debug, port=args.port, host='0.0.0.0')


if __name__ == "__main__":
    main()
