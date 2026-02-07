"""Core modules for Dimension Quality Analyzer."""

from .excel_parser import load_excel, parse_focus_dimensions
from .statistics import (
    calc_out_of_spec,
    pick_spec_values,
    stats_table,
    cp_cpk_summary,
    cpk_with_rating,
    imr_spc_points,
    calculate_normalized_deviation,
    calculate_correlation_matrix,
    get_high_correlation_pairs,
)
from .visualization import (
    add_spec_lines,
    build_fig,
    apply_y_range,
    add_spec_edge_markers,
    add_out_of_spec_points,
    build_cpk_heatmap,
    build_normalized_deviation_chart,
    build_position_comparison_chart,
    build_imr_chart,
    build_correlation_heatmap,
    build_correlation_scatter,
)
from .export import (
    download_plot_button,
    download_excel_button,
    download_stats_excel,
    download_quality_reports,
    build_report_html,
    generate_pdf_report,
    download_pdf_report_button,
)
from .grouping import format_pos, assign_groups_vectorized

__all__ = [
    # Excel parser
    "load_excel",
    "parse_focus_dimensions",
    # Statistics
    "calc_out_of_spec",
    "pick_spec_values",
    "stats_table",
    "cp_cpk_summary",
    "cpk_with_rating",
    "imr_spc_points",
    "calculate_normalized_deviation",
    "calculate_correlation_matrix",
    "get_high_correlation_pairs",
    # Visualization
    "add_spec_lines",
    "build_fig",
    "apply_y_range",
    "add_spec_edge_markers",
    "add_out_of_spec_points",
    "build_cpk_heatmap",
    "build_normalized_deviation_chart",
    "build_position_comparison_chart",
    "build_imr_chart",
    "build_correlation_heatmap",
    "build_correlation_scatter",
    # Export
    "download_plot_button",
    "download_excel_button",
    "download_stats_excel",
    "download_quality_reports",
    "build_report_html",
    "generate_pdf_report",
    "download_pdf_report_button",
    # Grouping
    "format_pos",
    "assign_groups_vectorized",
]
