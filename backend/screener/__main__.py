"""
CLI dispatcher for screener module.

Usage:
    python -m screener universe_builder --config config.yaml
    python -m screener feature_screener --universe universe.json --config config.yaml
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="screener",
        description="Universe Builder + Feature Screener for copper correlation analysis"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Universe builder
    ub_parser = subparsers.add_parser(
        "universe_builder",
        aliases=["ub"],
        help="Build universe from seed sources"
    )
    ub_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML config file"
    )
    ub_parser.add_argument(
        "--output-dir", "-o",
        default="artifacts/universes",
        help="Output directory for universe artifacts"
    )
    
    # Feature screener
    fs_parser = subparsers.add_parser(
        "feature_screener",
        aliases=["fs"],
        help="Run correlation screening on universe"
    )
    fs_parser.add_argument(
        "--universe", "-u",
        required=True,
        help="Path to universe.json file"
    )
    fs_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML config file"
    )
    fs_parser.add_argument(
        "--overrides",
        help="Path to overrides.json file (optional)"
    )
    fs_parser.add_argument(
        "--output-dir", "-o",
        default="artifacts/runs",
        help="Output directory for screening results"
    )
    
    args = parser.parse_args()
    
    if args.command in ("universe_builder", "ub"):
        from screener.universe_builder import main as ub_main
        ub_main(config_path=args.config, output_dir=args.output_dir)
    elif args.command in ("feature_screener", "fs"):
        from screener.feature_screener import main as fs_main
        fs_main(
            universe_path=args.universe,
            config_path=args.config,
            overrides_path=args.overrides,
            output_dir=args.output_dir
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
