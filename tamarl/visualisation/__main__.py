"""CLI entry point for the TAMARL_Env visualization module.

Usage:
    python -m tamarl.visualisation <scenario_folder> <output_folder> [options]
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Generate GIF/MP4 visualization or open live viewer of a TAMARL_Env simulation."
    )
    parser.add_argument(
        "scenario_folder",
        help="Path to the scenario folder (must contain a *network*.xml file)."
    )
    parser.add_argument(
        "output_folder",
        help="Name of the output folder. Base path is assumed to be scenario_folder. (must contain an *events*.csv file)."
    )
    parser.add_argument(
        "--format", "-f",
        choices=["gif", "mp4"],
        default="gif",
        help="Output format (default: gif)."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second (default: 5)."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path. Defaults to <output_folder>/simulation.<format>."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI resolution (default: 150)."
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Scale factor for graph elements (nodes, agents, links). "
             "Use >1 for small scenarios, <1 for large ones (default: 1.0)."
    )
    parser.add_argument(
        "--text-scale-factor",
        type=float,
        default=1.0,
        help="Scale factor for text labels on nodes, links, and agent IDs (default: 1.0)."
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        default=False,
        help="Hide all text labels (node IDs, link info, agent IDs). "
             "Useful for large scenarios where text clutters the view."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Open an interactive live viewer instead of rendering to file. "
             "Ideal for large scenarios with many timesteps."
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=1,
        help="Initial playback speed in live mode (timesteps per tick, default: 1)."
    )
    parser.add_argument(
        "--hours",
        type=float,
        nargs=2,
        metavar=('START', 'END'),
        help="Filter events to a specific time range in hours (e.g. '--hours 7 7.25')."
    )

    args = parser.parse_args()

    output_folder = os.path.join(args.scenario_folder, args.output_folder)

    time_range = None
    if args.hours:
        time_range = (args.hours[0] * 3600, args.hours[1] * 3600)

    if args.live:
        from tamarl.visualisation.renderer import render_live
        render_live(
            scenario_folder=args.scenario_folder,
            output_folder=output_folder,
            scale_factor=args.scale_factor,
            text_scale_factor=args.text_scale_factor,
            show_labels=not args.no_labels,
            initial_speed=args.speed,
            time_range=time_range,
        )
    else:
        from tamarl.visualisation.renderer import render_animation
        render_name = args.output
        if render_name is None:
            output_path = os.path.join(output_folder, f"simulation.{args.format}")
        else:
            output_path = os.path.join(output_folder, f"{render_name}.{args.format}")

        render_animation(
            scenario_folder=args.scenario_folder,
            output_folder=output_folder,
            output_path=output_path,
            fmt=args.format,
            scale_factor=args.scale_factor,
            text_scale_factor=args.text_scale_factor,
            show_labels=not args.no_labels,
            fps=args.fps,
            dpi=args.dpi,
            time_range=time_range,
            speed=args.speed,
        )

if __name__ == "__main__":
    main()
