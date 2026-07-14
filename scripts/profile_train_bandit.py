import argparse
import cProfile
import inspect
import os
import pstats
import sys

from tamarl.rl.train_bandit import _CLI_TO_KWARGS, _build_parser, load_config, train


def main():
    parser = _build_parser()
    parser.description = "Profile the One-Shot Bandit DTA training runner"
    parser.add_argument("--top-n", type=int, default=50, help="Number of top functions to display")
    parser.add_argument(
        "--sort-by",
        default="cumulative",
        choices=["cumulative", "tottime", "ncalls"],
        help="Sort profiling results by (cumulative, tottime, ncalls)",
    )
    parser.add_argument(
        "--output-file", default="profile.stats", help="File to save profiling statistics"
    )

    args, unknown = parser.parse_known_args()

    # 1. Start with train() defaults
    sig = inspect.signature(train)
    kwargs = {
        k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty
    }

    # 2. Override with JSON config if provided
    if args.config:
        config_path = args.config
        if not os.path.isfile(config_path):
            parser.error(f"Config file not found: {config_path}")
        json_kwargs = load_config(config_path)
        kwargs.update(json_kwargs)
        print(f"  [Config] Loaded from: {config_path}")

    # 3. Override with CLI arguments (only those explicitly set)
    args_dict = vars(args)
    for cli_name, kwarg_name in _CLI_TO_KWARGS.items():
        cli_val = args_dict.get(cli_name)
        if cli_val is not None:
            kwargs[kwarg_name] = cli_val

    # 4. Ensure scenario_path is set
    if "scenario_path" not in kwargs or kwargs["scenario_path"] is None:
        parser.error("--scenario is required (via CLI or config file)")

    print(f"\n{'=' * 65}")
    print("  PROFILING START")
    print(f"{'=' * 65}")
    print(f"  Scenario:      {kwargs.get('scenario_path')}")
    print(f"  Population:    {kwargs.get('population_filter')}")
    print(f"  Episodes:      {kwargs.get('n_episodes')}")
    print(f"  Sorting by:    {args.sort_by}")
    print(f"{'=' * 65}\n")

    # 5. Profiling
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        train(**kwargs)
        profiler.disable()
    except Exception as e:
        profiler.disable()
        print(f"\n[!] Profiling interrupted by error: {e}")
        import traceback

        traceback.print_exc()

    # 6. Report
    stats = pstats.Stats(profiler).sort_stats(args.sort_by)

    print(f"\n{'=' * 65}")
    print(f"  PROFILING RESULTS (Top {args.top_n} by {args.sort_by})")
    print(f"{'=' * 65}")

    # Filter out some noise if necessary, or just show everything
    stats.print_stats(args.top_n)

    # Save to file
    stats.dump_stats(args.output_file)
    print(f"\n[+] Full profiling stats saved to: {args.output_file}")
    print(f"    You can visualize them using 'snakeviz {args.output_file}' if installed.")


if __name__ == "__main__":
    main()
