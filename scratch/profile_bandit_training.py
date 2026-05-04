import cProfile
import pstats
import io
from tamarl.rl.train_bandit import train, load_config
import os

def profile_bandit():
    config_path = "tamarl/data/configs/train_config_1.json"
    kwargs = load_config(config_path)
    
    # Force 1 episode for fast profiling
    kwargs["n_episodes"] = 1
    # Ensure relative gap is on to see its impact
    kwargs["relative_gap"] = True
    
    pr = cProfile.Profile()
    pr.enable()
    
    train(**kwargs)
    
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(50)  # Top 50 functions
    
    # Write to scratch directory in workspace
    profile_path = "scratch/bandit_profile.txt"
    with open(profile_path, "w") as f:
        f.write(s.getvalue())
    
    print(f"\nProfiling complete. Results saved to {profile_path}")

if __name__ == "__main__":
    profile_bandit()
