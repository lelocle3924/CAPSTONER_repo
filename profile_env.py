# file: profile_env.py
import time
import cProfile
import pstats
from ppo.rvrpenv import RVRPEnvironment
from config import PathConfig

def profile_run():
    print(">>> Setting up Environment for Profiling...")
    env = RVRPEnvironment(
        order_csv_path=PathConfig.ORDER_PATH,
        truck_csv_path=PathConfig.TRUCK_PATH,
        is_test_mode=True
    )
    
    env.reset()
    
    print(">>> Starting 10 Steps Loop...")
    start_time = time.time()
    
    # Random Actions Loop
    for _ in range(10):
        # Action space: [d_op, r_op, accept, stop]
        # Random sample valid actions
        action = env.action_space.sample()
        
        # Force valid action if needed (simple hack)
        mask = env.valid_action_mask()
        # (For profiling we assume sample is mostly valid or Env handles invalid gracefully)
        
        obs, reward, term, trunc, info = env.step(action)
        if term:
            env.reset()
            
    end_time = time.time()
    print(f">>> Done. Total Time: {end_time - start_time:.4f}s")

if __name__ == "__main__":
    # Run with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    
    profile_run()
    
    profiler.disable()
    
    print("\n" + "="*40)
    print(" TOP 20 TIME CONSUMING FUNCTIONS")
    print("="*40)
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)
    
    # Export for SnakeViz (Optional visualization tool)
    stats.dump_stats("profile_results.prof")
    print("\n>>> Profile saved to 'profile_results.prof'. Use 'snakeviz profile_results.prof' to view.")