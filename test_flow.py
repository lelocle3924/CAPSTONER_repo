# file: test_flow.py -> for testing up until destroy, chưa đụng đến PPOState, vì nó có dính đến ALNS4PPO
import numpy as np
import pandas as pd
from core.real_data_loader import RealDataLoader
from core.data_structures import RvrpState
import alns.vrp4ppo as vrp
from alns.vrp4ppo.destroyopt import create_random_customer_removal_operator

def test_pipeline():
    print(">>> STEP 1: LOAD DATA")
    loader = RealDataLoader()
    # Giả lập đường dẫn file (Em dùng file mẫu thầy đưa)
    problem_data = loader.load_day_data(
        order_csv_path="inputs/CleanData/Split_TransportOrder_1day.csv",
        truck_csv_path="inputs/MasterData/TruckMaster.csv"
    )
    print(f"Nodes: {problem_data.num_nodes} (Index 0 is Depot)")
    
    print("\n>>> STEP 2: GENERATE INITIAL SOLUTION (One-for-One)")
    init_sol = vrp.initial.one_for_one(problem_data)
    
    # Validation Requirement 1: Tất cả khách hàng được serve
    total_customers = problem_data.num_nodes - 1
    served_customers = sum(len(r.node_sequence) for r in init_sol.routes)
    
    print(f"Total Routes Created: {len(init_sol.routes)}")
    print(f"Total Customers Served: {served_customers}/{total_customers}")
    print(f"Unassigned: {init_sol.unassigned}")
    print(f"Total cost: {init_sol.objective():,.2f}")
    
    if len(init_sol.unassigned) == 0:
        print("✅ SUCCESS: All customers are served!")
    else:
        print("❌ WARNING: Some customers are unassigned. Check constraints.")

    # Validation Requirement 2: Destroy Operator (Random Removal)
    print("\n>>> STEP 3: RUN DESTROY OPERATOR")
    destroy_op = create_random_customer_removal_operator(problem_data, ratio=0.2) # Xóa 20%
    
    # Giả lập RNG
    rng = np.random.default_rng(42)
    
    try:
        destroyed_sol = destroy_op(init_sol, rng)
        
        # Kiểm tra tính toàn vẹn của RvrpState trả về
        num_removed = len(destroyed_sol.unassigned)
        num_remaining = sum(len(r.node_sequence) for r in destroyed_sol.routes)
        
        print(f"Destroyed Solution Status:")
        print(f"  - Routes remaining: {len(destroyed_sol.routes)}")
        print(f"  - Customers in Unassigned: {num_removed}")
        print(f"  - Customers remaining in routes: {num_remaining}")
        
        # Check sum
        if num_removed + num_remaining == total_customers:
             print("✅ SUCCESS: Customer count matches (Conservation of Mass).")
        else:
             print("❌ ERROR: Customer count mismatch!")
             
        # Check object type
        if isinstance(destroyed_sol, RvrpState):
            print("✅ SUCCESS: Returned object is valid RvrpState.")
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR during Destroy: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()