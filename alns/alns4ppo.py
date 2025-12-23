# file: alns/alns4ppo.py ver 3: sửa cách hiểu accept từ agent 
# trước kia: accept ==0 -> skip luôn buóc iterate và giữ nguyên lời giải cũ
# sửa lại: vẫn tính đầy đủ, cho ra candidate solution. nếu candidate > current và accept_mode == 0 thì giữ nguyên. 
# tuy nhiên như v thì không cần nạp prev vào làm gì cả

from typing import Dict, List, Optional, Protocol, Tuple, Any
import numpy.random as rnd
from core.data_structures import RvrpState

# Protocol cho Operator
class _OperatorType(Protocol):
    __name__: str
    def __call__(self, state: RvrpState, rng: rnd.Generator, **kwargs) -> RvrpState: ...

class ALNS4PPO:
    def __init__(self, rng: rnd.Generator = rnd.default_rng()):
        self._rng = rng
        self._d_ops: Dict[str, _OperatorType] = {}
        self._r_ops: Dict[str, _OperatorType] = {}

    @property
    def destroy_operators(self) -> List[Tuple[str, _OperatorType]]:
        return list(self._d_ops.items())

    @property
    def repair_operators(self) -> List[Tuple[str, _OperatorType]]:
        return list(self._r_ops.items())

    def add_destroy_operator(self, op: _OperatorType, name: Optional[str] = None):
        op_name = name if name else getattr(op, "__name__", "unknown_destroy_op")
        self._d_ops[op_name] = op

    def add_repair_operator(self, op: _OperatorType, name: Optional[str] = None):
        op_name = name if name else getattr(op, "__name__", "unknown_repair_op")
        self._r_ops[op_name] = op

    def iterate(
        self, 
        current_solution: RvrpState, 
        pre_solution: RvrpState, # Tham số này giữ lại để tương thích interface, nhưng logic chính nằm ở current
        d_idx: int, 
        r_idx: int, 
        accept_mode: int, # 0: Greedy Mode, 1: Exploration Mode
        **kwargs
    ) -> Tuple[RvrpState, RvrpState]:        
        # 1. Select Operators
        d_ops_list = self.destroy_operators
        r_ops_list = self.repair_operators
        
        # Safety check
        if d_idx >= len(d_ops_list) or r_idx >= len(r_ops_list):
            # Fallback an toàn: Không làm gì cả
            return current_solution, current_solution

        _, d_operator = d_ops_list[d_idx]
        _, r_operator = r_ops_list[r_idx]

        # 2. ALWAYS EXECUTE DESTROY & REPAIR (Tạo Candidate)
        # Các operator đã implement logic .copy() bên trong nên an toàn
        destroyed_state = d_operator(current_solution, self._rng, **kwargs)
        candidate_solution = r_operator(destroyed_state, self._rng, **kwargs)

        # 3. EVALUATE & APPLY ACCEPTANCE CRITERIA (Logic bài báo)
        curr_obj = current_solution.objective()
        cand_obj = candidate_solution.objective()
        
        # Case A: Improvement (Tốt hơn thì luôn lấy)
        if cand_obj < curr_obj:
            # New becomes Current
            return current_solution, candidate_solution
            
        # Case B: Worsening (Tệ hơn)
        else:
            if accept_mode == 1: 
                # Exploration Mode: Chấp nhận lời giải tồi
                return current_solution, candidate_solution
            else:
                # Greedy Mode: Từ chối, giữ nguyên lời giải cũ
                # Trả về (old, old) -> Delta cost = 0 -> Reward phạt nhẹ hoặc 0 tùy config
                return current_solution, current_solution

    def reset_opt(self):
        self._d_ops.clear()
        self._r_ops.clear()
        self._rng = rnd.default_rng()