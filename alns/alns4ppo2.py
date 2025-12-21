# file: alns/alns4ppo.py

from typing import Dict, List, Optional, Protocol, Tuple, Any
import numpy.random as rnd
from core.data_structures import RvrpState 

# Định nghĩa Protocol cho Operator để type checking rõ ràng hơn
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

    def iterate(self, cur_solution: RvrpState, pre_solution: RvrpState, d_idx: int, r_idx: int, accept: int, **kwargs) -> Tuple[RvrpState, RvrpState]:
        """
        Thực thi 1 vòng lặp ALNS dựa trên action của PPO Agent.
        
        Args:
            cur_solution: Giải pháp hiện tại (xuất phát điểm của vòng lặp này)
            pre_solution: Giải pháp của vòng lặp trước (để revert nếu reject)
            d_idx: Index của Destroy Operator
            r_idx: Index của Repair Operator
            accept: 1 (Chấp nhận thay đổi/thử nghiệm), 0 (Từ chối/Giữ nguyên)
            **kwargs: Các tham số phụ (ví dụ: data=ProblemData) truyền xuống operators
            
        Returns:
            (pre_solution_new, cur_solution_new)
        """

        if accept == 0:
            # agent bảo là, tôi sẽ không accept bất cứ cái gì anh cho ra ở iteration này, khỏi tính toán làm gì cho mệt???
            return pre_solution, pre_solution

        elif accept == 1:
            # 1. Select Operators
            d_ops_list = self.destroy_operators
            r_ops_list = self.repair_operators
            
            # Safety check index
            if d_idx >= len(d_ops_list) or r_idx >= len(r_ops_list):
                print(f"[ALNS ERROR] Index out of bounds. D:{d_idx}/{len(d_ops_list)}, R:{r_idx}/{len(r_ops_list)}")
                return pre_solution, cur_solution

            d_name, d_operator = d_ops_list[d_idx]
            r_name, r_operator = r_ops_list[r_idx]

            # 2. APPLY DESTROY
            # print(f"    -> Applying Destroy: {d_name}...") 
            destroyed_state = d_operator(cur_solution, self._rng, **kwargs)
            
            # 3. APPLY REPAIR
            # print(f"    -> Applying Repair: {r_name}...")
            repaired_state = r_operator(destroyed_state, self._rng, **kwargs)
            
            # 4. Update References
            # Repaired state trở thành Current state mới
            # Current state cũ trở thành Pre state (để lưu lịch sử cho vòng sau so sánh)
            return cur_solution, repaired_state

        return pre_solution, cur_solution

    def reset_opt(self):
        self._d_ops.clear()
        self._r_ops.clear()
        self._rng = rnd.default_rng()