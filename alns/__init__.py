from core.data_structures import RvrpState
from .destroyopt import (
    create_random_customer_removal_operator,
    create_random_route_removal_operator,
    create_related_removal_operator,
    create_string_removal_operator,
    create_worst_removal_operator,
    create_sequence_removal_operator,
)
from .repairopt import (
    create_greedy_repair_operator,
    create_criticality_repair_operator,
    create_regret_repair_operator,
)
from .initial import one_for_one
