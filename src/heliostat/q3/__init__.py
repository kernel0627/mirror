"""第三问固定 Campo 几何结构下的分组异构规格优化。"""

from .model import (
    EXPECTED_GROUP_COUNTS,
    GROUP_COUNT,
    CampoMotherField,
    GroupDesign,
    HeterogeneousGeometryCheck,
    build_campo_mother_field,
    expand_group_design,
    validate_heterogeneous_field,
)

__all__ = [
    "EXPECTED_GROUP_COUNTS",
    "GROUP_COUNT",
    "CampoMotherField",
    "GroupDesign",
    "HeterogeneousGeometryCheck",
    "build_campo_mother_field",
    "expand_group_design",
    "validate_heterogeneous_field",
]
