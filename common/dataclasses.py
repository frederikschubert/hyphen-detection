from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Sample:
    image_path: str
    positive_points: List[Tuple[float, float]] = field(default_factory=list)
    negative_points: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def mineral_type(self):
        if "Biotit" in self.image_path:
            return "Biotit"
        else:
            return "Muskovit"

    @property
    def region(self):
        if "Asendorf" in self.image_path:
            return "Asendorf"
        elif "CHILE" in self.image_path:
            return "Chile"
        else:
            return "Israel"