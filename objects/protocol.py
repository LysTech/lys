from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict, ClassVar, Set
from pathlib import Path
import os
import numpy as np
from lys.utils.paths import lys_data_dir, check_file_exists

#TODO: check how slow it would be to just read the file every time, if too slow: think about caching but... seems unlikely?

@dataclass(frozen=True)
class Protocol:
    """
    Represents a neuro experiment protocol as an ordered list of (t_start, t_end, label) tuples.

    Respects OCP: we can add a from_whatever constructor for our future formats.
    """
    intervals: List[Tuple[float, float, str]] = field(default_factory=list)

    @property
    def tasks(self) -> Set[str]:
        """
        Returns the set of unique task labels in the protocol.
        """
        return set([interval[2] for interval in self.intervals])

    @classmethod
    def from_dict(cls, protocol_dict: Dict[str, List[Tuple[float, float]]]) -> 'Protocol':
        """
        Construct a Protocol from a dictionary mapping labels to lists of (t_start, t_end) tuples.

        Example:
            {
                "MT": [(10.0, 20.0), (59.7, 64.7)],
                "SN": [(15.0, 25.0)]
            }

        The resulting intervals are sorted by t_start.
        """
        intervals = [
            (start, end, label)
            for label, intervals_list in protocol_dict.items()
            for (start, end) in intervals_list
        ]
        intervals.sort(key=lambda x: x[0])
        return cls(intervals=intervals)

    @classmethod
    def from_prt(cls, prt_path: Path) -> 'Protocol':
        """
        Construct a Protocol from a .prt file path.
        The .prt file format is expected to follow the BrainVoyager protocol format.
        """
        protocol_dict = cls._parse_prt(prt_path)
        return cls.from_dict(protocol_dict)

    @staticmethod
    def _parse_prt(prt_path: Path) -> Dict[str, List[Tuple[float, float]]]:
        """
        Parse a .prt file and return a dictionary mapping condition names to lists of (t_start, t_end) tuples.
        """
        with open(prt_path, 'r') as f:
            lines = [ln.rstrip() for ln in f if ln.strip()]

        # Skip ahead to the first block
        i = 0
        while i < len(lines) and not lines[i].startswith("NrOfConditions"):
            i += 1
        i += 1  # skip the NrOfConditions line

        protocol = {}
        while i < len(lines):
            cond_name = lines[i].strip()
            i += 1
            try:
                n_intervals = int(lines[i].strip())
            except ValueError:
                raise ValueError(f"Expected interval count after condition name '{cond_name}' at line {i+1}")
            i += 1

            intervals = []
            for _ in range(n_intervals):
                parts = lines[i].split()
                if len(parts) < 2:
                    raise ValueError(f"Malformed interval line at {i+1}: '{lines[i]}'")
                start, end = map(float, parts[:2])
                intervals.append((start, end))
                i += 1

            protocol[cond_name] = intervals

            # skip the Color line if present
            if i < len(lines) and lines[i].startswith("Color"):
                i += 1

        return protocol

    @classmethod
    def from_session_path(cls, session_path: Path) -> 'Protocol':
        """
        Search for a .prt file in the given session_path directory and construct a Protocol from it.
        Raises FileNotFoundError if no .prt file is found.
        """
        prt_files = list(session_path.glob('*.prt'))
        if not prt_files:
            raise FileNotFoundError(f"No .prt file found in {session_path}")
        if len(prt_files) > 1:
            raise ValueError(f"Multiple .prt files found in {session_path}: {prt_files}")
        return cls.from_prt(prt_files[0])


def create_protocol(patient: str, experiment: str, session: str) -> Protocol:
    """
    Loads a Protocol object for a given patient, experiment, and session.
    Finds the protocol.prt file, checks it exists, and loads it.
    """
    path = _protocol_path(patient, experiment, session)
    check_file_exists(path)
    return Protocol.from_prt(Path(path))


def _protocol_path(patient: str, experiment: str, session: str) -> str:
    """
    Constructs the path to the protocol.prt file for a given patient, experiment, and session.
    """
    root = lys_data_dir()
    return os.path.join(root, patient, 'nirs', experiment, session, 'protocol.prt')

