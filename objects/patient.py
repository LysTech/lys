from __future__ import annotations
from dataclasses import dataclass

from lys.objects.mesh import Mesh, load_unMNI_mesh
from lys.objects.segmentation import load_charm_segmentation
from lys.objects.atlas import Atlas


@dataclass(frozen=True)
class Patient:
    """Represents a patient with their name, segmentation, and mesh.

    This class is a dataclass, and it is frozen, so its instances are immutable.
    It is recommended to create Patient instances using the `from_name` classmethod.

    Attributes:
        name: The patient's identifier string (e.g., "P03").
        segmentation: An Atlas object representing the patient's brain segmentation.
        mesh: A Mesh object representing the patient's brain surface mesh in native space.
    """
    name: str
    segmentation: Atlas
    mesh: Mesh

    @classmethod
    def from_name(cls, name: str) -> Patient:
        """Constructs a Patient object from a patient identifier string.

        This factory method handles the loading of the segmentation and mesh data
        for the specified patient.

        Args:
            name: The patient's identifier string (e.g., "P03").

        Returns:
            A new Patient instance.
        """
        segmentation = load_charm_segmentation(name, show=False)
        mesh = load_unMNI_mesh(name, segmentation)
        return cls(name, segmentation, mesh)
