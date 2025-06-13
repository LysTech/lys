from lys.objects import Mesh
import numpy as np

def test_loading_mesh_from_mat_file():
    pass

def test_downsampling_a_mesh():
    """ Creates a test mesh and downsample it to 4 vertices. 
    
        Checks that:
        - the downsampled mesh is a new Mesh instance, 
        - has the correct number of vertices, 
        - the set of vertices is a subset of the original mesh vertices,
        - and that the faces are reasonable and 0-indexed.
    """
    vertices = np.array([
        [0, 0, 0],  
        [1, 0, 0],  
        [0, 1, 0],  
        [1, 1, 0],  
        [0, 0, 1],  
        [1, 0, 1],  
    ])
    faces = np.array([
        [0, 1, 2],  
        [1, 3, 2],  
        [0, 2, 4],  
        [1, 5, 3],  
    ])
    
    original_mesh = Mesh(vertices, faces)
    downsampled_mesh = original_mesh.downsample(n_vertices=4)
    
    # Check that we got a new Mesh instance
    assert isinstance(downsampled_mesh, Mesh)
    assert downsampled_mesh is not original_mesh
    
    # Check that the downsampled mesh has the correct number of vertices
    assert len(downsampled_mesh.vertices) == 4
    
    # Check that vertices are a subset of original vertices
    for vertex in downsampled_mesh.vertices:
        # Check if this vertex exists in the original mesh
        vertex_found = any(np.allclose(vertex, orig_vertex) for orig_vertex in original_mesh.vertices)
        assert vertex_found, f"Vertex {vertex} not found in original mesh"
    
    # Check that the faces are still 0-indexed
    min_face_idx = min([min(f) for f in downsampled_mesh.faces])
    assert min_face_idx == 0
    
    # Check that all face indices are valid (less than number of vertices)
    max_face_idx = max([max(f) for f in downsampled_mesh.faces])
    assert max_face_idx < len(downsampled_mesh.vertices)
    
    # Check that faces are reasonable
    for face in downsampled_mesh.faces:
        # Check that each face is a triangle (3 vertices)
        assert len(face) == 3, f"Face {face} is not a triangle"
        
        # Check that face has no duplicate vertices (no degenerate triangles)
        assert len(set(face)) == 3, f"Face {face} has duplicate vertices"
        
        # Check that all indices in face are valid
        for idx in face:
            assert 0 <= idx < len(downsampled_mesh.vertices), f"Face index {idx} is out of bounds"
    
    # Check that the vertices are still 3D points
    assert downsampled_mesh.vertices.shape[1] == 3