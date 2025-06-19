


if __name__ == "__main__":

    from lys.objects.mesh import from_mat
    from lys.objects.segmentation import load_charm_segmentation
    from lys.visualization.plot3d import VTKScene

    mesh_file = "../Geometric-Eigenmodes/data/P03/anat/meshes/P03_EIGMOD_MPR_IIHC_MNI_WM_LH_edited_again_RECOSM_D32k.mat"
    mesh = from_mat(mesh_file, show=False)
    seg = load_charm_segmentation("P03", show=False)

    scene = VTKScene()
    scene.add(mesh)
    scene.add(seg)
    scene.show()