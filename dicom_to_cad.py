import os, sys, json, math, argparse, tempfile, warnings
from pathlib import Path

import numpy as np

# Prefer SimpleITK for robust series IO; fallback to pydicom
try:
    import SimpleITK as sitk
    HAS_SITK = True
except Exception:
    HAS_SITK = False
    import pydicom

from skimage import measure, filters, morphology
import trimesh
from trimesh.smoothing import filter_laplacian

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# OpenCascade for STEP/IGES (OCP preferred; fallback to pythonocc-core)
try:
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.IGESControl import IGESControl_Writer
    from OCP.StlAPI import StlAPI_Reader
    from OCP.TopoDS import TopoDS_Shape
    from OCP.IFSelect import IFSelect_RetDone
    HAS_OCC = True
except Exception:
    try:
        from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
        from OCC.Core.IGESControl import IGESControl_Writer
        from OCC.Core.StlAPI import StlAPI_Reader
        from OCC.Core.TopoDS import TopoDS_Shape
        from OCC.Core.IFSelect import IFSelect_RetDone
        HAS_OCC = True
    except Exception:
        HAS_OCC = False
        warnings.warn("OpenCascade not found; STEP/IGES export disabled.", RuntimeWarning)

def load_series_sitk(dicom_dir: Path):
   
    return vol.astype(np.int16), voxel, {"origin": origin, "direction": direction}

def load_series_pydicom(dicom_dir: Path):
    
    return vol, voxel, meta

def load_dicom_series(dicom_dir: str):
    dicom_dir = Path(dicom_dir)
    if HAS_SITK:
        return load_series_sitk(dicom_dir)
    return load_series_pydicom(dicom_dir)

def segment_volume(vol: np.ndarray, use_otsu: bool, threshold: float):
    
    return seg, float(thr)

def largest_component(mask: np.ndarray):
    
    return labels == lab

def marching_cubes(mask: np.ndarray, voxel_spacing):
    # skimage expects (z,y,x) mask; spacing order the same
    verts, faces, norms, _ = measure.marching_cubes(
        mask.astype(np.uint8), level=0.5, spacing=voxel_spacing
    )
    return verts, faces

def postprocess_mesh(verts, faces, do_smooth=True, decimate_ratio=None):
   
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh = mesh.as_triangles()
    return mesh

def compute_metrics(mesh: trimesh.Trimesh):
   
    return {
        "surface_area_mm2": area,
        "volume_mm3": vol,
        "principal_length_mm": L,
        "principal_width_mm": W,
        "principal_height_mm": H,
        "equivalent_sphere_diameter_mm": eq_diam,
        "aabb_min_mm": aabb_min,
        "aabb_max_mm": aabb_max,
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
    }

def export_stl(mesh: trimesh.Trimesh, out_path: Path):
    mesh.export(out_path)

def export_step_iges_from_stl(stl_path: Path, step_path: Path = None, iges_path: Path = None):
    if not HAS_OCC:
        warnings.warn("OpenCascade not available; skipping STEP/IGES export.")
        return
    shape = TopoDS_Shape()
    reader = StlAPI_Reader()
    # StlAPI_Reader.Read requires C-style strings
    if not reader.Read(shape, str(stl_path)):
        warnings.warn("Failed to read STL into OpenCascade; STEP/IGES not written.")
        return
    if step_path:
        step_writer = STEPControl_Writer()
        status = step_writer.Transfer(shape, STEPControl_AsIs)
        if status == 1:
            status = step_writer.Write(str(step_path))
        if status != IFSelect_RetDone:
            warnings.warn("STEP export failed.")
    if iges_path:
        iges_writer = IGESControl_Writer()
        iges_writer.AddShape(shape)
        if iges_writer.Write(str(iges_path)) != IFSelect_RetDone:
            warnings.warn("IGES export failed.")

def save_previews(vol, seg, mesh, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    mid = vol.shape[0] // 2

    # Slice preview
    plt.figure(figsize=(6,6))
    plt.imshow(vol[mid,:,:], cmap="gray")
    plt.axis("off")
    plt.title("Mid Slice")
    plt.tight_layout()
    plt.savefig(out_dir / "preview_slice.png", dpi=150)
    plt.close()

    # Segmentation preview
    plt.figure(figsize=(6,6))
    plt.imshow(seg[mid,:,:], cmap="gray")
    plt.axis("off")
    plt.title("Segmentation (mid slice)")
    plt.tight_layout()
    plt.savefig(out_dir / "preview_seg.png", dpi=150)
    plt.close()

    # 3D preview (pyvista)
    try:
        import pyvista as pv
        pv.start_xvfb()
        p = pv.Plotter(off_screen=True, window_size=(800,800))
        grid = pv.PolyData(mesh.vertices, np.hstack([np.full((len(mesh.faces),1),3), mesh.faces]).ravel())
        p.add_mesh(grid, show_edges=False, opacity=1.0)
        p.show(screenshot=str(out_dir / "preview_mesh.png"))
        p.close()
    except Exception:
        # Fallback: simple matplotlib 3D scatter (not ideal)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, projection='3d')
        tri = Poly3DCollection(mesh.triangles, alpha=1.0)
        ax.add_collection3d(tri)
        scale = mesh.vertices.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(out_dir / "preview_mesh.png", dpi=150)
        plt.close()

def main():
    ap = argparse.ArgumentParser(description="DICOM → STL/STEP/IGES with metrics")
    ap.add_argument("--dicom_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--threshold", type=float, default=300.0, help="Intensity/HU threshold (ignored if --otsu true)")
    ap.add_argument("--otsu", type=str, default="false", help="true/false")
    ap.add_argument("--no_smooth", action="store_true", help="disable Laplacian smoothing")
    ap.add_argument("--decimate", type=float, default=None, help="0–1 fraction of faces to keep (e.g., 0.5)")
    ap.add_argument("--export_step", action="store_true")
    ap.add_argument("--export_iges", action="store_true")
    args = ap.parse_args()

    dicom_dir = Path(args.dicom_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DICOM series...")
    vol, voxel_spacing, meta = load_dicom_series(dicom_dir)
    print(f"Volume shape (z,y,x): {vol.shape}; voxel spacing (z,y,x) mm: {voxel_spacing}")

    use_otsu = str(args.otsu).lower() in ("true","1","yes","y","t")
    print(f"Segmentation: {'Otsu' if use_otsu else f'Threshold {args.threshold}'}")
    seg, used_thr = segment_volume(vol, use_otsu, args.threshold)
    seg = largest_component(seg)

    print("Running Marching Cubes...")
    verts, faces = marching_cubes(seg, voxel_spacing)
    print(f"Raw mesh: {len(verts)} verts, {len(faces)} faces")

    print("Post-processing mesh...")
    mesh = postprocess_mesh(verts, faces, do_smooth=not args.no_smooth, decimate_ratio=args.decimate)

    print("Computing metrics...")
    metrics = compute_metrics(mesh)
    metrics.update({
        "threshold_used": used_thr,
        "voxel_spacing_mm": {"z": voxel_spacing[0], "y": voxel_spacing[1], "x": voxel_spacing[2]},
    })
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Exporting STL...")
    stl_path = out_dir / "model.stl"
    export_stl(mesh, stl_path)

    if args.export_step or args.export_iges:
        if not HAS_OCC:
            warnings.warn("STEP/IGES requested but OpenCascade not available.")
        else:
            print("Exporting STEP/IGES (faceted B-Rep)...")
            step_path = out_dir / "model.step" if args.export_step else None
            iges_path = out_dir / "model.igs"  if args.export_iges else None
            export_step_iges_from_stl(stl_path, step_path, iges_path)

    print("Saving previews...")
    save_previews(vol, seg, mesh, out_dir)

    print("Done.")
    print(f"Outputs in: {out_dir}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
