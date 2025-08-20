# Dicom-2D-Image-to-3D-Model-Generation
Convert large DICOM series (~30,000 slices) into watertight 3D meshes suitable for printing, and export to STL, STEP, IGES. Includes surface area, volume, principal dimensions, and equivalent diameters.

#############################################################################
Loads and sorts huge DICOM series reliably (uses SimpleITK or pydicom fallback)

Builds a 3D volume with correct voxel spacing

Segments (threshold or Otsu) and extracts a clean watertight mesh (marching cubes)

Post-processes (largest component, smoothing, optional decimation)

Exports STL natively; and STEP/IGES via OpenCascade (faceted B-Rep)

Computes area, volume, principal length/width/height, equivalent diameter, and more

Saves preview screenshots (slice, segmentation, mesh)

Designed for big studies (30k slices) with memory tips and CLI flags
###################################################################################
✨ Features

Robust DICOM series loading (sort by ImagePositionPatient or InstanceNumber)

HU rescale (CT) and spacing handling

Threshold / Otsu segmentation

Marching Cubes mesh extraction

Largest-component filtering, smoothing, optional decimation

Exports: STL (mesh), STEP/IGES (faceted B-Rep via OpenCascade)

Metrics: surface area, volume, principal L/W/H, bounding box, equivalent diameter

Saves preview images: slice, segmentation, 3D mesh

🔧 Install
# (recommended) new environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# If OCP fails, try: pip install pythonocc-core
pip install -r requirements.txt
▶️ Run
python -m src.dicom_to_cad \
  --dicom_dir ./examples/sample_series \
  --out_dir   ./examples/outputs \
  --threshold 300 \
  --otsu      false \
  --decimate  0.5
Key args

--threshold: scalar (HU or intensity). For metals/bone use 250–700; for plastics adjust empirically.

--otsu: set true to ignore --threshold and use Otsu.

--decimate: 0–1 (fraction of faces to keep). 0.5 keeps ~50%.

--no_smooth: skip Laplacian smoothing (faster).

--export_step, --export_iges: toggle CAD exports.
📁 Outputs (in --out_dir)

model.stl, optionally model.step, model.igs

metrics.json

preview_slice.png, preview_seg.png, preview_mesh.png
🖨️ Print flow

Import model.stl into Cura/PrusaSlicer → scale/supports/infill → slice → print.
⚠️ Notes on STEP/IGES

Exported STEP/IGES are faceted (triangulated) B-Reps reconstructed from the mesh; that’s expected when the source is voxel CT and not a native parametric CAD. For true NURBS/analytic surfaces, a dedicated reverse-engineering pipeline is required.
