# Preprocess a Neural 3D Video (DyNeRF) scene for TAD-GS training:
#   1. unzip the scene archive (if not already extracted)
#   2. extract frames from every camXX.mp4
#   3. build known-pose COLMAP inputs and run COLMAP (sparse triangulation,
#      optionally dense MVS + point cloud downsampling) on every Nth frame
#   4. give every other timestamp the same colmap_<t>/images + manual structure
#   5. remove the temporary camXX/ frame folders
#
# Adapted from SpacetimeGaussians (MIT License, Copyright (c) 2023 OPPO):
# https://github.com/oppo-us-research/SpacetimeGaussians/blob/main/script/pre_n3d.py
# https://github.com/oppo-us-research/SpacetimeGaussians/blob/main/script/utils_pre.py
# with references to Swift4D: https://github.com/WuJH2001/Swift4d
#
# Usage (see preprocess_n3d.sh for the batch driver):
#   python utils/preprocessing/neural_3d_video/pre_n3d.py --videopath data/neural_3d_video/cook_spinach \
#       --startframe 0 --endframe 300 --frame_interval 20 --downscale 2 --pointcloud sparse

import argparse
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import tqdm

from colmap_database import COLMAPDatabase
from colmap_runner import run_colmap_dense, run_colmap_sparse, cleanup_dense_workspace
from pose_utils import posetow2c_matrcs, rotmat2qvec


def unzip_scene(videopath: Path):
    """Extract <scene>.zip into videopath if the scene folder isn't populated yet.

    The scene archives live at the dataset root (<root>/<scene>.zip), while the
    extracted scenes live under <root>/scenes/<scene>. The archive is therefore
    looked up both next to the scene folder (flat layout) and one level above
    the scenes/ folder (scenes/ layout).
    """
    if videopath.exists() and sorted(videopath.glob("cam*.mp4")):
        print(f"Scene already extracted: {videopath}")
        return

    scene = videopath.name
    candidates = [videopath.with_suffix(".zip"),                 # <root>/<scene>.zip (flat)
                  videopath.parent.parent / f"{scene}.zip"]      # <root>/<scene>.zip above scenes/
    zippath = next((z for z in candidates if z.exists()), None)
    if zippath is None:
        raise FileNotFoundError(f"No cam*.mp4 in {videopath} and no archive found at "
                                + " or ".join(str(c) for c in candidates))

    print(f"Unzipping {zippath} -> {videopath.parent} ...")
    videopath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zippath) as zf:
        # Archives contain a top-level <scene>/ folder, so extract into the parent.
        zf.extractall(videopath.parent)
    assert sorted(videopath.glob("cam*.mp4")), f"No cam*.mp4 found in {videopath} after unzip"


def sequential_camera_names(videopath: Path):
    """Map each present cam*.mp4 to a gap-free sequential name (cam00, cam01, ...).

    Some N3D scenes are missing individual cameras (e.g. coffee_martini ships
    18 of 21, cut_roasted_beef 20 of 21). poses_bounds.npy only has rows for
    the cameras that are actually present, in the same sorted order as
    cam*.mp4, so positional matching against sorted glob order is always
    correct. What is NOT safe to keep is the original camera number as the
    output filename, since downstream code expects a dense 0..N-1 range with
    no gaps. This renumbers by position while leaving pose lookup untouched.
    """
    video_paths = sorted(videopath.glob("cam*.mp4"))
    return {v.stem: f"cam{i:02d}" for i, v in enumerate(video_paths)}


def frame_done(scene: Path, camname: str, seqname: str, i: int, ext="png"):
    """A frame is done if it exists in camXX/ or was already moved to colmap_<i>/images/."""
    return ((scene / camname / f"{i}.{ext}").exists()
            or (scene / f"colmap_{i}" / "images" / f"{seqname}.{ext}").exists())


def extractframes(videopath: Path, seqname: str, startframe=0, endframe=300, downscale=1, ext="png"):
    """Extract frames [startframe, endframe) of one video into <scene>/<camXX>/<i>.png.

    The temporary per-video folder keeps the video's own name (camname); only
    the completion check against colmap_<i>/images/ uses the renumbered
    seqname, since that's the name frames get distributed under.
    """
    scene = videopath.parent
    camname = videopath.stem
    output_dir = scene / camname

    if all(frame_done(scene, camname, seqname, i, ext) for i in range(startframe, endframe)):
        print(f"Already extracted all the frames of {camname}")
        return

    cam = cv2.VideoCapture(str(videopath))
    cam.set(cv2.CAP_PROP_POS_FRAMES, startframe)

    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(startframe, endframe):
        success, frame = cam.read()
        if not success:
            print(f"Error reading frame {i} of {videopath.name}")
            break

        if frame_done(scene, camname, seqname, i, ext):
            continue

        if downscale > 1:
            new_width = int(frame.shape[1] / downscale)
            new_height = int(frame.shape[0] / downscale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(output_dir / f"{i}.{ext}"), frame)

    cam.release()


def preparecolmapdynerf(folder: Path, cameras, offset=0):
    """Stage frame <offset> of every camera into colmap_<offset>/input/.

    Frames normally come from the temporary camXX/ folders (symlinked). If a
    frame was already distributed to colmap_<offset>/images/ by a previous run,
    it is moved back into input/ instead, since COLMAP's undistorter will
    regenerate images/. Source frames live under the camera's original video
    name; staged/distributed copies use the renumbered sequential name.
    """
    projectfolder = folder / f"colmap_{offset}"
    savedir = projectfolder / "input"
    savedir.mkdir(exist_ok=True, parents=True)

    for cam in cameras:
        origname = cam["orig_stem"]
        seqfilename = cam["filename"]
        imagepath = folder / origname / f"{offset}.png"
        distributedpath = projectfolder / "images" / seqfilename
        imagesavepath = savedir / seqfilename

        if imagesavepath.exists():
            continue

        if imagepath.exists():
            imagesavepath.symlink_to(imagepath.resolve())
        elif distributedpath.exists():
            shutil.move(str(distributedpath), str(imagesavepath))
        else:
            raise FileNotFoundError(f"Missing frame for {origname} at offset {offset}: "
                                    f"neither {imagepath} nor {distributedpath} exists")


def load_cameras(path: Path, seq_names: dict, downscale=1):
    """Read poses_bounds.npy and return per-camera COLMAP intrinsics/extrinsics.

    poses_bounds.npy rows correspond positionally to sorted cam*.mp4 (one row
    per camera actually present), so video_paths[i] always gets the correct
    pose even when the scene is missing some camera numbers. The output
    filename uses the gap-free sequential name from seq_names, not the
    original camera number.
    """
    originnumpy = path / "poses_bounds.npy"
    video_paths = sorted(path.glob("cam*.mp4"))

    with open(originnumpy, "rb") as numpy_file:
        poses_bounds = np.load(numpy_file)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)

    llffposes = poses.copy().transpose(1, 2, 0)
    w2c_matriclist = posetow2c_matrcs(llffposes)
    assert type(w2c_matriclist) == list

    cameras = []
    for i in range(len(poses)):
        m = w2c_matriclist[i]
        colmapR = m[:3, :3]
        T = m[:3, 3]
        H, W, focal = poses[i, :, -1] / downscale
        origstem = video_paths[i].stem

        cameras.append({
            "id": i + 1,
            "orig_stem": origstem,
            "filename": f"{seq_names[origstem]}.png",
            "w": W,
            "h": H,
            "fx": focal,
            "fy": focal,
            "cx": W // 2,
            "cy": H // 2,
            "q": rotmat2qvec(colmapR),
            "t": T,
        })
    return cameras


def write_manual_model(projectfolder: Path, cameras):
    """Write the known-pose text model (cameras.txt, images.txt, points3D.txt)."""
    manualfolder = projectfolder / "manual"
    manualfolder.mkdir(exist_ok=True, parents=True)

    imagetxtlist = []
    cameratxtlist = []
    for cam in cameras:
        line = (f"{cam['id']} " + " ".join(map(str, cam["q"])) + " "
                + " ".join(map(str, cam["t"])) + f" {cam['id']} {cam['filename']}\n")
        imagetxtlist.append(line)
        imagetxtlist.append("\n")
        cameratxtlist.append(f"{cam['id']} PINHOLE {cam['w']} {cam['h']} "
                             f"{cam['fx']} {cam['fy']} {cam['cx']} {cam['cy']}\n")

    (manualfolder / "images.txt").write_text("".join(imagetxtlist))
    (manualfolder / "cameras.txt").write_text("".join(cameratxtlist))
    (manualfolder / "points3D.txt").write_text("")


def write_colmap_db(projectfolder: Path, cameras):
    """Write input.db pre-filled with the known cameras and image priors."""
    db_file = projectfolder / "input.db"
    if db_file.exists():
        db_file.unlink()

    db = COLMAPDatabase.connect(db_file)
    db.create_tables()
    for cam in cameras:
        params = np.array((cam["fx"], cam["fy"], cam["cx"], cam["cy"]))
        camera_id = db.add_camera(1, cam["w"], cam["h"], params)  # model 1 = PINHOLE
        db.add_image(cam["filename"], camera_id, prior_q=cam["q"], prior_t=cam["t"],
                     image_id=cam["id"])
        db.commit()
    db.close()


def distribute_frames(path: Path, startframe, endframe, colmap_offsets, cameras, ext="png"):
    """Give every non-COLMAP timestamp the same colmap_<t>/images + manual structure.

    COLMAP offsets already have undistorted images/ and manual/ from the COLMAP
    run; for all other timestamps the extracted frames are moved into
    colmap_<t>/images/ and the same known-pose manual/ model is written.
    Source frames live under each camera's original video name; distributed
    copies use the renumbered sequential name (see sequential_camera_names).
    """
    for offset in tqdm.tqdm(range(startframe, endframe), desc="Distribute frames"):
        projectfolder = path / f"colmap_{offset}"

        if offset not in colmap_offsets:
            imagesdir = projectfolder / "images"
            imagesdir.mkdir(exist_ok=True, parents=True)
            for cam in cameras:
                src = path / cam["orig_stem"] / f"{offset}.{ext}"
                dst = imagesdir / cam["filename"]
                if dst.exists():
                    continue
                assert src.exists(), f"Missing frame {src}"
                shutil.move(str(src), str(dst))

        if not (projectfolder / "manual" / "cameras.txt").exists():
            write_manual_model(projectfolder, cameras)


def cleanup_cam_folders(path: Path, startframe, endframe, cameras, ext="png"):
    """Remove the temporary camXX/ frame folders once every timestamp is populated."""
    for offset in range(startframe, endframe):
        imagesdir = path / f"colmap_{offset}" / "images"
        for cam in cameras:
            assert (imagesdir / cam["filename"]).exists(), \
                f"Refusing cleanup: missing {imagesdir / cam['filename']}"

    for cam in cameras:
        camfolder = path / cam["orig_stem"]
        if camfolder.exists():
            shutil.rmtree(camfolder)
            print(f"Removed temporary frame folder {camfolder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a Neural 3D Video scene for TAD-GS")
    parser.add_argument("--videopath", required=True, type=str,
                        help="scene folder, e.g. data/neural_3d_video/cook_spinach")
    parser.add_argument("--startframe", default=0, type=int)
    parser.add_argument("--endframe", default=300, type=int)
    parser.add_argument("--frame_interval", default=20, type=int,
                        help="run COLMAP on every Nth frame")
    parser.add_argument("--downscale", default=2, type=int,
                        help="image downscale factor (2 -> 1352x1014 for N3D)")
    parser.add_argument("--pointcloud", default="sparse", choices=["sparse", "dense"],
                        help="sparse: triangulated SfM points on every COLMAP offset; "
                             "dense: additionally run MVS and voxel-downsample the fused "
                             "point cloud, but only on offset 0 (all other offsets stay sparse)")
    parser.add_argument("--max_dense_points", default=40000, type=int,
                        help="target size of the downsampled dense point cloud")

    args = parser.parse_args()
    videopath = Path(args.videopath)

    if args.startframe >= args.endframe:
        raise SystemExit("start frame must be smaller than end frame")
    if args.startframe < 0 or args.endframe > 300:
        raise SystemExit("frame must be in range 0-300")

    print(f"params: startframe={args.startframe} endframe={args.endframe} "
          f"frame_interval={args.frame_interval} downscale={args.downscale} "
          f"pointcloud={args.pointcloud} videopath={videopath}")

    # step 1: unzip the scene archive if needed
    unzip_scene(videopath)

    # step 2: extract all frames from all videos (training uses every frame)
    # Cameras are renumbered sequentially (cam00, cam01, ...) by sorted position
    # so scenes missing individual camera numbers (coffee_martini, cut_roasted_beef)
    # still produce a gap-free 0..N-1 range; pose lookup stays keyed by position.
    videoslist = sorted(videopath.glob("cam*.mp4"))
    seq_names = sequential_camera_names(videopath)
    for v in tqdm.tqdm(videoslist, desc="Extract frames from videos"):
        extractframes(v, seq_names[v.stem], args.startframe, args.endframe, downscale=args.downscale)

    cameras = load_cameras(videopath, seq_names, args.downscale)

    # step 3: COLMAP on every Nth frame, always including the final frame
    colmap_offsets = list(range(args.startframe, args.endframe, args.frame_interval))
    last_frame = args.endframe - 1
    if last_frame not in colmap_offsets:
        colmap_offsets.append(last_frame)
    print(f"Running COLMAP on {len(colmap_offsets)} frames: {colmap_offsets}")
    for offset in colmap_offsets:
        colmapfolder = videopath / f"colmap_{offset}"

        if (colmapfolder / "sparse" / "0" / "points3D.bin").exists():
            print(f"Skipping {colmapfolder} (sparse model already exists)")
        else:
            preparecolmapdynerf(videopath, cameras, offset)
            write_manual_model(colmapfolder, cameras)
            write_colmap_db(colmapfolder, cameras)
            run_colmap_sparse(str(colmapfolder))

        # Dense MVS runs on every COLMAP offset for dense-point-cloud scenes.
        # Each offset's MVS is an isolated COLMAP subprocess, so GPU memory is
        # released between offsets; the heavy intermediates are deleted right
        # after downsampling to keep the disk footprint bounded across the
        # sequence. Gating on fused_downsample.ply keeps the whole step
        # resumable without re-running MVS.
        if args.pointcloud == "dense":
            workspace = colmapfolder / "dense" / "workspace"
            fused = workspace / "fused.ply"
            downsampled = workspace / "fused_downsample.ply"
            if not downsampled.exists():
                if not fused.exists():
                    run_colmap_dense(str(colmapfolder))
                # Imported lazily: open3d is only needed for dense mode.
                from downsample_point import downsample_ply
                downsample_ply(fused, downsampled, max_points=args.max_dense_points)
                cleanup_dense_workspace(str(colmapfolder))

    # step 4: same colmap_<t>/images + manual structure for every other timestamp
    distribute_frames(videopath, args.startframe, args.endframe, colmap_offsets, cameras)

    # step 5: the temporary per-camera frame folders are no longer needed
    cleanup_cam_folders(videopath, args.startframe, args.endframe, cameras)

    print("Done.")
