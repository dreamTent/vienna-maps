#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
import argparse


# ----------------------------
# COLMAP parsing
# ----------------------------

def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ])


def parse_cameras(path):
    cams = {}
    for l in open(path):
        if l.startswith("#") or not l.strip():
            continue
        t = l.split()
        cams[int(t[0])] = {
            "model": t[1],
            "params": list(map(float, t[4:])),
        }
    return cams


def parse_images(path):
    imgs = {}
    lines = open(path).read().splitlines()
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("#") or not l.strip():
            i += 1
            continue
        t = l.split()
        imgs[int(t[0])] = {
            "q": np.array(list(map(float, t[1:5]))),
            "t": np.array(list(map(float, t[5:8]))),
            "cam": int(t[8]),
            "name": t[9],
        }
        i += 2
    return imgs


def K_from_cam(cam):
    p = cam["params"]
    if cam["model"] == "PINHOLE":
        fx, fy, cx, cy = p
    else:  # SIMPLE_PINHOLE fallback
        fx = fy = p[0]
        cx, cy = p[1:3]

    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])


def projection(img, cam):
    R = qvec_to_rotmat(img["q"])
    t = img["t"].reshape(3, 1)
    return K_from_cam(cam) @ np.hstack([R, t])


# ----------------------------
# Features
# ----------------------------

def extractor():
    if hasattr(cv2, "SIFT_create"):
        return "SIFT", cv2.SIFT_create(4000)
    return "ORB", cv2.ORB_create(4000)


def match(desc1, desc2, mode):
    if desc1 is None or desc2 is None:
        return []

    norm = cv2.NORM_L2 if mode == "SIFT" else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm)

    matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


# ----------------------------
# Geometry
# ----------------------------

def triangulate(P1, P2, pts1, pts2):
    X = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    return (X[:3] / X[3]).T


# ----------------------------
# Output
# ----------------------------

def write_ply(path, pts, cols):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, cols):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[2]} {c[1]} {c[0]}\n")


def write_points3D_txt(path, pts, cols):
    with open(path, "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR\n")
        for i, (p, c) in enumerate(zip(pts, cols)):
            f.write(f"{i+1} {p[0]} {p[1]} {p[2]} {c[2]} {c[1]} {c[0]} 0.0\n")


# ----------------------------
# Main
# ----------------------------

def run(model, images_dir, out):
    cams = parse_cameras(Path(model) / "cameras.txt")
    imgs = parse_images(Path(model) / "images.txt")

    name, ext = extractor()
    print("Using:", name)

    cache = {}
    for i, im in imgs.items():
        img = cv2.imread(str(Path(images_dir) / im["name"]))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = ext.detectAndCompute(gray, None)
        pts = np.array([k.pt for k in kp]) if kp else np.empty((0, 2))
        cache[i] = (img, pts, desc)

    pts_all = []
    col_all = []

    ids = list(cache.keys())

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id1, id2 = ids[i], ids[j]
            img1, pts1, d1 = cache[id1]
            img2, pts2, d2 = cache[id2]

            matches = match(d1, d2, name)
            if len(matches) < 30:
                continue

            p1 = np.float32([pts1[m.queryIdx] for m in matches])
            p2 = np.float32([pts2[m.trainIdx] for m in matches])

            F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 2.0)
            if F is None:
                continue

            p1 = p1[mask.ravel() == 1]
            p2 = p2[mask.ravel() == 1]

            P1 = projection(imgs[id1], cams[imgs[id1]["cam"]])
            P2 = projection(imgs[id2], cams[imgs[id2]["cam"]])

            X = triangulate(P1, P2, p1, p2)

            # basic filtering
            good = np.isfinite(X).all(axis=1) & (np.linalg.norm(X, axis=1) < 1e6)
            X = X[good]
            p1 = p1[good]

            colors = []
            for x, y in p1:
                xi, yi = int(x), int(y)
                h, w = img1.shape[:2]
                xi = max(0, min(w - 1, xi))
                yi = max(0, min(h - 1, yi))
                colors.append(img1[yi, xi])

            pts_all.append(X)
            col_all.append(np.array(colors))

            print(f"{imgs[id1]['name']} <-> {imgs[id2]['name']} : {len(X)} pts")

    pts = np.vstack(pts_all)
    cols = np.vstack(col_all)

    # simple dedup
    grid = np.floor(pts / 0.05)
    _, idx = np.unique(grid, axis=0, return_index=True)
    pts = pts[idx]
    cols = cols[idx]

    write_ply(out, pts, cols)
    write_points3D_txt(Path(out).with_name("points3D.txt"), pts, cols)

    print("Done.")
    print("PLY:", out)
    print("points3D:", Path(out).with_name("points3D.txt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="cloud.ply")
    args = ap.parse_args()

    run(args.model, args.images, args.out)