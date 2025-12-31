import os
import csv
from pathlib import Path
from collections import defaultdict
from compare_organ import compare_organ
from typing import List
import itertools
import nibabel as nib
import numpy as np
from pathlib import Path
from collections import defaultdict
import itertools
import csv
import re

def get_mask_folder_name(mask_path: str) -> str:

    path = os.path.normpath(mask_path)

    # 优先匹配 AA1.1_xxx_ccvl 模式
    m = re.search(r"AA1\.1_([^_/]+)_ccvl", path)
    if m:
        return m.group(1)

    # 否则取 /data/ 后一级目录名
    parts = path.split(os.sep)
    if "data" in parts:
        idx = parts.index("data")
        if idx + 1 < len(parts):
            folder = parts[idx + 1]
            # 去掉类似 "_top10"、"_ccvl40" 这类尾缀
            folder = re.sub(r"(_top\d+|_ccvl\d+)$", "", folder)
            return folder

    # fallback：取最后一级目录名
    return os.path.basename(path)
def dice_score(mask1, mask2):
    """计算两个 segmentation 的 Dice 系数"""
    mask1_bin = mask1 > 0
    mask2_bin = mask2 > 0
    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    total = mask1_bin.sum() + mask2_bin.sum()
    if total == 0:
        return 1.0  # 两者都为空，认为完全一致
    return 2.0 * intersection / total


import nibabel as nib
import numpy as np
import csv
import itertools
from collections import defaultdict
from pathlib import Path

def dice_score(a, b):
    """计算 Dice 系数"""
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    intersection = np.sum(a * b)
    denom = np.sum(a) + np.sum(b)
    return 2.0 * intersection / denom if denom > 0 else 1.0


def record_repeated_organs_with_dice(ct_root: str, mask_roots: list, output_root: str, dice_threshold: float = 0.8):
    """
    遍历所有 subject，找出在多个 mask 目录中重复出现的器官，
    并计算两两 Dice Score。
    若 Dice < 阈值（默认 0.8），则单独记录到一个 summary CSV。
    """
    print("=== record_repeated_organs_with_dice START ===")
    print(f"[ARGS] ct_root={ct_root}")
    print(f"[ARGS] mask_roots({len(mask_roots)}): {mask_roots}")
    print(f"[ARGS] output_root={output_root}")

    ct_root = Path(ct_root)
    mask_roots = [Path(m) for m in mask_roots]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 低 Dice 的汇总文件（跨 subject）
    low_dice_summary = output_root / "low_dice_summary.csv"
    summary_header = ["Subject", "Organ", "Mask1", "Mask2", "Dice_Score"]

    with open(low_dice_summary, "w", newline="") as sf:
        summary_writer = csv.writer(sf)
        summary_writer.writerow(summary_header)

        # 遍历每个 subject
        for subject_dir in ct_root.iterdir():
            if not subject_dir.is_dir():
                continue

            subject_name = subject_dir.name
            ct_file = subject_dir / "ct.nii.gz"
            if not ct_file.exists():
                print(f"[SKIP] No CT found for {subject_name}")
                continue

            # 收集器官 mask
            organ_masks = defaultdict(list)
            for mask_root in mask_roots:
                seg_dir = mask_root / subject_name / "segmentations"
                if not seg_dir.exists():
                    continue

                for organ_file in seg_dir.glob("*.nii.gz"):
                    organ_name = organ_file.name[:-7]
                    if organ_name in [
                        'kidneys','liver','pancreas','kidney_left','kidney_right',
                        'aorta','postcava','spleen','stomach','gall_bladder'
                    ]:
                        organ_masks[organ_name].append(str(organ_file))

            # 只保留出现在 ≥2 个 mask 中的器官
            repeated_organs = {org: paths for org, paths in organ_masks.items() if len(paths) >= 2}
            if not repeated_organs:
                print(f"[SKIP] no repeated organs for {subject_name}")
                continue

            subject_out_dir = output_root / subject_name
            subject_out_dir.mkdir(parents=True, exist_ok=True)
            csv_file = subject_out_dir / "repeated_organs_with_dice.csv"

            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["Subject", "Organ", "Mask1", "Mask2", "Dice_Score"]
                writer.writerow(header)

                for organ, paths in repeated_organs.items():
                    print(f"[COMPARE] {subject_name} | {organ} ({len(paths)} masks)")
                    for mask_a, mask_b in itertools.combinations(paths, 2):
                        try:
                            data_a = nib.load(mask_a).get_fdata()
                            data_b = nib.load(mask_b).get_fdata()
                            dice = dice_score(data_a, data_b)
                            writer.writerow([subject_name, organ, mask_a, mask_b, f"{dice:.4f}"])
                            print(f"  Dice={dice:.4f} | {Path(mask_a).parent.parent.name} vs {Path(mask_b).parent.parent.name}")

                            # 若 Dice < 阈值，则写入 summary 表
                            if dice < dice_threshold:
                                summary_writer.writerow([subject_name, organ, mask_a, mask_b, f"{dice:.4f}"])
                        except Exception as e:
                            print(f"[ERROR] failed to compute Dice for {mask_a}, {mask_b}: {e}")
                            writer.writerow([subject_name, organ, mask_a, mask_b, "ERROR"])
                            summary_writer.writerow([subject_name, organ, mask_a, mask_b, "ERROR"])

            print(f"[OK] {subject_name} -> {csv_file}")

    print(f"[SUMMARY] Low-Dice summary written to: {low_dice_summary}")
    print("=== record_repeated_organs_with_dice END ===")



def select_best_mask(ct_path: str,
                     mask_paths: list,
                     organs: list,
                     base_output: str = "./comparison_results",
                     base_csv: str = "./results",
                     port: str = "8000",
                     log_file: str = "./comparison_summary.log") -> str:
    """
    在多个 mask 中选出综合最优者。
    对每个 organ 做一轮相邻两两比较，
    每次比较后保证“更好的 mask 在右侧”，
    多轮后队尾即为综合最优 mask。

    ✔ 在只有两个 masks 时也严格正确
    """

    # ---------- 基本保护 ----------
    if not mask_paths:
        return ""

    if len(mask_paths) == 1:
        return mask_paths[0]

    if not organs:
        # 没有任何可比较 organ 时，明确退化策略
        return mask_paths[0]

    bdmap_id = os.path.basename(os.path.dirname(ct_path))
    organ_order = sorted(set(organs))
    n = len(mask_paths)

    # 初始化顺序（保留输入顺序）
    mask_info = [{"path": m, "id": f"mask{idx+1}"} for idx, m in enumerate(mask_paths)]

    # ---------- 核心逻辑 ----------
    for organ in organ_order:
        for i in range(n - 1):
            m1, m2 = mask_info[i], mask_info[i + 1]

            run_id = (
                f"{bdmap_id}_{organ}_"
                f"{get_mask_folder_name(m1['path'])}_vs_"
                f"{get_mask_folder_name(m2['path'])}"
            )

            best = compare_organ(
                ct_path,
                m1["path"],
                m2["path"],
                organ,
                base_output=base_output,
                base_csv=base_csv,
                port=port,
                log_file=log_file,
                run_id=run_id
            )

            # -------- 关键修复点 --------
            # 无论更好的是左还是右，都要保证：
            # “更好的在右边”
            if best == m1["path"]:
                # 左边更好 → swap，把它推到右边
                mask_info[i], mask_info[i + 1] = mask_info[i + 1], mask_info[i]
            elif best == m2["path"]:
                # 右边更好 → 已经在右，无需动作
                pass
            else:
                # compare_organ 返回异常值，直接跳过（保守）
                print(f"[WARN] Unexpected compare_organ result: {best}")

    # 队尾一定是“综合更优”的那个
    return mask_info[-1]["path"]

def select_best_mask_2_wrong(ct_path: str,
                     mask_paths: List[str],
                     organs: List[str],
                     base_output: str = "./comparison_results",
                     base_csv: str = "./results",
                     port: str = "8000",
                     log_file: str = "./comparison_summary.log") -> str:
    """
    用冒泡排序在多个 mask 中选出“最佳”者。
    规则：对每个 organ，按相邻两两比较把更好的 mask 往右“冒”，
    多轮后队尾即为综合更优的 mask。

    Parameters
    ----------
    ct_path : str
        该 subject 的 ct.nii.gz 路径
    mask_paths : List[str]
        该 subject 的 segmentation 目录列表（/subject/segmentation）
    organs : List[str]
        至少出现在两个及以上 mask 的重复器官列表
    base_output/base_csv/port/log_file :
        透传给 compare_organ 的参数

    Returns
    -------
    str
        最终认为最佳的 mask 目录路径
    """
    bdmap_id = os.path.basename(os.path.dirname(ct_path))
    masks = mask_paths  # 去重且保持顺序
    n = len(masks)
    if False:
        if n == 0:
            return ""
        if n == 1:
            return masks[0]
        if not organs:
            # 没有可比较的器官时，保守返回第一个
            return masks[0]

    # 为了可重复性，固定 organ 顺序
    organ_order = sorted(set(organs))

    # 为每个 mask 绑定固定 ID
    mask_info = [{"path": m, "id": f"mask{idx+1}"} for idx, m in enumerate(masks)]

    organ_order = sorted(set(organs))
    if True:
        
        for organ in organ_order:
            for i in range(n -1):
                m1, m2 = mask_info[i], mask_info[i + 1]

                run_id = f"{bdmap_id}_{organ}_{get_mask_folder_name(m1['path'])}_vs_{get_mask_folder_name(m2['path'])}"
                print(base_output, base_csv, ct_path)
                
                if True:
                    best = compare_organ(
                        ct_path, m1["path"], m2["path"], organ,
                        base_output=base_output,
                        base_csv=base_csv,
                        port=port,
                        log_file=log_file,
                        run_id=run_id
                    )
                if False:
                    print(f"[WARN] compare_organ failed for {run_id}: {e}")
                    continue

                # 根据 compare_organ 结果更新顺序
                if best == m2["path"]:
                    mask_info[i], mask_info[i + 1] = mask_info[i + 1], mask_info[i]
                    swapped_any = True



    return mask_info[-1]["path"]

from pathlib import Path
from collections import defaultdict
import csv
import os


def record_repeated_organs(ct_root: str, mask_roots: list, output_root: str):
    """
    ct_root: 包含多个subject文件夹，每个文件夹里有ct.nii.gz
    mask_roots: 多个mask根目录，每个mask目录下同样有subject文件夹
    output_root: 结果输出的根目录
    """
    print("=== record_repeated_organs START ===")
    print(f"[ARGS] ct_root={ct_root}")
    print(f"[ARGS] mask_roots({len(mask_roots)}): {mask_roots}")
    print(f"[ARGS] output_root={output_root}")

    ct_root = Path(ct_root)
    mask_roots = [Path(m) for m in mask_roots]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    subject_dirs = list(ct_root.iterdir())
    print(f"[INFO] Total entries: {len(subject_dirs)}")

    for subject_dir in subject_dirs:
        print("\n---")
        print(f"[SCAN] checking entry: {subject_dir}")

        if not subject_dir.is_dir():
            print(f"[SKIP] not a directory")
            continue

        subject_name = subject_dir.name
        ct_file = subject_dir / "ct.nii.gz"

        if not ct_file.exists():
            print(f"[SKIP] CT not found: {ct_file}")
            continue

        print(f"[SUBJECT] {subject_name}")

        # organ -> list of mask_root
        organ_masks = defaultdict(list)
        found_any_mask = False

        for mask_root in mask_roots:
            seg_dir = mask_root / subject_name / "segmentations"
            print(f"[LOOKUP] {seg_dir}")

            if not seg_dir.exists():
                print(f"[MISS] segmentation dir not found")
                continue

            organ_files = list(seg_dir.glob("*.nii.gz"))
            if organ_files:
                found_any_mask = True

            for organ_file in organ_files:
                fname = organ_file.name
                organ_name = (
                    fname[:-7] if fname.endswith(".nii.gz")
                    else fname[:-4] if fname.endswith(".nii")
                    else organ_file.stem
                )
                organ_masks[organ_name].append(str(mask_root))
                print(f"[MAP] {organ_name} <- {mask_root}")

        if not found_any_mask:
            print(f"[SKIP] no masks found")
            continue

        # 只保留重复器官
        repeated_organs = {
            organ: paths for organ, paths in organ_masks.items()
            if len(paths) >= 2
        }

        if not repeated_organs:
            print(f"[SKIP] no repeated organs")
            continue

        print(f"[SUMMARY] repeated organs:")
        for organ, paths in repeated_organs.items():
            print(f"  - {organ}: {paths}")

        # 输出目录
        subject_out_dir = output_root / subject_name
        subject_out_dir.mkdir(parents=True, exist_ok=True)

        csv_file = subject_out_dir / "repeated_organs.csv"
        file_exists = csv_file.exists()

        print(f"[WRITE] CSV -> {csv_file} (append mode)")

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["Organ", "Mask_Paths", "Best_Mask"])

            # =========================
            # ✅ 关键改动：按 organ 单独选择 best mask
            # =========================
            for organ, paths in repeated_organs.items():
                print(f"\n[ORGAN] {organ}")

                # 该 organ 的候选 segmentation dirs
                seg_dirs = [
                    str(Path(p) / subject_name / "segmentations")
                    for p in paths
                ]

                print(f"[CALL] select_best_mask")
                print(f"       ct={ct_file}")
                print(f"       organ={organ}")
                print(f"       masks={seg_dirs}")

                best_mask = select_best_mask(
                    str(ct_file),
                    seg_dirs,
                    [organ],  # ⚠️ 单 organ
                    base_output="/standard/CBIG-Standard-ECE/zejun/Deprecated/AnnotationVLM/data/vs"
                )

                print(f"[RESULT] {organ} -> best_mask={best_mask}")

                writer.writerow([
                    organ,
                    "; ".join(paths),
                    best_mask
                ])

        print(f"[OK] {subject_name} -> {csv_file}")

    print("\n=== record_repeated_organs END ===")

def record_repeated_organs_old(ct_root: str, mask_roots: list, output_root: str):
    """
    ct_root: 包含多个subject文件夹，每个文件夹里有ct.nii.gz
    mask_roots: 多个mask根目录，每个mask目录下同样有subject文件夹
    output_root: 结果输出的根目录
    """
    print("=== record_repeated_organs START ===")
    print(f"[ARGS] ct_root={ct_root}")
    print(f"[ARGS] mask_roots({len(mask_roots)}): {mask_roots}")
    print(f"[ARGS] output_root={output_root}")

    ct_root = Path(ct_root)
    mask_roots = [Path(m) for m in mask_roots]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[PATH] ct_root resolved: {ct_root.resolve()}")
    except Exception:
        print(f"[PATH] ct_root (unresolved): {ct_root}")
    try:
        print(f"[PATH] output_root resolved: {output_root.resolve()}")
    except Exception:
        print(f"[PATH] output_root (unresolved): {output_root}")
    print("[PATH] mask_roots resolved:")
    for mr in mask_roots:
        try:
            print(f"  - {mr.resolve()}")
        except Exception:
            print(f"  - {mr}")
    # 遍历ct_root中所有subject文件夹
    subject_dirs = list(ct_root.iterdir())   # 一次性读取，避免生成器分页问题
    print(f"[INFO] Total entries: {len(subject_dirs)}")

    for subject_dir in subject_dirs:
        print("\n---")
        print(f"[SCAN] checking entry: {subject_dir}")
        if not subject_dir.is_dir():
            print(f"[SKIP] not a directory: {subject_dir}")
            continue

        subject_name = subject_dir.name
        path = "/standard/CBIG-Standard-ECE/zejun/Deprecated/AnnotationVLM/data/vs/" + str(subject_name) + "_aorta_Deprecated_vs_Deprecated"
        print(path)
        print(os.path.exists(path))
        if os.path.exists(path):
            continue
        print(f"[SUBJECT] name={subject_name}")
        ct_file = subject_dir / "ct.nii.gz"
        print(f"[CHECK] expect CT file: {ct_file}")
        if not ct_file.exists():
            print(f"[SKIP] CT not found for subject {subject_name}: {ct_file}")
            continue

        # 统计该subject每个器官在哪些mask中出现
        organ_masks = defaultdict(list)  # organ_name -> list of mask_root
        found_any_mask = False

        for mask_root in mask_roots:
            mask_subject_seg_dir = mask_root / subject_name / "segmentations"
            print(f"[LOOKUP] mask_root={mask_root} -> seg_dir={mask_subject_seg_dir}")
            if not mask_subject_seg_dir.exists():
                print(f"[MISS] segmentation dir NOT found: {mask_subject_seg_dir}")
                continue

            organ_files = list(mask_subject_seg_dir.glob("*.nii.gz"))
            print(f"[FOUND] organ files ({len(organ_files)}) in {mask_subject_seg_dir}:")
            if not organ_files:
                print("        (none)")
            else:
                print("        " + ", ".join(f.name for f in organ_files))
                found_any_mask = True

            for organ_file in organ_files:
                # 正确剥离 .nii.gz（而非 Path.stem 只去掉最后一个后缀）
                fname = organ_file.name

                if True:#['kidneys','liver','pancreas', 'kidney_left', 'kidney_right', 'aorta','postcava','spleen','stomach','gall_bladder']:
                    
                    organ_name = fname[:-7] if fname.endswith(".nii.gz") else (fname[:-4] if fname.endswith(".nii") else organ_file.stem)
                    print(f"[MAP] organ_file={organ_file.name} -> organ_name={organ_name} | add mask_root={mask_root}")
                    organ_masks[organ_name].append(str(mask_root))

        if not found_any_mask:
            print(f"[SKIP] no organ masks found across mask_roots for subject {subject_name}")
            continue

        # 找出至少出现在两个或以上mask的器官
        repeated_organs = {organ: paths for organ, paths in organ_masks.items() if len(paths) >= 2}
        print(f"[SUMMARY] total organs seen={len(organ_masks)} | repeated_organs={len(repeated_organs)}")
        if repeated_organs:
            for organ, paths in repeated_organs.items():
                print(f"  - organ={organ}, count={len(paths)}, mask_roots=" + "; ".join(paths))
        else:
            print(f"[SKIP] no repeated organs (>=2 masks) for subject {subject_name}")
            continue  # 没有重复的器官则跳过

        # 为该subject创建输出文件夹
        subject_out_dir = output_root / subject_name
        subject_out_dir.mkdir(parents=True, exist_ok=True)
        try:
            print(f"[OUTPUT] subject_out_dir: {subject_out_dir.resolve()}")
        except Exception:
            print(f"[OUTPUT] subject_out_dir: {subject_out_dir}")

        # === 修复点1：把 mask_root 转成该 subject 的 segmentation 目录 ===
        #mask_roots_used = list({p for paths in repeated_organs.values() for p in paths})
        # 不去重版本
        mask_roots_used = [p for paths in repeated_organs.values() for p in paths]


        seg_dirs_for_subject = [str(Path(p) / subject_name / "segmentations") for p in mask_roots_used]
        print(f"[SELECT] seg_dirs_for_subject({len(seg_dirs_for_subject)}): " + ", ".join(seg_dirs_for_subject))

        # === 修复点2：把 organs 列表传给 select_best_mask，避免 TypeError ===
        organs_list = list(repeated_organs.keys())
        print(f"[SELECT] organs_list({len(organs_list)}): " + ", ".join(organs_list))

        print(f"[CALL] select_best_mask(ct={ct_file}, masks={len(seg_dirs_for_subject)}, organs={len(organs_list)})")

        best_mask = select_best_mask(str(ct_file), seg_dirs_for_subject, organs_list, base_output= "/standard/CBIG-Standard-ECE/zejun/Deprecated/AnnotationVLM/data/vs")
        print(f"[RESULT] best_mask={best_mask}")

        # 生成csv表格
        csv_file = subject_out_dir / "repeated_organs.csv"
        file_exists = csv_file.exists()

        print(f"[WRITE] CSV -> {csv_file} (append mode)")

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                header = ["Organ", "Mask_Paths", "Best_Mask"]
                print(f"[CSV] header: {header}")
                writer.writerow(header)

            for organ, paths in repeated_organs.items():
                row = [organ, "; ".join(paths), best_mask]
                print(f"[CSV] row: {row}")
                writer.writerow(row)

        print(f"[OK] {subject_name} -> {csv_file}")

    print("\n=== record_repeated_organs END ===")

import csv
from pathlib import Path
from collections import defaultdict

def record_repeated_organs_from_csv(
    csv_path: str,
    ct_root: str,
    output_root: str,
    base_output: str = "/home2/jzs6wq/data/Deprecated/pants/PanTS/projectio_aorta",
):
    """
    从 low_dice_summary.csv 或其子集（如 0.3–0.8 过滤后）读取需要处理的条目，
    按 subject 聚合，调用 select_best_mask，然后为每个 subject 写 repeated_organs.csv。

    要求 CSV 至少包含列：
      - Subject
      - Organ
      - Mask
      - Ref_Mask
    其他多出来的列（例如 CT_Path, Dice_Score）会被忽略。
    """
    csv_path = "/standard/CBIG-Standard-ECE/zejun/Deprecated/AnnotationVLM/dice_0.3_0.8_cases.csv"
    print("=== record_repeated_organs_from_csv START ===")
    csv_path = Path(csv_path)
    ct_root = Path(ct_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[ARGS] csv_path={csv_path}")
    print(f"[ARGS] ct_root={ct_root}")
    print(f"[ARGS] output_root={output_root}")

    # ---- Step 1: 读 CSV，检查列名，并按 subject / organ 分组 ----
    subjects = defaultdict(lambda: defaultdict(list))

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        print(f"[CSV] fieldnames = {fieldnames}")

        required_cols = {"Subject", "Organ", "Mask", "Ref_Mask"}
        if not required_cols.issubset(set(fieldnames or [])):
            raise ValueError(
                f"CSV 缺少必要列：{required_cols - set(fieldnames or [])}，"
                f"实际列名为：{fieldnames}"
            )

        for row in reader:
            subj = row["Subject"].strip()
            organ = row["Organ"].strip()
            mask_path = row["Mask"].strip()
            ref_mask_path = row["Ref_Mask"].strip()

            if not subj or not organ or not mask_path:
                # 避免空行或坏行
                print(f"[WARN] 跳过异常行：{row}")
                continue

            subjects[subj][organ].append((mask_path, ref_mask_path))

    print(f"[INFO] Loaded subjects from CSV: {len(subjects)}")

    # ---- Step 2: 遍历每个 subject，调用 select_best_mask ----
    for subject, organ_dict in subjects.items():
        print("\n---")
        print(f"[SUBJECT] {subject}")

        ct_path = ct_root / subject / "ct.nii.gz"
        print(f"[CT] expect: {ct_path}")
        if not ct_path.exists():
            print(f"[SKIP] CT not found for subject {subject}: {ct_path}")
            continue

        # 收集所有相关 segmentation 目录
        seg_dirs = set()
        for organ, mask_ref_pairs in organ_dict.items():
            for mask_path, ref_mask_path in mask_ref_pairs:
                seg_dir = Path(mask_path).parent  # .../subject/segmentations
                seg_dirs.add(str(seg_dir))

        seg_dirs = sorted(seg_dirs)
        organs_list = sorted(organ_dict.keys())

        print(f"[INFO] organs({len(organs_list)}): {', '.join(organs_list)}")
        print(f"[INFO] seg_dirs({len(seg_dirs)}):")
        for d in seg_dirs:
            print(f"   - {d}")

        if not seg_dirs:
            print(f"[SKIP] no seg_dirs for subject {subject}")
            continue

        print(f"[CALL] select_best_mask(")
        print(f"       ct={ct_path},")
        print(f"       masks={len(seg_dirs)},")
        print(f"       organs={len(organs_list)},")
        print(f"       base_output={base_output}")
        print(")")

        # 这里假定 select_best_mask 签名为：
        #   select_best_mask(ct_path: str, mask_dirs: List[str], organs: List[str], base_output: str) -> str
        best_mask = select_best_mask(str(ct_path), seg_dirs, organs_list, base_output=base_output)
        print(f"[RESULT] best_mask={best_mask}")

        # ---- Step 3: 为该 subject 写 repeated_organs.csv ----
        subject_out_dir = output_root / subject
        subject_out_dir.mkdir(parents=True, exist_ok=True)

        csv_file = subject_out_dir / "repeated_organs.csv"
        print(f"[WRITE] {csv_file}")

        with csv_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Organ", "Mask_Paths", "Best_Mask"])
            for organ, mask_ref_pairs in organ_dict.items():
                mask_paths = [m for (m, _) in mask_ref_pairs]
                writer.writerow([organ, "; ".join(mask_paths), best_mask])

        print(f"[OK] {subject} -> {csv_file}")

    print("\n=== record_repeated_organs_from_csv END ===")


# 运行示例（替换成你实际的路径）
if __name__ == "__main__":
    ct_root = "/standard/CBIG-Standard-ECE/zejun/Deprecated/AnnotationVLM/data/dice_0.3_0.8"
    mask_roots = [
              "/standard/CBIG-Standard-ECE/zejun/Deprecated/AnnotationVLM/data/dice_0.3_0.8_mask/AA1.1_dhe23_ccvl40",
              "/standard/CBIG-Standard-ECE/zejun/Deprecated/AnnotationVLM/data/dice_0.3_0.8_mask/AA1.1_jliu452_ccvl40",
              "/standard/CBIG-Standard-ECE/zejun/Deprecated/AnnotationVLM/data/dice_0.3_0.8_mask/AA1.1_nwang52_ccvl40",
              "/standard/CBIG-Standard-ECE/zejun/Deprecated/AnnotationVLM/data/dice_0.3_0.8_mask/AbdomenAtlasPro"]
    output_root = "/standard/CBIG-Standard-ECE/zejun/Deprecated/AnnotationVLM/data/dice_0.3_0.8_results"
    record_repeated_organs(ct_root, mask_roots, output_root)
    #record_repeated_organs_from_csv(ct_root, mask_roots, output_root)