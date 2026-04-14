import math
from typing import List, Dict, Any, Tuple

import gemmi
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CIF Σ Calculator", layout="wide")

METAL_ELEMENTS = {
    "Fe", "Co", "Mn", "Ni", "Cu", "Zn", "Cr", "V", "Ru", "Rh", "Pd", "Pt"
}
DONOR_ELEMENTS = {"N", "O", "S", "Cl", "Br", "F"}
DEFAULT_MAX_CENTERS = 2
DEFAULT_RADIUS = 2.5
DEFAULT_EXPECTED_CN = 6


def angle_deg(v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> float:
    dot = sum(x * y for x, y in zip(v1, v2))
    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(x * x for x in v2))
    if n1 == 0 or n2 == 0:
        return float("nan")
    c = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(c))


def find_metal_sites(small: gemmi.SmallStructure, max_centers: int) -> List[gemmi.SmallStructure.Site]:
    centers = []
    for site in small.sites:
        if site.element.name in METAL_ELEMENTS:
            centers.append(site)
    return centers[:max_centers]


def build_neighbor_candidates(
    small: gemmi.SmallStructure,
    center_site: gemmi.SmallStructure.Site,
    radius: float
) -> List[Dict[str, Any]]:
    """
    center_site の周囲にある donor 候補の実像を返す。
    PBC / 対称操作の影響で複数像が半径内に入る場合にも対応。
    """
    ns = gemmi.NeighborSearch(small, radius).populate()
    center_frac = center_site.fract
    center_cart = small.cell.orthogonalize(center_frac)

    candidates: List[Dict[str, Any]] = []
    seen = set()

    marks = ns.find_site_neighbors(center_site, min_dist=0.1, max_dist=radius)

    for mark in marks:
        site = mark.to_site(small)

        if site.label == center_site.label:
            continue
        if site.element.name not in DONOR_ELEMENTS:
            continue

        # mark.pos は対称操作後の位置
        fpos = small.cell.fractionalize(mark.pos)

        images = small.cell.find_nearest_pbc_images(center_frac, radius, fpos, 0)
        if not images:
            images = [small.cell.find_nearest_pbc_image(center_cart, mark.pos, 0)]

        for im in images:
            im_frac = small.cell.fract_image(im, fpos)
            im_cart = small.cell.orthogonalize(im_frac)

            dx = im_cart.x - center_cart.x
            dy = im_cart.y - center_cart.y
            dz = im_cart.z - center_cart.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            key = (
                site.label,
                site.element.name,
                round(im_cart.x, 5),
                round(im_cart.y, 5),
                round(im_cart.z, 5),
            )
            if key in seen:
                continue
            seen.add(key)

            candidates.append(
                {
                    "label": site.label,
                    "element": site.element.name,
                    "fract": im_frac,
                    "cart": im_cart,
                    "distance": dist,
                    "image_idx": mark.image_idx,
                }
            )

    candidates.sort(key=lambda x: x["distance"])
    return candidates


def choose_ligands(candidates: List[Dict[str, Any]], expected_cn: int) -> List[Dict[str, Any]]:
    """
    距離の近い順を基本に、まず同一ラベル重複を避けて expected_cn 個選ぶ。
    """
    chosen = []
    used_labels = set()

    for c in candidates:
        if c["label"] in used_labels:
            continue
        chosen.append(c)
        used_labels.add(c["label"])
        if len(chosen) == expected_cn:
            return chosen

    for c in candidates:
        if c in chosen:
            continue
        chosen.append(c)
        if len(chosen) == expected_cn:
            return chosen

    return chosen


def compute_angles_from_ligands(
    center_cart: gemmi.Position,
    ligands: List[Dict[str, Any]]
) -> Dict[str, Any]:
    vecs = []
    for lig in ligands:
        cart = lig["cart"]
        vecs.append(
            (
                cart.x - center_cart.x,
                cart.y - center_cart.y,
                cart.z - center_cart.z,
            )
        )

    all_angles = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            ang = angle_deg(vecs[i], vecs[j])
            all_angles.append(
                {
                    "pair": f"{ligands[i]['label']} - {ligands[j]['label']}",
                    "angle": ang,
                    "delta90": abs(90.0 - ang),
                }
            )

    # 15 個のうち 90° に近い 12 個を Σ に使う
    all_angles_sorted_for_sigma = sorted(all_angles, key=lambda x: x["delta90"])
    cis_used = all_angles_sorted_for_sigma[:12]
    trans_like = all_angles_sorted_for_sigma[12:]
    sigma = sum(x["delta90"] for x in cis_used)

    return {
        "all_angles": sorted(all_angles, key=lambda x: x["angle"]),
        "cis_used": sorted(cis_used, key=lambda x: x["angle"]),
        "trans_like": sorted(trans_like, key=lambda x: x["angle"]),
        "sigma": sigma,
    }


def analyze_center(
    small: gemmi.SmallStructure,
    center_site: gemmi.SmallStructure.Site,
    radius: float,
    expected_cn: int,
) -> Dict[str, Any] | None:
    center_cart = small.cell.orthogonalize(center_site.fract)
    candidates = build_neighbor_candidates(small, center_site, radius)

    if len(candidates) < expected_cn:
        return None

    ligands = choose_ligands(candidates, expected_cn)
    if len(ligands) < expected_cn:
        return None

    angle_info = compute_angles_from_ligands(center_cart, ligands)
    bond_lengths = [lig["distance"] for lig in ligands]
    mean_bond_length = sum(bond_lengths) / len(bond_lengths)

    return {
        "metal_label": center_site.label,
        "metal_element": center_site.element.name,
        "mean_bond_length": round(mean_bond_length, 4),
        "ligands": [
            {
                "label": lig["label"],
                "element": lig["element"],
                "distance": round(lig["distance"], 4),
                "image_idx": lig["image_idx"],
                "fract_x": round(lig["fract"].x, 5),
                "fract_y": round(lig["fract"].y, 5),
                "fract_z": round(lig["fract"].z, 5),
            }
            for lig in ligands
        ],
        "sigma": round(angle_info["sigma"], 3),
        "cis_angles_used": [round(x["angle"], 3) for x in angle_info["cis_used"]],
        "trans_like_angles": [round(x["angle"], 3) for x in angle_info["trans_like"]],
        "all_angles": [
            {
                "pair": x["pair"],
                "angle": round(x["angle"], 3),
                "delta90": round(x["delta90"], 3),
            }
            for x in angle_info["all_angles"]
        ],
    }


def analyze_cif(file_bytes: bytes, max_centers: int, radius: float, expected_cn: int) -> List[Dict[str, Any]]:
    text = file_bytes.decode("utf-8", errors="ignore")
    doc = gemmi.cif.read_string(text)
    block = doc.sole_block()
    small = gemmi.make_small_structure_from_block(block)

    centers = find_metal_sites(small, max_centers)
    results = []

    for center_site in centers:
        res = analyze_center(
            small=small,
            center_site=center_site,
            radius=radius,
            expected_cn=expected_cn,
        )
        if res is not None:
            results.append(res)

    return results


# ---------------- UI ----------------

st.title("CIF Σ Calculator")
st.caption("CIF をアップロードすると、八面体金属中心について Σ 値と平均配位結合長を計算します。")

with st.sidebar:
    st.header("設定")
    max_centers = st.number_input(
        "解析する金属中心数の上限",
        min_value=1,
        max_value=10,
        value=DEFAULT_MAX_CENTERS,
        step=1
    )
    radius = st.number_input(
        "近傍探索半径 (Å)",
        min_value=2.0,
        max_value=5.0,
        value=DEFAULT_RADIUS,
        step=0.1
    )
    expected_cn = st.number_input(
        "想定配位数",
        min_value=4,
        max_value=8,
        value=DEFAULT_EXPECTED_CN,
        step=1
    )
    st.markdown("**対象金属**: Fe, Co, Mn, Ni, Cu, Zn, Cr, V, Ru, Rh, Pd, Pt")
    st.markdown("**対象 donor**: N, O, S, Cl, Br, F")

uploaded = st.file_uploader("CIF ファイルを選択", type=["cif"])

if uploaded is not None:
    try:
        results = analyze_cif(uploaded.read(), int(max_centers), float(radius), int(expected_cn))

        if not results:
            st.warning("解析できる金属中心が見つかりませんでした。近傍探索半径や配位数を見直してください。")
        else:
            st.subheader("計算結果")

            summary_rows = []
            for i, res in enumerate(results, start=1):
                ligand_text = ", ".join([x["label"] for x in res["ligands"]])
                summary_rows.append(
                    {
                        "Center": i,
                        "Metal": f"{res['metal_label']} ({res['metal_element']})",
                        "Σ": res["sigma"],
                        "Mean bond length (Å)": res["mean_bond_length"],
                        "Ligands": ligand_text,
                    }
                )

            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True)

            st.markdown("### 見やすい要約")
            for i, res in enumerate(results, start=1):
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1.3, 1, 1])

                    with c1:
                        st.markdown(f"**Center {i}: {res['metal_label']} ({res['metal_element']})**")
                        st.write("採用配位原子:", ", ".join([x["label"] for x in res["ligands"]]))

                    with c2:
                        st.metric("Σ", f"{res['sigma']:.3f}")

                    with c3:
                        st.metric("平均配位結合長 (Å)", f"{res['mean_bond_length']:.4f}")

            st.markdown("### 詳細")
            for i, res in enumerate(results, start=1):
                with st.expander(f"Center {i}: {res['metal_label']} ({res['metal_element']}) の詳細", expanded=False):
                    st.markdown("#### 採用した 6 配位原子像")
                    st.dataframe(pd.DataFrame(res["ligands"]), use_container_width=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Σ に使用した 12 角")
                        st.write(", ".join(f"{x:.3f}" for x in res["cis_angles_used"]))
                    with col2:
                        st.markdown("#### 残り 3 角")
                        st.write(", ".join(f"{x:.3f}" for x in res["trans_like_angles"]))

                    st.markdown("#### 全 15 角")
                    st.dataframe(pd.DataFrame(res["all_angles"]), use_container_width=True)

            csv_rows = []
            for i, res in enumerate(results, start=1):
                csv_rows.append(
                    {
                        "center": i,
                        "metal_label": res["metal_label"],
                        "metal_element": res["metal_element"],
                        "sigma": res["sigma"],
                        "mean_bond_length_A": res["mean_bond_length"],
                        "ligands": "; ".join([x["label"] for x in res["ligands"]]),
                        "distances_A": "; ".join(str(x["distance"]) for x in res["ligands"]),
                        "cis_angles_used_deg": "; ".join(map(str, res["cis_angles_used"])),
                        "trans_like_angles_deg": "; ".join(map(str, res["trans_like_angles"])),
                    }
                )

            csv_data = pd.DataFrame(csv_rows).to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "CSV をダウンロード",
                csv_data,
                file_name="sigma_results.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"解析に失敗しました: {e}")

else:
    st.info("CIF ファイルをアップロードしてください。")

st.markdown("---")
st.markdown(
    "このアプリでは、採用した 6 配位原子から 15 個の角を計算し、"
    "90° に最も近い 12 個を用いて Σ を求めています。"
)
