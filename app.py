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
DEFAULT_RADIUS = 3.0
DEFAULT_EXPECTED_CN = 6


def distance(a: gemmi.Position, b: gemmi.Position) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def angle_deg(v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> float:
    dot = sum(x * y for x, y in zip(v1, v2))
    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(x * x for x in v2))
    if n1 == 0 or n2 == 0:
        return float("nan")
    c = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(c))


def classify_angles(angles: List[Tuple[int, int, float]]) -> Tuple[List[float], List[float]]:
    # 15個の角から 180° に最も近い3つを trans、それ以外を cis とする
    trans_triplets = sorted(angles, key=lambda x: abs(180.0 - x[2]))[:3]
    trans_pairs = {(min(i, j), max(i, j)) for i, j, _ in trans_triplets}

    cis = []
    trans = []
    for i, j, ang in angles:
        if (min(i, j), max(i, j)) in trans_pairs:
            trans.append(ang)
        else:
            cis.append(ang)

    cis = sorted(cis, key=lambda x: abs(90.0 - x))[:12]
    trans = sorted(trans)
    return cis, trans


def sigma_from_cis(cis_angles: List[float]) -> float:
    return sum(abs(90.0 - ang) for ang in cis_angles)


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
    ns = gemmi.NeighborSearch(small, radius).populate()
    center_cart = small.cell.orthogonalize(center_site.fract)

    candidates = []
    seen = set()

    # small-molecule向けの近傍探索
    for mark in ns.find_site_neighbors(center_site, min_dist=0.1, max_dist=radius):
        site = mark.to_site(small)

        if site.label == center_site.label:
            continue
        if site.element.name not in DONOR_ELEMENTS:
            continue

        # mark.pos は対称操作・PBC込みの実際の近接位置
        neigh_cart = mark.pos
        d = distance(center_cart, neigh_cart)

        key = (
            site.label,
            site.element.name,
            round(neigh_cart.x, 4),
            round(neigh_cart.y, 4),
            round(neigh_cart.z, 4),
        )
        if key in seen:
            continue
        seen.add(key)

        candidates.append({
            "site": site,
            "label": site.label,
            "element": site.element.name,
            "cart": neigh_cart,
            "distance": d,
            "image_idx": mark.image_idx,
        })

    candidates.sort(key=lambda x: x["distance"])
    return candidates


def analyze_center(
    small: gemmi.SmallStructure,
    center_site: gemmi.SmallStructure.Site,
    radius: float = DEFAULT_RADIUS,
    expected_cn: int = DEFAULT_EXPECTED_CN,
) -> Dict[str, Any] | None:
    center_cart = small.cell.orthogonalize(center_site.fract)
    candidates = build_neighbor_candidates(small, center_site, radius)

    if len(candidates) < expected_cn:
        return None

    ligands = candidates[:expected_cn]

    vecs = []
    lengths = []
    labels = []

    for lig in ligands:
        cart = lig["cart"]
        vec = (
            cart.x - center_cart.x,
            cart.y - center_cart.y,
            cart.z - center_cart.z,
        )
        vecs.append(vec)
        lengths.append(lig["distance"])
        labels.append(f"{lig['label']} ({lig['element']})")

    pair_angles = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            ang = angle_deg(vecs[i], vecs[j])
            pair_angles.append((i, j, ang))

    cis_angles, trans_angles = classify_angles(pair_angles)
    if len(cis_angles) < 12:
        return None

    sigma = sigma_from_cis(cis_angles)

    return {
        "metal_label": center_site.label,
        "metal_element": center_site.element.name,
        "ligand_labels": labels,
        "bond_lengths": [round(x, 4) for x in lengths],
        "cis_angles": [round(x, 3) for x in cis_angles],
        "trans_angles": [round(x, 3) for x in trans_angles],
        "sigma": round(sigma, 3),
    }


def analyze_cif(file_bytes: bytes, max_centers: int, radius: float, expected_cn: int) -> List[Dict[str, Any]]:
    text = file_bytes.decode("utf-8", errors="ignore")
    doc = gemmi.cif.read_string(text)
    block = doc.sole_block()
    small = gemmi.make_small_structure_from_block(block)

    centers = find_metal_sites(small, max_centers)

    results = []
    for center_site in centers:
        result = analyze_center(
            small,
            center_site,
            radius=radius,
            expected_cn=expected_cn,
        )
        if result:
            results.append(result)

    return results


st.title("CIF Σ Calculator")
st.caption("CIF をアップロードして、八面体金属中心の Σ 値を計算します。非対称単位の 2 分子程度までを想定した簡易版です。")

with st.sidebar:
    st.header("設定")
    max_centers = st.number_input("解析する金属中心数の上限", min_value=1, max_value=10, value=2, step=1)
    radius = st.number_input("近傍探索半径 (Å)", min_value=2.0, max_value=4.5, value=3.0, step=0.1)
    expected_cn = st.number_input("想定配位数", min_value=4, max_value=8, value=6, step=1)
    st.markdown("**対象金属**: Fe, Co, Mn, Ni, Cu, Zn, Cr, V, Ru, Rh, Pd, Pt")
    st.markdown("**対象配位原子**: N, O, S, Cl, Br, F")

uploaded = st.file_uploader("CIF ファイルを選択", type=["cif"])

if uploaded is not None:
    data = uploaded.read()
    try:
        results = analyze_cif(data, int(max_centers), float(radius), int(expected_cn))

        if not results:
            st.warning("Σ 値を計算できる金属中心が見つかりませんでした。探索半径や配位数を見直してください。")
        else:
            summary_rows = []
            for i, res in enumerate(results, start=1):
                summary_rows.append({
                    "Center": i,
                    "Metal": f"{res['metal_label']} ({res['metal_element']})",
                    "Sigma": res["sigma"],
                    "Ligands": ", ".join(res["ligand_labels"]),
                })

            st.subheader("結果一覧")
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

            for i, res in enumerate(results, start=1):
                with st.expander(f"Center {i}: {res['metal_label']} ({res['metal_element']})", expanded=True):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Σ", res["sigma"])
                        st.write("**配位原子**")
                        for lab, bl in zip(res["ligand_labels"], res["bond_lengths"]):
                            st.write(f"- {lab}: {bl:.4f} Å")

                    with col2:
                        st.write("**cis angles (12)**")
                        st.write(", ".join(f"{x:.3f}" for x in res["cis_angles"]))
                        st.write("**trans angles (3)**")
                        st.write(", ".join(f"{x:.3f}" for x in res["trans_angles"]))

            csv_rows = []
            for i, res in enumerate(results, start=1):
                csv_rows.append({
                    "center": i,
                    "metal_label": res["metal_label"],
                    "metal_element": res["metal_element"],
                    "sigma": res["sigma"],
                    "ligands": "; ".join(res["ligand_labels"]),
                    "bond_lengths_A": "; ".join(map(str, res["bond_lengths"])),
                    "cis_angles_deg": "; ".join(map(str, res["cis_angles"])),
                    "trans_angles_deg": "; ".join(map(str, res["trans_angles"])),
                })

            csv_data = pd.DataFrame(csv_rows).to_csv(index=False).encode("utf-8-sig")
            st.download_button("CSV をダウンロード", csv_data, file_name="sigma_results.csv", mime="text/csv")

    except Exception as e:
        st.error(f"解析に失敗しました: {e}")
else:
    st.info("左の設定を確認してから CIF をアップロードしてください。")

st.markdown("---")
st.markdown(
    "**注意**: この版は近接原子から 6 配位を推定する簡易実装です。"
    " 対称操作や disorder が複雑な CIF、架橋配位や特殊配位環境では手修正が必要になることがあります。"
)
