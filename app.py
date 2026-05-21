import math
import shlex
from typing import List, Dict, Any, Tuple, Optional

import gemmi
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Σ Value, Coord. bond length Calculator",
    layout="wide"
)

METAL_ELEMENTS = {
    "Fe", "Co", "Mn", "Ni", "Cu", "Zn",
    "Cr", "V", "Ru", "Rh", "Pd", "Pt"
}

DONOR_ELEMENTS = {
    "N", "O", "S", "Cl", "Br", "F"
}

DEFAULT_MAX_CENTERS = 2
DEFAULT_RADIUS = 2.5
DEFAULT_EXPECTED_CN = 6


# ------------------------------------------------
# Basic utilities
# ------------------------------------------------

def clean_label(label: str) -> str:
    return str(label).strip().strip("'").strip('"')


def parse_value_esd(text: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    2.186(19) -> value=2.186, esd=0.019
    89.4(6)   -> value=89.4,  esd=0.6
    """
    raw = str(text).strip()

    if raw in {"?", ".", ""}:
        return None, None, raw

    if "(" not in raw or ")" not in raw:
        try:
            return float(raw), None, raw
        except Exception:
            return None, None, raw

    try:
        main = raw.split("(")[0]
        esd_digits = raw.split("(")[1].split(")")[0]

        value = float(main)

        if "." in main:
            decimals = len(main.split(".")[1])
        else:
            decimals = 0

        esd = int(esd_digits) * (10 ** (-decimals))

        return value, esd, raw

    except Exception:
        return None, None, raw


def format_with_esd(
    value: Optional[float],
    esd: Optional[float],
    decimals: int
) -> str:
    if value is None:
        return "N/A"

    if esd is None:
        return f"{value:.{decimals}f}"

    esd_int = int(round(esd * (10 ** decimals)))

    return f"{value:.{decimals}f}({esd_int})"


def calc_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def propagate_mean_esd(esds: List[Optional[float]], n: int) -> Optional[float]:
    valid = [x for x in esds if x is not None]

    if len(valid) != n or n == 0:
        return None

    return math.sqrt(sum(x * x for x in valid)) / n


def propagate_sum_esd(esds: List[Optional[float]], n: int) -> Optional[float]:
    valid = [x for x in esds if x is not None]

    if len(valid) != n:
        return None

    return math.sqrt(sum(x * x for x in valid))


def angle_deg(
    v1: Tuple[float, float, float],
    v2: Tuple[float, float, float]
) -> float:
    dot = sum(x * y for x, y in zip(v1, v2))

    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(x * x for x in v2))

    if n1 == 0 or n2 == 0:
        return float("nan")

    c = max(-1.0, min(1.0, dot / (n1 * n2)))

    return math.degrees(math.acos(c))


# ------------------------------------------------
# CIF loop parser
# ------------------------------------------------

def split_cif_line(line: str) -> List[str]:
    try:
        return shlex.split(line, posix=False)
    except Exception:
        return line.split()


def parse_cif_loops(text: str) -> List[Dict[str, Any]]:
    lines = text.splitlines()
    loops = []

    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.lower() != "loop_":
            i += 1
            continue

        i += 1
        headers = []

        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            if line.startswith("_"):
                headers.append(line.split()[0])
                i += 1
            else:
                break

        rows = []

        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            lower = line.lower()

            if (
                lower == "loop_"
                or lower.startswith("data_")
                or lower.startswith("save_")
                or line.startswith("_")
            ):
                break

            parts = split_cif_line(line)

            if len(parts) == len(headers):
                rows.append(parts)
            elif len(parts) > len(headers):
                for j in range(0, len(parts), len(headers)):
                    chunk = parts[j:j + len(headers)]
                    if len(chunk) == len(headers):
                        rows.append(chunk)

            i += 1

        loops.append(
            {
                "headers": headers,
                "rows": rows
            }
        )

    return loops


def header_index(headers: List[str], suffix: str) -> Optional[int]:
    suffix = suffix.lower()

    for i, h in enumerate(headers):
        if h.lower().endswith(suffix):
            return i

    return None


def extract_geom_bonds(text: str) -> List[Dict[str, Any]]:
    loops = parse_cif_loops(text)
    bonds = []

    for loop in loops:
        headers = loop["headers"]

        i_a1 = header_index(headers, "_geom_bond_atom_site_label_1")
        i_a2 = header_index(headers, "_geom_bond_atom_site_label_2")
        i_dist = header_index(headers, "_geom_bond_distance")

        if i_a1 is None or i_a2 is None or i_dist is None:
            continue

        i_sym1 = header_index(headers, "_geom_bond_site_symmetry_1")
        i_sym2 = header_index(headers, "_geom_bond_site_symmetry_2")
        i_flag = header_index(headers, "_geom_bond_publ_flag")

        for row in loop["rows"]:
            a1 = clean_label(row[i_a1])
            a2 = clean_label(row[i_a2])
            value, esd, raw = parse_value_esd(row[i_dist])

            bonds.append(
                {
                    "atom1": a1,
                    "atom2": a2,
                    "distance": value,
                    "distance_esd": esd,
                    "distance_raw": raw,
                    "sym1": row[i_sym1] if i_sym1 is not None else "",
                    "sym2": row[i_sym2] if i_sym2 is not None else "",
                    "publ_flag": row[i_flag] if i_flag is not None else "",
                }
            )

    return bonds


def extract_geom_angles(text: str) -> List[Dict[str, Any]]:
    loops = parse_cif_loops(text)
    angles = []

    for loop in loops:
        headers = loop["headers"]

        i_a1 = header_index(headers, "_geom_angle_atom_site_label_1")
        i_a2 = header_index(headers, "_geom_angle_atom_site_label_2")
        i_a3 = header_index(headers, "_geom_angle_atom_site_label_3")
        i_ang = header_index(headers, "_geom_angle")

        if i_a1 is None or i_a2 is None or i_a3 is None or i_ang is None:
            continue

        i_sym1 = header_index(headers, "_geom_angle_site_symmetry_1")
        i_sym2 = header_index(headers, "_geom_angle_site_symmetry_2")
        i_sym3 = header_index(headers, "_geom_angle_site_symmetry_3")
        i_flag = header_index(headers, "_geom_angle_publ_flag")

        for row in loop["rows"]:
            a1 = clean_label(row[i_a1])
            a2 = clean_label(row[i_a2])
            a3 = clean_label(row[i_a3])
            value, esd, raw = parse_value_esd(row[i_ang])

            angles.append(
                {
                    "atom1": a1,
                    "atom2": a2,
                    "atom3": a3,
                    "angle": value,
                    "angle_esd": esd,
                    "angle_raw": raw,
                    "sym1": row[i_sym1] if i_sym1 is not None else "",
                    "sym2": row[i_sym2] if i_sym2 is not None else "",
                    "sym3": row[i_sym3] if i_sym3 is not None else "",
                    "publ_flag": row[i_flag] if i_flag is not None else "",
                }
            )

    return angles


# ------------------------------------------------
# Structure utilities
# ------------------------------------------------

def get_site_dict(
    small: gemmi.SmallStructure
) -> Dict[str, gemmi.SmallStructure.Site]:
    d = {}

    for site in small.sites:
        d[clean_label(site.label)] = site

    return d


def find_metal_sites(
    small: gemmi.SmallStructure,
    max_centers: int
) -> List[gemmi.SmallStructure.Site]:
    centers = []

    for site in small.sites:
        if site.element.name in METAL_ELEMENTS:
            centers.append(site)

    return centers[:max_centers]


def get_cart(
    small: gemmi.SmallStructure,
    site: gemmi.SmallStructure.Site
) -> gemmi.Position:
    return small.cell.orthogonalize(site.fract)


# ------------------------------------------------
# Fallback coordinate-based neighbor search
# ------------------------------------------------

def build_neighbor_candidates_by_coord(
    small: gemmi.SmallStructure,
    center_site: gemmi.SmallStructure.Site,
    radius: float
) -> List[Dict[str, Any]]:
    ns = gemmi.NeighborSearch(
        small,
        radius
    ).populate()

    center_frac = center_site.fract
    center_cart = small.cell.orthogonalize(center_frac)

    candidates = []
    seen = set()

    marks = ns.find_site_neighbors(
        center_site,
        min_dist=0.1,
        max_dist=radius
    )

    for mark in marks:
        site = mark.to_site(small)

        if site.label == center_site.label:
            continue

        if site.element.name not in DONOR_ELEMENTS:
            continue

        fpos = small.cell.fractionalize(mark.pos)

        images = small.cell.find_nearest_pbc_images(
            center_frac,
            radius,
            fpos,
            0
        )

        if not images:
            images = [
                small.cell.find_nearest_pbc_image(
                    center_cart,
                    mark.pos,
                    0
                )
            ]

        for im in images:
            im_frac = small.cell.fract_image(im, fpos)
            im_cart = small.cell.orthogonalize(im_frac)

            dx = im_cart.x - center_cart.x
            dy = im_cart.y - center_cart.y
            dz = im_cart.z - center_cart.z

            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            key = (
                clean_label(site.label),
                round(im_cart.x, 5),
                round(im_cart.y, 5),
                round(im_cart.z, 5),
            )

            if key in seen:
                continue

            seen.add(key)

            candidates.append(
                {
                    "label": clean_label(site.label),
                    "element": site.element.name,
                    "cart": im_cart,
                    "distance": dist,
                    "distance_esd": None,
                    "distance_raw": f"{dist:.4f}",
                    "source": "calculated from coordinates",
                }
            )

    candidates.sort(key=lambda x: x["distance"])

    return candidates


# ------------------------------------------------
# Bond and angle lookup
# ------------------------------------------------

def choose_ligands_from_geom_bond(
    center_label: str,
    site_dict: Dict[str, gemmi.SmallStructure.Site],
    geom_bonds: List[Dict[str, Any]],
    expected_cn: int
) -> List[Dict[str, Any]]:
    ligands = []

    for b in geom_bonds:
        a1 = clean_label(b["atom1"])
        a2 = clean_label(b["atom2"])

        if a1 == center_label:
            lig_label = a2
        elif a2 == center_label:
            lig_label = a1
        else:
            continue

        if lig_label not in site_dict:
            continue

        lig_site = site_dict[lig_label]

        if lig_site.element.name not in DONOR_ELEMENTS:
            continue

        if b["distance"] is None:
            continue

        ligands.append(
            {
                "label": lig_label,
                "element": lig_site.element.name,
                "distance": b["distance"],
                "distance_esd": b["distance_esd"],
                "distance_raw": b["distance_raw"],
                "sym1": b["sym1"],
                "sym2": b["sym2"],
                "publ_flag": b["publ_flag"],
                "cart": get_cart_from_site(lig_site),
                "source": "_geom_bond",
            }
        )

    ligands.sort(key=lambda x: x["distance"])

    chosen = []
    used = set()

    for lig in ligands:
        if lig["label"] in used:
            continue

        chosen.append(lig)
        used.add(lig["label"])

        if len(chosen) == expected_cn:
            return chosen

    return chosen


def get_cart_from_site(site: gemmi.SmallStructure.Site) -> gemmi.Position:
    global CURRENT_CELL
    return CURRENT_CELL.orthogonalize(site.fract)


def find_geom_angle(
    geom_angles: List[Dict[str, Any]],
    atom1: str,
    center: str,
    atom3: str
) -> Optional[Dict[str, Any]]:
    atom1 = clean_label(atom1)
    center = clean_label(center)
    atom3 = clean_label(atom3)

    for a in geom_angles:
        a1 = clean_label(a["atom1"])
        a2 = clean_label(a["atom2"])
        a3 = clean_label(a["atom3"])

        if a2 != center:
            continue

        if (a1 == atom1 and a3 == atom3) or (a1 == atom3 and a3 == atom1):
            return a

    return None


# ------------------------------------------------
# Main center analysis
# ------------------------------------------------

def analyze_center(
    small: gemmi.SmallStructure,
    center_site: gemmi.SmallStructure.Site,
    geom_bonds: List[Dict[str, Any]],
    geom_angles: List[Dict[str, Any]],
    radius: float,
    expected_cn: int,
) -> Optional[Dict[str, Any]]:
    site_dict = get_site_dict(small)

    center_label = clean_label(center_site.label)
    center_cart = get_cart(small, center_site)

    ligands = choose_ligands_from_geom_bond(
        center_label=center_label,
        site_dict=site_dict,
        geom_bonds=geom_bonds,
        expected_cn=expected_cn
    )

    if len(ligands) < expected_cn:
        ligands = build_neighbor_candidates_by_coord(
            small=small,
            center_site=center_site,
            radius=radius
        )[:expected_cn]

    if len(ligands) < expected_cn:
        return None

    bond_values = [
        lig["distance"]
        for lig in ligands
        if lig["distance"] is not None
    ]

    bond_esds = [
        lig.get("distance_esd")
        for lig in ligands
    ]

    mean_bond_length = calc_mean(bond_values)
    mean_bond_esd = propagate_mean_esd(
        bond_esds,
        len(ligands)
    )

    all_angles = []

    for i in range(len(ligands)):
        for j in range(i + 1, len(ligands)):
            lig1 = ligands[i]
            lig2 = ligands[j]

            found = find_geom_angle(
                geom_angles,
                lig1["label"],
                center_label,
                lig2["label"]
            )

            if found is not None and found["angle"] is not None:
                angle_value = found["angle"]
                angle_esd = found["angle_esd"]
                angle_raw = found["angle_raw"]
                source = "_geom_angle"
            else:
                cart1 = lig1["cart"]
                cart2 = lig2["cart"]

                v1 = (
                    cart1.x - center_cart.x,
                    cart1.y - center_cart.y,
                    cart1.z - center_cart.z,
                )

                v2 = (
                    cart2.x - center_cart.x,
                    cart2.y - center_cart.y,
                    cart2.z - center_cart.z,
                )

                angle_value = angle_deg(v1, v2)
                angle_esd = None
                angle_raw = f"{angle_value:.3f}"
                source = "calculated from coordinates"

            delta90 = abs(90.0 - angle_value)

            all_angles.append(
                {
                    "pair": f"{lig1['label']} - {lig2['label']}",
                    "angle": angle_value,
                    "angle_esd": angle_esd,
                    "angle_raw": angle_raw,
                    "delta90": delta90,
                    "source": source,
                }
            )

    sorted_for_sigma = sorted(
        all_angles,
        key=lambda x: x["delta90"]
    )

    cis_used = sorted_for_sigma[:12]
    trans_like = sorted_for_sigma[12:]

    sigma_value = sum(
        x["delta90"]
        for x in cis_used
    )

    sigma_esd = propagate_sum_esd(
        [x["angle_esd"] for x in cis_used],
        len(cis_used)
    )

    return {
        "metal_label": center_label,
        "metal_element": center_site.element.name,

        "mean_bond_length": mean_bond_length,
        "mean_bond_esd": mean_bond_esd,

        "sigma": sigma_value,
        "sigma_esd": sigma_esd,

        "ligands": [
            {
                "label": lig["label"],
                "element": lig["element"],
                "distance": format_with_esd(
                    lig["distance"],
                    lig.get("distance_esd"),
                    3
                ),
                "distance_value_A": round(lig["distance"], 5)
                    if lig["distance"] is not None else None,
                "distance_esd_A": round(lig["distance_esd"], 5)
                    if lig.get("distance_esd") is not None else None,
                "source": lig.get("source", ""),
                "sym1": lig.get("sym1", ""),
                "sym2": lig.get("sym2", ""),
                "publ_flag": lig.get("publ_flag", ""),
            }
            for lig in ligands
        ],

        "cis_angles_used": [
            {
                "pair": x["pair"],
                "angle": format_with_esd(
                    x["angle"],
                    x["angle_esd"],
                    1
                ),
                "angle_value_deg": round(x["angle"], 4),
                "angle_esd_deg": round(x["angle_esd"], 4)
                    if x["angle_esd"] is not None else None,
                "delta90": round(x["delta90"], 4),
                "source": x["source"],
            }
            for x in sorted(cis_used, key=lambda y: y["angle"])
        ],

        "trans_like_angles": [
            {
                "pair": x["pair"],
                "angle": format_with_esd(
                    x["angle"],
                    x["angle_esd"],
                    1
                ),
                "angle_value_deg": round(x["angle"], 4),
                "angle_esd_deg": round(x["angle_esd"], 4)
                    if x["angle_esd"] is not None else None,
                "delta90": round(x["delta90"], 4),
                "source": x["source"],
            }
            for x in sorted(trans_like, key=lambda y: y["angle"])
        ],

        "all_angles": [
            {
                "pair": x["pair"],
                "angle": format_with_esd(
                    x["angle"],
                    x["angle_esd"],
                    1
                ),
                "angle_value_deg": round(x["angle"], 4),
                "angle_esd_deg": round(x["angle_esd"], 4)
                    if x["angle_esd"] is not None else None,
                "delta90": round(x["delta90"], 4),
                "source": x["source"],
            }
            for x in sorted(all_angles, key=lambda y: y["angle"])
        ],
    }


# ------------------------------------------------
# CIF analysis
# ------------------------------------------------

CURRENT_CELL = None


def analyze_cif(
    file_bytes: bytes,
    max_centers: int,
    radius: float,
    expected_cn: int
) -> List[Dict[str, Any]]:
    global CURRENT_CELL

    text = file_bytes.decode(
        "utf-8",
        errors="ignore"
    )

    geom_bonds = extract_geom_bonds(text)
    geom_angles = extract_geom_angles(text)

    doc = gemmi.cif.read_string(text)
    block = doc.sole_block()
    small = gemmi.make_small_structure_from_block(block)

    CURRENT_CELL = small.cell

    centers = find_metal_sites(
        small,
        max_centers
    )

    results = []

    for center_site in centers:
        res = analyze_center(
            small=small,
            center_site=center_site,
            geom_bonds=geom_bonds,
            geom_angles=geom_angles,
            radius=radius,
            expected_cn=expected_cn,
        )

        if res is not None:
            results.append(res)

    return results


# ------------------------------------------------
# UI
# ------------------------------------------------

st.title(
    "Σ Value and Coord. bond length Calculator"
)

st.caption(
    "CIF の _geom_bond / _geom_angle に記録された "
    "2.186(19) 型の値を読み取り、"
    "平均配位結合長と Σ 値を ESD 付きで計算します。"
)

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

    st.markdown(
        "**対象金属**: Fe, Co, Mn, Ni, Cu, Zn, Cr, V, Ru, Rh, Pd, Pt"
    )

    st.markdown(
        "**対象 donor**: N, O, S, Cl, Br, F"
    )

uploaded = st.file_uploader(
    "CIF ファイルを選択",
    type=["cif"]
)

if uploaded is not None:
    try:
        results = analyze_cif(
            uploaded.read(),
            int(max_centers),
            float(radius),
            int(expected_cn)
        )

        if not results:
            st.warning(
                "解析できる金属中心が見つかりませんでした。"
                "近傍探索半径や配位数を見直してください。"
            )

        else:
            st.subheader("計算結果")

            summary_rows = []

            for i, res in enumerate(results, start=1):
                ligand_text = ", ".join(
                    [x["label"] for x in res["ligands"]]
                )

                summary_rows.append(
                    {
                        "Center": i,

                        "Metal":
                            f"{res['metal_label']} "
                            f"({res['metal_element']})",

                        "Σ":
                            format_with_esd(
                                res["sigma"],
                                res["sigma_esd"],
                                1
                            ),

                        "Mean bond length (Å)":
                            format_with_esd(
                                res["mean_bond_length"],
                                res["mean_bond_esd"],
                                3
                            ),

                        "Ligands":
                            ligand_text,
                    }
                )

            summary_df = pd.DataFrame(summary_rows)

            st.dataframe(
                summary_df,
                use_container_width=True
            )

            st.markdown("### 見やすい要約")

            for i, res in enumerate(results, start=1):
                with st.container(border=True):
                    c1, c2, c3 = st.columns(
                        [1.5, 1, 1]
                    )

                    with c1:
                        st.markdown(
                            f"**Center {i}: "
                            f"{res['metal_label']} "
                            f"({res['metal_element']})**"
                        )

                        st.write(
                            "採用配位原子:",
                            ", ".join(
                                [x["label"] for x in res["ligands"]]
                            )
                        )

                    with c2:
                        st.metric(
                            "Σ",
                            format_with_esd(
                                res["sigma"],
                                res["sigma_esd"],
                                1
                            )
                        )

                    with c3:
                        st.metric(
                            "平均結合長 (Å)",
                            format_with_esd(
                                res["mean_bond_length"],
                                res["mean_bond_esd"],
                                3
                            )
                        )

            st.markdown("### 詳細")

            for i, res in enumerate(results, start=1):
                with st.expander(
                    f"Center {i}: "
                    f"{res['metal_label']} "
                    f"({res['metal_element']}) の詳細",
                    expanded=False
                ):
                    st.markdown("#### 採用した配位結合")

                    st.dataframe(
                        pd.DataFrame(res["ligands"]),
                        use_container_width=True
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Σ に使用した 12 角")

                        st.dataframe(
                            pd.DataFrame(res["cis_angles_used"]),
                            use_container_width=True
                        )

                    with col2:
                        st.markdown("#### 残り 3 角")

                        st.dataframe(
                            pd.DataFrame(res["trans_like_angles"]),
                            use_container_width=True
                        )

                    st.markdown("#### 全 15 角")

                    st.dataframe(
                        pd.DataFrame(res["all_angles"]),
                        use_container_width=True
                    )

            csv_rows = []

            for i, res in enumerate(results, start=1):
                csv_rows.append(
                    {
                        "center": i,
                        "metal_label": res["metal_label"],
                        "metal_element": res["metal_element"],

                        "sigma": res["sigma"],
                        "sigma_esd": res["sigma_esd"],
                        "sigma_formatted": format_with_esd(
                            res["sigma"],
                            res["sigma_esd"],
                            1
                        ),

                        "mean_bond_length_A": res["mean_bond_length"],
                        "mean_bond_length_esd_A": res["mean_bond_esd"],
                        "mean_bond_length_formatted": format_with_esd(
                            res["mean_bond_length"],
                            res["mean_bond_esd"],
                            3
                        ),

                        "ligands": "; ".join(
                            [x["label"] for x in res["ligands"]]
                        ),

                        "bond_lengths": "; ".join(
                            [x["distance"] for x in res["ligands"]]
                        ),
                    }
                )

            csv_data = pd.DataFrame(
                csv_rows
            ).to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "CSV をダウンロード",
                csv_data,
                file_name="sigma_results.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(
            f"解析に失敗しました: {e}"
        )

else:
    st.info(
        "CIF ファイルをアップロードしてください。"
    )

st.markdown("---")

st.markdown(
    "平均配位結合長の ESD は、CIF 中の各 _geom_bond_distance の ESD から "
    "sqrt(esd1^2 + esd2^2 + ... + esdn^2) / n として伝播しています。"
)

st.markdown(
    "Σ の ESD は、Σ に使用した 12 個の _geom_angle の ESD から "
    "sqrt(esd1^2 + esd2^2 + ... + esd12^2) として伝播しています。"
)

st.markdown(
    "Copyright © 2026 Yu ODASHIMA All Rights Reserved."
)
