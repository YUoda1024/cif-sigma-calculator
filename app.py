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

    return (
        str(label)
        .strip()
        .strip("'")
        .strip('"')
    )


def parse_value_esd(
    text: str
) -> Tuple[Optional[float], Optional[float], str]:

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

        esd_digits = (
            raw
            .split("(")[1]
            .split(")")[0]
        )

        value = float(main)

        if "." in main:
            decimals = len(
                main.split(".")[1]
            )
        else:
            decimals = 0

        esd = (
            int(esd_digits)
            * (10 ** (-decimals))
        )

        return value, esd, raw

    except Exception:

        return None, None, raw


def format_with_esd(
    value: Optional[float],
    esd: Optional[float]
) -> str:

    if value is None:
        return "N/A"

    if esd is None:
        return str(value)

    if esd == 0:
        return str(value)

    exponent = math.floor(
        math.log10(abs(esd))
    )

    scaled = esd / (10 ** exponent)

    if scaled < 2:
        sig_digits = 2
    else:
        sig_digits = 1

    esd_rounded = round(
        esd,
        -exponent + (sig_digits - 1)
    )

    decimals = max(
        0,
        -math.floor(math.log10(esd_rounded))
        + (sig_digits - 1)
    )

    value_rounded = round(
        value,
        decimals
    )

    esd_int = int(
        round(
            esd_rounded
            * (10 ** decimals)
        )
    )

    return (
        f"{value_rounded:.{decimals}f}"
        f"({esd_int})"
    )


def calc_mean(
    values: List[float]
) -> Optional[float]:

    if not values:
        return None

    return sum(values) / len(values)


def propagate_mean_esd(
    esds: List[Optional[float]],
    n: int
) -> Optional[float]:

    valid = [
        x for x in esds
        if x is not None
    ]

    if len(valid) != n or n == 0:
        return None

    return (
        math.sqrt(
            sum(x * x for x in valid)
        ) / n
    )


def propagate_sum_esd(
    esds: List[Optional[float]],
    n: int
) -> Optional[float]:

    valid = [
        x for x in esds
        if x is not None
    ]

    if len(valid) != n:
        return None

    return math.sqrt(
        sum(x * x for x in valid)
    )


def angle_deg(
    v1: Tuple[float, float, float],
    v2: Tuple[float, float, float]
) -> float:

    dot = sum(
        x * y
        for x, y in zip(v1, v2)
    )

    n1 = math.sqrt(
        sum(x * x for x in v1)
    )

    n2 = math.sqrt(
        sum(x * x for x in v2)
    )

    if n1 == 0 or n2 == 0:
        return float("nan")

    c = max(
        -1.0,
        min(1.0, dot / (n1 * n2))
    )

    return math.degrees(
        math.acos(c)
    )


# ------------------------------------------------
# CIF loop parser
# ------------------------------------------------

def split_cif_line(
    line: str
) -> List[str]:

    try:
        return shlex.split(
            line,
            posix=False
        )

    except Exception:
        return line.split()


def parse_cif_loops(
    text: str
) -> List[Dict[str, Any]]:

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

                headers.append(
                    line.split()[0]
                )

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

                for j in range(
                    0,
                    len(parts),
                    len(headers)
                ):

                    chunk = parts[
                        j:j + len(headers)
                    ]

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


def header_index(
    headers: List[str],
    suffix: str
) -> Optional[int]:

    suffix = suffix.lower()

    for i, h in enumerate(headers):

        if h.lower().endswith(suffix):
            return i

    return None


# ------------------------------------------------
# geom_bond parser
# ------------------------------------------------

def extract_geom_bonds(
    text: str
) -> List[Dict[str, Any]]:

    loops = parse_cif_loops(text)

    bonds = []

    for loop in loops:

        headers = loop["headers"]

        i_a1 = header_index(
            headers,
            "_geom_bond_atom_site_label_1"
        )

        i_a2 = header_index(
            headers,
            "_geom_bond_atom_site_label_2"
        )

        i_dist = header_index(
            headers,
            "_geom_bond_distance"
        )

        if (
            i_a1 is None
            or i_a2 is None
            or i_dist is None
        ):
            continue

        i_sym1 = header_index(
            headers,
            "_geom_bond_site_symmetry_1"
        )

        i_sym2 = header_index(
            headers,
            "_geom_bond_site_symmetry_2"
        )

        i_flag = header_index(
            headers,
            "_geom_bond_publ_flag"
        )

        for row in loop["rows"]:

            a1 = clean_label(
                row[i_a1]
            )

            a2 = clean_label(
                row[i_a2]
            )

            value, esd, raw = parse_value_esd(
                row[i_dist]
            )

            bonds.append(
                {
                    "atom1": a1,
                    "atom2": a2,

                    "distance": value,
                    "distance_esd": esd,
                    "distance_raw": raw,

                    "sym1":
                        row[i_sym1]
                        if i_sym1 is not None
                        else "",

                    "sym2":
                        row[i_sym2]
                        if i_sym2 is not None
                        else "",

                    "publ_flag":
                        row[i_flag]
                        if i_flag is not None
                        else "",
                }
            )

    return bonds


# ------------------------------------------------
# geom_angle parser
# ------------------------------------------------

def extract_geom_angles(
    text: str
) -> List[Dict[str, Any]]:

    loops = parse_cif_loops(text)

    angles = []

    for loop in loops:

        headers = loop["headers"]

        i_a1 = header_index(
            headers,
            "_geom_angle_atom_site_label_1"
        )

        i_a2 = header_index(
            headers,
            "_geom_angle_atom_site_label_2"
        )

        i_a3 = header_index(
            headers,
            "_geom_angle_atom_site_label_3"
        )

        i_ang = header_index(
            headers,
            "_geom_angle"
        )

        if (
            i_a1 is None
            or i_a2 is None
            or i_a3 is None
            or i_ang is None
        ):
            continue

        for row in loop["rows"]:

            value, esd, raw = parse_value_esd(
                row[i_ang]
            )

            angles.append(
                {
                    "atom1":
                        clean_label(row[i_a1]),

                    "atom2":
                        clean_label(row[i_a2]),

                    "atom3":
                        clean_label(row[i_a3]),

                    "angle": value,
                    "angle_esd": esd,
                    "angle_raw": raw,
                }
            )

    return angles


# ------------------------------------------------
# Structure helpers
# ------------------------------------------------

def get_site_dict(
    small: gemmi.SmallStructure
) -> Dict[str, gemmi.SmallStructure.Site]:

    d = {}

    for site in small.sites:

        d[
            clean_label(site.label)
        ] = site

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


# ------------------------------------------------
# Main analysis
# ------------------------------------------------

def analyze_center(
    center_site: gemmi.SmallStructure.Site,
    site_dict: Dict[str, gemmi.SmallStructure.Site],
    geom_bonds: List[Dict[str, Any]],
    geom_angles: List[Dict[str, Any]],
    expected_cn: int,
) -> Optional[Dict[str, Any]]:

    center_label = clean_label(
        center_site.label
    )

    ligands = []

    for b in geom_bonds:

        a1 = clean_label(b["atom1"])
        a2 = clean_label(b["atom2"])

        if a1 == center_label:
            lig = a2

        elif a2 == center_label:
            lig = a1

        else:
            continue

        if lig not in site_dict:
            continue

        lig_site = site_dict[lig]

        if lig_site.element.name not in DONOR_ELEMENTS:
            continue

        if b["distance"] is None:
            continue

        ligands.append(
            {
                "label": lig,
                "element":
                    lig_site.element.name,

                "distance":
                    b["distance"],

                "distance_esd":
                    b["distance_esd"],

                "distance_raw":
                    b["distance_raw"],
            }
        )

    ligands.sort(
        key=lambda x: x["distance"]
    )

    used = set()

    chosen = []

    for lig in ligands:

        if lig["label"] in used:
            continue

        chosen.append(lig)

        used.add(lig["label"])

        if len(chosen) == expected_cn:
            break

    if len(chosen) < expected_cn:
        return None

    bond_values = [
        x["distance"]
        for x in chosen
    ]

    bond_esds = [
        x["distance_esd"]
        for x in chosen
    ]

    mean_bond = calc_mean(
        bond_values
    )

    mean_bond_esd = propagate_mean_esd(
        bond_esds,
        len(chosen)
    )

    all_angles = []

    for i in range(len(chosen)):

        for j in range(i + 1, len(chosen)):

            lig1 = chosen[i]["label"]
            lig2 = chosen[j]["label"]

            found = None

            for a in geom_angles:

                a1 = clean_label(a["atom1"])
                a2 = clean_label(a["atom2"])
                a3 = clean_label(a["atom3"])

                if a2 != center_label:
                    continue

                if (
                    (a1 == lig1 and a3 == lig2)
                    or
                    (a1 == lig2 and a3 == lig1)
                ):
                    found = a
                    break

            if found is None:
                continue

            delta90 = abs(
                90.0 - found["angle"]
            )

            all_angles.append(
                {
                    "pair":
                        f"{lig1} - {lig2}",

                    "angle":
                        found["angle"],

                    "angle_esd":
                        found["angle_esd"],

                    "angle_raw":
                        found["angle_raw"],

                    "delta90":
                        delta90,
                }
            )

    sorted_angles = sorted(
        all_angles,
        key=lambda x: x["delta90"]
    )

    cis_used = sorted_angles[:12]

    trans_like = sorted_angles[12:]

    sigma = sum(
        x["delta90"]
        for x in cis_used
    )

    sigma_esd = propagate_sum_esd(
        [
            x["angle_esd"]
            for x in cis_used
        ],
        len(cis_used)
    )

    return {

        "metal_label":
            center_label,

        "metal_element":
            center_site.element.name,

        "mean_bond":
            mean_bond,

        "mean_bond_esd":
            mean_bond_esd,

        "sigma":
            sigma,

        "sigma_esd":
            sigma_esd,

        "ligands": [
            {
                "label":
                    lig["label"],

                "element":
                    lig["element"],

                "distance":
                    lig["distance_raw"],

                "distance_value":
                    lig["distance"],

                "distance_esd":
                    lig["distance_esd"],
            }
            for lig in chosen
        ],

        "cis_angles": [
            {
                "pair":
                    x["pair"],

                "angle":
                    x["angle_raw"],

                "delta90":
                    round(
                        x["delta90"],
                        4
                    ),
            }
            for x in cis_used
        ],

        "trans_angles": [
            {
                "pair":
                    x["pair"],

                "angle":
                    x["angle_raw"],

                "delta90":
                    round(
                        x["delta90"],
                        4
                    ),
            }
            for x in trans_like
        ],

        "all_angles": [
            {
                "pair":
                    x["pair"],

                "angle":
                    x["angle_raw"],

                "delta90":
                    round(
                        x["delta90"],
                        4
                    ),
            }
            for x in all_angles
        ],
    }


def analyze_cif(
    file_bytes: bytes,
    max_centers: int,
    expected_cn: int
) -> List[Dict[str, Any]]:

    text = file_bytes.decode(
        "utf-8",
        errors="ignore"
    )

    geom_bonds = extract_geom_bonds(
        text
    )

    geom_angles = extract_geom_angles(
        text
    )

    doc = gemmi.cif.read_string(
        text
    )

    block = doc.sole_block()

    small = gemmi.make_small_structure_from_block(
        block
    )

    site_dict = get_site_dict(
        small
    )

    centers = find_metal_sites(
        small,
        max_centers
    )

    results = []

    for center_site in centers:

        res = analyze_center(
            center_site=center_site,
            site_dict=site_dict,
            geom_bonds=geom_bonds,
            geom_angles=geom_angles,
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
    "CIF の _geom_bond / _geom_angle "
    "から ESD を読み取って "
    "平均配位結合長と Σ を計算します。"
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

    expected_cn = st.number_input(
        "想定配位数",
        min_value=4,
        max_value=8,
        value=DEFAULT_EXPECTED_CN,
        step=1
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
            int(expected_cn)
        )

        if not results:

            st.warning(
                "解析できる金属中心が "
                "見つかりませんでした。"
            )

        else:

            st.subheader("計算結果")

            summary_rows = []

            for i, res in enumerate(results, start=1):

                ligand_text = ", ".join(
                    [
                        x["label"]
                        for x in res["ligands"]
                    ]
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
                                res["sigma_esd"]
                            ),

                        "Mean bond length (Å)":
                            format_with_esd(
                                res["mean_bond"],
                                res["mean_bond_esd"]
                            ),

                        "Ligands":
                            ligand_text,
                    }
                )

            summary_df = pd.DataFrame(
                summary_rows
            )

            st.dataframe(
                summary_df,
                use_container_width=True
            )

            st.markdown(
                "### 見やすい要約"
            )

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
                                [
                                    x["label"]
                                    for x in res["ligands"]
                                ]
                            )
                        )

                    with c2:

                        st.metric(
                            "Σ",
                            format_with_esd(
                                res["sigma"],
                                res["sigma_esd"]
                            )
                        )

                    with c3:

                        st.metric(
                            "平均結合長 (Å)",
                            format_with_esd(
                                res["mean_bond"],
                                res["mean_bond_esd"]
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

                    st.markdown(
                        "#### 採用した配位結合"
                    )

                    st.dataframe(
                        pd.DataFrame(
                            res["ligands"]
                        ),
                        use_container_width=True
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.markdown(
                            "#### Σ に使用した 12 角"
                        )

                        st.dataframe(
                            pd.DataFrame(
                                res["cis_angles"]
                            ),
                            use_container_width=True
                        )

                    with col2:

                        st.markdown(
                            "#### 残り 3 角"
                        )

                        st.dataframe(
                            pd.DataFrame(
                                res["trans_angles"]
                            ),
                            use_container_width=True
                        )

                    st.markdown(
                        "#### 全 15 角"
                    )

                    st.dataframe(
                        pd.DataFrame(
                            res["all_angles"]
                        ),
                        use_container_width=True
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
    "Copyright © 2026 "
    "Yu ODASHIMA All Rights Reserved."
)
