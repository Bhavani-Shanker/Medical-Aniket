import math
import itertools

import streamlit as st
import numpy as np
import pandas as pd

from scipy import __version__ as scipy_version
from scipy.stats import (
    fisher_exact,
    chi2_contingency,
    f_oneway,
    ttest_ind
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Statistical Test Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 600;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 Statistical Test Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Fisher Exact • Fisher–Freeman–Halton • Chi-square • '
    'ANOVA • t-test'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

alpha = st.sidebar.selectbox(
    "Significance Level (α)",
    [0.01, 0.05, 0.10],
    index=1
)

st.sidebar.divider()

st.sidebar.markdown("### Software")

st.sidebar.write(
    f"**Python:** {'.'.join(map(str, __import__('sys').version_info[:3]))}"
)

st.sidebar.write(
    f"**NumPy:** {np.__version__}"
)

st.sidebar.write(
    f"**Pandas:** {pd.__version__}"
)

st.sidebar.write(
    f"**SciPy:** {scipy_version}"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def combination(n, k):
    """
    Calculate n choose k.
    """

    if k < 0 or k > n:
        return 0

    return math.comb(int(n), int(k))


def fisher_table_probability(
    top_row,
    column_totals
):
    """
    Probability of a 2 x C table conditional
    on fixed row and column margins.

    P(T) =
        Product[ C(column_total_j, x_j) ]
        ---------------------------------
             C(N, row_total_1)
    """

    top_row = tuple(int(x) for x in top_row)
    column_totals = tuple(
        int(x) for x in column_totals
    )

    total = sum(column_totals)
    row1_total = sum(top_row)

    numerator = 1

    for x, column_total in zip(
        top_row,
        column_totals
    ):
        numerator *= combination(
            column_total,
            x
        )

    denominator = combination(
        total,
        row1_total
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


def enumerate_2xc_tables(
    column_totals,
    row1_total
):
    """
    Enumerate all possible 2 x C tables
    with fixed row and column margins.

    Returns:
        List of dictionaries containing:
        - top row
        - bottom row
        - probability
    """

    column_totals = tuple(
        int(x)
        for x in column_totals
    )

    number_columns = len(
        column_totals
    )

    results = []

    # --------------------------------------------------------
    # Enumerate all columns except final column.
    #
    # The final column is calculated from the fixed
    # first-row total.
    # --------------------------------------------------------

    ranges = [
        range(column_total + 1)
        for column_total
        in column_totals[:-1]
    ]

    for values in itertools.product(*ranges):

        final_value = (
            row1_total - sum(values)
        )

        # Final value must be valid
        if final_value < 0:
            continue

        if final_value > column_totals[-1]:
            continue

        top_row = (
            tuple(values)
            + (final_value,)
        )

        # Validate every first-row cell
        valid = True

        for x, column_total in zip(
            top_row,
            column_totals
        ):

            if x < 0 or x > column_total:
                valid = False
                break

        if not valid:
            continue

        bottom_row = tuple(
            column_total - x
            for column_total, x
            in zip(
                column_totals,
                top_row
            )
        )

        probability = fisher_table_probability(
            top_row,
            column_totals
        )

        results.append(
            {
                "top": top_row,
                "bottom": bottom_row,
                "probability": probability
            }
        )

    return results


def validate_contingency_table(table):
    """
    Validate a contingency table.
    """

    table = np.asarray(table)

    if table.ndim != 2:

        return False, (
            "The table must be two-dimensional."
        )

    if table.shape[0] < 2:

        return False, (
            "At least 2 rows are required."
        )

    if table.shape[1] < 2:

        return False, (
            "At least 2 columns are required."
        )

    if np.any(table < 0):

        return False, (
            "Counts cannot be negative."
        )

    if not np.all(
        np.equal(
            table,
            np.floor(table)
        )
    ):

        return False, (
            "Contingency table values must "
            "be integers."
        )

    if table.sum() == 0:

        return False, (
            "The table cannot contain only zeros."
        )

    if np.any(table.sum(axis=1) == 0):

        return False, (
            "Every row must have a positive total."
        )

    if np.any(table.sum(axis=0) == 0):

        return False, (
            "Every column must have a positive total."
        )

    return True, ""


def interpretation(
    p_value,
    alpha
):
    """
    Generate statistical interpretation.
    """

    if p_value < alpha:

        return (
            f"Statistically significant: "
            f"p-value = {p_value:.8g} < "
            f"α = {alpha}. Reject the null hypothesis."
        )

    return (
        f"Not statistically significant: "
        f"p-value = {p_value:.8g} ≥ "
        f"α = {alpha}. Fail to reject "
        f"the null hypothesis."
    )


# ============================================================
# FISHER EXACT / FISHER-FREEMAN-HALTON
# ============================================================

def run_fisher_analysis(table):
    """
    Fisher analysis.

    2x2:
        Standard Fisher Exact Test using SciPy.

    2xC:
        Exact Fisher-Freeman-Halton enumeration.

    Returns:
        result dictionary
    """

    table = np.asarray(
        table,
        dtype=int
    )

    rows, cols = table.shape

    # --------------------------------------------------------
    # Standard Fisher exact test
    # --------------------------------------------------------

    if rows == 2 and cols == 2:

        result = fisher_exact(
            table,
            alternative="two-sided"
        )

        observed_probability = (
            fisher_table_probability(
                tuple(table[0]),
                tuple(table.sum(axis=0))
            )
        )

        return {
            "test": "Fisher's Exact Test",
            "statistic_name": "Odds Ratio",
            "statistic": float(
                result.statistic
            ),
            "p_value": float(
                result.pvalue
            ),
            "observed_probability":
                observed_probability,
            "tables": None,
            "exact_enumeration": False
        }

    # --------------------------------------------------------
    # Fisher-Freeman-Halton
    #
    # Exact enumeration currently implemented for 2 x C.
    # --------------------------------------------------------

    if rows != 2:

        raise ValueError(
            "Exact Fisher-Freeman-Halton enumeration "
            "in this application currently supports "
            "2 × C tables. For R × C tables with "
            "more than 2 rows, use the Chi-square test "
            "or a Monte Carlo implementation."
        )

    column_totals = tuple(
        int(x)
        for x in table.sum(axis=0)
    )

    row1_total = int(
        table[0].sum()
    )

    observed_top = tuple(
        int(x)
        for x in table[0]
    )

    observed_probability = (
        fisher_table_probability(
            observed_top,
            column_totals
        )
    )

    # --------------------------------------------------------
    # Enumerate all possible tables
    # --------------------------------------------------------

    possible_tables = enumerate_2xc_tables(
        column_totals,
        row1_total
    )

    # --------------------------------------------------------
    # Determine which tables contribute to
    # two-sided Fisher-Freeman-Halton p-value
    #
    # Probability ordering:
    #
    # P(table) <= P(observed table)
    # --------------------------------------------------------

    tolerance = 1e-12

    for item in possible_tables:

        item["included"] = (
            item["probability"]
            <= observed_probability
            + tolerance
        )

        item["is_observed"] = (
            item["top"] == observed_top
        )

    exact_p_value = sum(
        item["probability"]
        for item in possible_tables
        if item["included"]
    )

    # --------------------------------------------------------
    # Sort by probability descending
    # --------------------------------------------------------

    possible_tables.sort(
        key=lambda x: x["probability"],
        reverse=True
    )

    return {
        "test":
            "Fisher–Freeman–Halton Exact Test",

        "statistic_name":
            "Observed Table Probability",

        "statistic":
            observed_probability,

        "p_value":
            exact_p_value,

        "observed_probability":
            observed_probability,

        "tables":
            possible_tables,

        "exact_enumeration":
            True
    }


# ============================================================
# TEST SELECTION
# ============================================================

test = st.selectbox(
    "Select Statistical Test",
    [
        "Fisher / Fisher–Freeman–Halton",
        "Chi-square",
        "One-way ANOVA",
        "Independent Samples t-test"
    ]
)

st.divider()


# ============================================================
# FISHER / FISHER-FREEMAN-HALTON UI
# ============================================================

if test == "Fisher / Fisher–Freeman–Halton":

    st.header(
        "🔬 Fisher Exact / Fisher–Freeman–Halton"
    )

    st.info(
        """
        **2×2:** Standard Fisher's Exact Test.

        **2×C:** Exact Fisher–Freeman–Halton Test.

        For the 2×C case, all possible tables with the
        same row and column margins are enumerated.
        """
    )

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        rows = st.number_input(
            "Rows",
            min_value=2,
            max_value=2,
            value=2,
            step=1
        )

    with col2:

        cols = st.number_input(
            "Columns",
            min_value=2,
            max_value=8,
            value=4,
            step=1
        )

    st.subheader(
        f"Enter Observed {rows} × {cols} Table"
    )

    # --------------------------------------------------------
    # Input table
    # --------------------------------------------------------

    data = []

    header = st.columns(
        cols + 1
    )

    header[0].markdown(
        "**Row**"
    )

    for j in range(cols):

        header[j + 1].markdown(
            f"**C{j + 1}**"
        )

    for i in range(rows):

        input_cols = st.columns(
            cols + 1
        )

        input_cols[0].markdown(
            f"**R{i + 1}**"
        )

        row = []

        for j in range(cols):

            value = input_cols[
                j + 1
            ].number_input(
                f"R{i+1}C{j+1}",
                min_value=0,
                value=0,
                step=1,
                key=f"fisher_{i}_{j}",
                label_visibility="collapsed"
            )

            row.append(
                int(value)
            )

        data.append(row)

    table_df = pd.DataFrame(
        data,
        index=[
            f"Row {i+1}"
            for i in range(rows)
        ],
        columns=[
            f"Column {j+1}"
            for j in range(cols)
        ]
    )

    st.subheader(
        "Observed Contingency Table"
    )

    st.dataframe(
        table_df,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Calculate
    # --------------------------------------------------------

    if st.button(
        "🔬 Calculate Fisher Test",
        type="primary",
        use_container_width=True
    ):

        valid, message = (
            validate_contingency_table(
                table_df.values
            )
        )

        if not valid:

            st.error(message)

            st.stop()

        try:

            with st.spinner(
                "Calculating..."
            ):

                result = run_fisher_analysis(
                    table_df.values
                )

            # ==================================================
            # RESULTS
            # ==================================================

            st.success(
                "Calculation completed."
            )

            st.header(
                "1. Test Results"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Test",
                    result["test"]
                )

            with col2:

                st.metric(
                    result["statistic_name"],
                    f"{result['statistic']:.12f}"
                )

            with col3:

                st.metric(
                    "P-value",
                    f"{result['p_value']:.12f}"
                )

            # ------------------------------------------------
            # Interpretation
            # ------------------------------------------------

            if result["p_value"] < alpha:

                st.success(
                    interpretation(
                        result["p_value"],
                        alpha
                    )
                )

            else:

                st.warning(
                    interpretation(
                        result["p_value"],
                        alpha
                    )
                )

            # ==================================================
            # EXACT ENUMERATION
            # ==================================================

            if result["exact_enumeration"]:

                possible_tables = (
                    result["tables"]
                )

                st.header(
                    "2. Exact Enumeration"
                )

                total_tables = len(
                    possible_tables
                )

                included_tables = sum(
                    item["included"]
                    for item in possible_tables
                )

                excluded_tables = (
                    total_tables
                    - included_tables
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "All Possible Tables",
                        f"{total_tables:,}"
                    )

                with col2:

                    st.metric(
                        "Tables Included",
                        f"{included_tables:,}"
                    )

                with col3:

                    st.metric(
                        "Tables Excluded",
                        f"{excluded_tables:,}"
                    )

                # ==================================================
                # MARGINS
                # ==================================================

                st.header(
                    "3. Fixed Margins"
                )

                observed_array = (
                    table_df.values
                )

                row_totals = (
                    observed_array.sum(axis=1)
                )

                column_totals = (
                    observed_array.sum(axis=0)
                )

                margin_col1, margin_col2 = (
                    st.columns(2)
                )

                with margin_col1:

                    st.write(
                        "**Row Totals**"
                    )

                    row_df = pd.DataFrame(
                        {
                            "Row":
                                [
                                    f"Row {i+1}"
                                    for i in range(rows)
                                ],

                            "Total":
                                row_totals
                        }
                    )

                    st.dataframe(
                        row_df,
                        use_container_width=True,
                        hide_index=True
                    )

                with margin_col2:

                    st.write(
                        "**Column Totals**"
                    )

                    column_df = pd.DataFrame(
                        {
                            "Column":
                                [
                                    f"Column {i+1}"
                                    for i in range(cols)
                                ],

                            "Total":
                                column_totals
                        }
                    )

                    st.dataframe(
                        column_df,
                        use_container_width=True,
                        hide_index=True
                    )

                # ==================================================
                # OBSERVED TABLE PROBABILITY
                # ==================================================

                st.header(
                    "4. Observed Table Probability"
                )

                st.latex(
                    r"""
                    P(T) =
                    \frac{
                    \prod_j {c_j \choose x_j}
                    }{
                    {N \choose R_1}
                    }
                    """
                )

                st.metric(
                    "Observed Table Probability",
                    f"{result['observed_probability']:.12f}"
                )

                st.write(
                    """
                    The observed table probability is the
                    probability of obtaining the exact observed
                    table, conditional on the observed row and
                    column margins.
                    """
                )

                # ==================================================
                # ALL TABLES
                # ==================================================

                st.header(
                    f"5. All {total_tables:,} Possible Tables"
                )

                st.write(
                    """
                    Each table below has exactly the same
                    row totals and column totals as the observed
                    table.
                    """
                )

                display_rows = []

                for number, item in enumerate(
                    possible_tables,
                    start=1
                ):

                    row_data = {
                        "Table #":
                            number
                    }

                    for j, value in enumerate(
                        item["top"],
                        start=1
                    ):

                        row_data[
                            f"R1C{j}"
                        ] = value

                    for j, value in enumerate(
                        item["bottom"],
                        start=1
                    ):

                        row_data[
                            f"R2C{j}"
                        ] = value

                    row_data[
                        "Probability"
                    ] = item["probability"]

                    row_data[
                        "≤ Observed?"
                    ] = (
                        "YES"
                        if item["included"]
                        else "NO"
                    )

                    row_data[
                        "Observed?"
                    ] = (
                        "YES"
                        if item["is_observed"]
                        else "NO"
                    )

                    display_rows.append(
                        row_data
                    )

                all_tables_df = pd.DataFrame(
                    display_rows
                )

                st.dataframe(
                    all_tables_df.style.format(
                        {
                            "Probability":
                                "{:.12f}"
                        }
                    ),
                    use_container_width=True,
                    height=650
                )

                # ==================================================
                # P-VALUE CALCULATION
                # ==================================================

                st.header(
                    "6. Exact P-value Calculation"
                )

                st.latex(
                    r"""
                    p =
                    \sum_{
                    P(T) \leq P(T_{observed})
                    }
                    P(T)
                    """
                )

                included_probability = sum(
                    item["probability"]
                    for item in possible_tables
                    if item["included"]
                )

                st.write(
                    f"""
                    Observed table probability:

                    **{result['observed_probability']:.12f}**

                    Number of tables included in the
                    two-sided p-value:

                    **{included_tables:,}**

                    Sum of their probabilities:

                    **{included_probability:.12f}**

                    Therefore:

                    **Exact P-value = {result['p_value']:.12f}**
                    """
                )

                # ==================================================
                # PROBABILITY VALIDATION
                # ==================================================

                st.header(
                    "7. Probability Validation"
                )

                total_probability = sum(
                    item["probability"]
                    for item in possible_tables
                )

                st.metric(
                    "Sum of All Table Probabilities",
                    f"{total_probability:.12f}"
                )

                if abs(
                    total_probability - 1.0
                ) < 1e-10:

                    st.success(
                        "✓ Validation passed: "
                        "all possible table probabilities "
                        "sum to 1."
                    )

                else:

                    st.warning(
                        "The probability sum differs from 1. "
                        "This may indicate numerical precision "
                        "issues."
                    )

                # ==================================================
                # DOWNLOAD
                # ==================================================

                st.header(
                    "8. Download Results"
                )

                csv_data = (
                    all_tables_df
                    .to_csv(index=False)
                    .encode("utf-8")
                )

                st.download_button(
                    label="⬇️ Download All Tables CSV",
                    data=csv_data,
                    file_name=(
                        "fisher_freeman_halton_tables.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Calculation error: {e}"
            )


# ============================================================
# CHI-SQUARE
# ============================================================

elif test == "Chi-square":

    st.header(
        "📐 Chi-square Test of Independence"
    )

    st.write(
        "Chi-square can be applied to any R × C "
        "contingency table."
    )

    col1, col2 = st.columns(2)

    with col1:

        rows = st.number_input(
            "Number of Rows",
            min_value=2,
            max_value=20,
            value=4,
            step=1,
            key="chi_rows"
        )

    with col2:

        cols = st.number_input(
            "Number of Columns",
            min_value=2,
            max_value=20,
            value=2,
            step=1,
            key="chi_cols"
        )

    st.subheader(
        f"Enter {rows} × {cols} Table"
    )

    data = []

    header = st.columns(
        cols + 1
    )

    header[0].markdown(
        "**Row**"
    )

    for j in range(cols):

        header[j + 1].markdown(
            f"**C{j+1}**"
        )

    for i in range(rows):

        input_cols = st.columns(
            cols + 1
        )

        input_cols[0].markdown(
            f"**R{i+1}**"
        )

        row = []

        for j in range(cols):

            value = input_cols[
                j + 1
            ].number_input(
                f"R{i+1}C{j+1}",
                min_value=0,
                value=0,
                step=1,
                key=f"chi_{i}_{j}",
                label_visibility="collapsed"
            )

            row.append(int(value))

        data.append(row)

    table_df = pd.DataFrame(
        data,
        index=[
            f"Row {i+1}"
            for i in range(rows)
        ],
        columns=[
            f"Column {j+1}"
            for j in range(cols)
        ]
    )

    st.subheader(
        "Observed Table"
    )

    st.dataframe(
        table_df,
        use_container_width=True
    )

    if st.button(
        "📐 Calculate Chi-square",
        type="primary",
        use_container_width=True
    ):

        valid, message = (
            validate_contingency_table(
                table_df.values
            )
        )

        if not valid:

            st.error(message)

        else:

            try:

                chi2_stat, p_value, df, expected = (
                    chi2_contingency(
                        table_df.values
                    )
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Chi-square",
                        f"{chi2_stat:.8f}"
                    )

                with col2:

                    st.metric(
                        "Degrees of Freedom",
                        int(df)
                    )

                with col3:

                    st.metric(
                        "P-value",
                        f"{p_value:.8f}"
                    )

                st.subheader(
                    "Expected Frequencies"
                )

                expected_df = pd.DataFrame(
                    expected,
                    index=table_df.index,
                    columns=table_df.columns
                )

                st.dataframe(
                    expected_df.style.format(
                        "{:.4f}"
                    ),
                    use_container_width=True
                )

                if np.any(
                    expected < 5
                ):

                    st.warning(
                        "Some expected frequencies are below 5. "
                        "Consider an exact or resampling method."
                    )

                if p_value < alpha:

                    st.success(
                        interpretation(
                            p_value,
                            alpha
                        )
                    )

                else:

                    st.warning(
                        interpretation(
                            p_value,
                            alpha
                        )
                    )

            except Exception as e:

                st.error(
                    f"Chi-square calculation error: {e}"
                )


# ============================================================
# ANOVA
# ============================================================

elif test == "One-way ANOVA":

    st.header(
        "📈 One-way ANOVA"
    )

    st.write(
        "Compare the means of two or more independent "
        "numerical groups."
    )

    number_groups = st.number_input(
        "Number of Groups",
        min_value=2,
        max_value=20,
        value=3,
        step=1
    )

    groups = []

    for i in range(number_groups):

        text = st.text_area(
            f"Group {i+1}",
            value="10, 12, 11, 13, 12",
            key=f"anova_{i}"
        )

        try:

            values = [
                float(x.strip())
                for x in text.split(",")
                if x.strip()
            ]

            groups.append(values)

        except ValueError:

            groups.append([])

            st.error(
                f"Invalid value in Group {i+1}."
            )

    if st.button(
        "📈 Calculate ANOVA",
        type="primary",
        use_container_width=True
    ):

        if any(
            len(group) < 2
            for group in groups
        ):

            st.error(
                "Every group must contain at least "
                "two observations."
            )

        else:

            try:

                f_stat, p_value = f_oneway(
                    *groups
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "F Statistic",
                        f"{f_stat:.8f}"
                    )

                with col2:

                    st.metric(
                        "P-value",
                        f"{p_value:.8f}"
                    )

                if p_value < alpha:

                    st.success(
                        interpretation(
                            p_value,
                            alpha
                        )
                    )

                else:

                    st.warning(
                        interpretation(
                            p_value,
                            alpha
                        )
                    )

            except Exception as e:

                st.error(
                    f"ANOVA calculation error: {e}"
                )


# ============================================================
# T-TEST
# ============================================================

elif test == "Independent Samples t-test":

    st.header(
        "🧪 Independent Samples t-test"
    )

    st.info(
        "Welch's t-test is used by default because it "
        "does not require equal population variances."
    )

    col1, col2 = st.columns(2)

    with col1:

        group1_text = st.text_area(
            "Group 1",
            value="10, 12, 11, 13, 12",
            key="ttest_group1"
        )

    with col2:

        group2_text = st.text_area(
            "Group 2",
            value="15, 16, 14, 17, 16",
            key="ttest_group2"
        )

    equal_var = st.checkbox(
        "Assume equal variances",
        value=False
    )

    try:

        group1 = [
            float(x.strip())
            for x in group1_text.split(",")
            if x.strip()
        ]

        group2 = [
            float(x.strip())
            for x in group2_text.split(",")
            if x.strip()
        ]

    except ValueError:

        group1 = []
        group2 = []

        st.error(
            "Enter valid numeric values."
        )

    if st.button(
        "🧪 Calculate t-test",
        type="primary",
        use_container_width=True
    ):

        if len(group1) < 2:

            st.error(
                "Group 1 requires at least "
                "two observations."
            )

        elif len(group2) < 2:

            st.error(
                "Group 2 requires at least "
                "two observations."
            )

        else:

            try:

                t_stat, p_value = ttest_ind(
                    group1,
                    group2,
                    equal_var=equal_var
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "t Statistic",
                        f"{t_stat:.8f}"
                    )

                with col2:

                    st.metric(
                        "P-value",
                        f"{p_value:.8f}"
                    )

                if equal_var:

                    st.info(
                        "Student's independent two-sample "
                        "t-test"
                    )

                else:

                    st.info(
                        "Welch's independent two-sample "
                        "t-test"
                    )

                if p_value < alpha:

                    st.success(
                        interpretation(
                            p_value,
                            alpha
                        )
                    )

                else:

                    st.warning(
                        interpretation(
                            p_value,
                            alpha
                        )
                    )

            except Exception as e:

                st.error(
                    f"t-test calculation error: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Statistical Test Analyzer | "
    "Python + NumPy + Pandas + SciPy + Streamlit"
)