"""Regression tests for Shapley attribution over a prepared conversion table.

These tests use ready fixtures only (no CSV reading, no raw exports):

* ``DF_AGG`` - a ready journeys DataFrame (channels + conversion value);
* ``CONV_TABLE`` - a ready conversion table written by hand, i.e. exactly what
  an external caller would pass to ``attribution_shapley_from_conversion_table``.

They check:

1. the table-based path matches the classic path (``attribution_shapley``);
2. golden values for a fixed conversion table (three different value columns);
3. the prepared-table contract: ``size`` is not re-applied, the caller's table
   is not modified;
4. validation of a broken external table (it must fail loudly instead of
   silently turning v(S) into 0);
5. ``coalitions()`` still returns exactly the same matrix after the refactor,
   including exact membership for channel names where one is a substring of
   another (``p_43`` vs ``p_430``).
"""

import pandas as pd
import pytest

from marketing_attribution_models import MAM
from marketing_attribution_models.MAM import (
    channel_values_from_conversion_table,
    shapley_from_conversion_table,
)

SEP = " > "
SIZE = 3
CONV_VALUE = 3

# Ready journeys fixture (same shape as in tests/test_mam.py).
DF_AGG = pd.DataFrame(
    {
        "channels_agg": [
            "A",
            "A > B",
            "A > B > C",
            "B",
            "B > A",
            "B > C > A",
            "C",
            "C > A",
            "C > B > A",
        ],
        "conversion_value": [CONV_VALUE for _ in range(9)],
    }
)

# Conversion table that MAM builds from DF_AGG (journey_conversion_table(size=3)).
TABLE_FOR_DF_AGG = pd.DataFrame(
    {
        "combinations": ["A", "A > B", "A > B > C", "A > C", "B", "C"],
        "conversions": [1, 2, 3, 1, 1, 1],
        "total_sequences": [1, 2, 3, 1, 1, 1],
        "conversion_value": [3, 6, 9, 3, 3, 3],
        "conv_rate": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }
)

EXPECTED_FOR_DF_AGG = {"A": 13.5, "B": 7.5, "C": 6.0}

# Ready conversion table written by hand: the caller has already applied its own
# size truncation, so the library must not truncate anything again.
CONV_TABLE = pd.DataFrame(
    {
        "combinations": ["A", "A > B", "A > B > C", "B", "B > C"],
        "conversions": [2, 3, 1, 1, 2],
        "total_sequences": [5, 4, 2, 3, 4],
        "conversion_value": [6, 12, 5, 3, 8],
        "conv_rate": [0.4, 0.75, 0.5, 1 / 3, 0.5],
    }
)

EXPECTED_CONV_RATE = {"A": 14.561111111, "B": 19.327777778, "C": 0.111111111}
EXPECTED_CONVERSION_VALUE = {"A": 16.0, "B": 18.0, "C": 0.0}
EXPECTED_CONVERSIONS = {"A": 4.666666667, "B": 4.666666667, "C": -0.333333333}


def _build_mam() -> MAM:
    """MAM object built from the ready journeys fixture."""
    return MAM(
        DF_AGG.copy(),
        conversion_value="conversion_value",
        channels_colname="channels_agg",
    )


def _frame_to_dict(frame: pd.DataFrame) -> dict:
    """Per-channel result of a (channels, value) frame: {channel: value}."""
    return {
        str(channel): float(value)
        for channel, value in zip(frame.iloc[:, 0], frame.iloc[:, 1], strict=True)
    }


def _series_to_dict(series: pd.Series) -> dict:
    """Per-channel result of a Series: {channel: value}."""
    return {str(channel): float(value) for channel, value in series.items()}


def test_journey_conversion_table_fixture() -> None:
    """The ready table fixture equals what the library builds from DF_AGG."""
    table = _build_mam().journey_conversion_table(size=SIZE)

    pd.testing.assert_frame_equal(
        table.astype({"combinations": object}),
        TABLE_FOR_DF_AGG.astype({"combinations": object}),
    )


def test_from_table_matches_attribution_shapley() -> None:
    """Table-based Shapley == classic Shapley on the same ready fixture."""
    mam = _build_mam()
    table = mam.journey_conversion_table(size=SIZE)
    classic = _frame_to_dict(
        mam.attribution_shapley(size=SIZE, values_col="conv_rate")[1]
    )

    result_table, model_name = shapley_from_conversion_table(
        table, sep=SEP, size=SIZE, order=False, values_col="conv_rate"
    )
    by_table = _series_to_dict(
        channel_values_from_conversion_table(result_table, model_name, sep=SEP)
    )

    assert by_table == pytest.approx(classic, rel=1e-9)
    assert by_table == pytest.approx(EXPECTED_FOR_DF_AGG, rel=1e-9)


def test_from_table_via_mam_method() -> None:
    """The new MAM method reproduces the classic call and names the column."""
    mam = _build_mam()
    table = mam.journey_conversion_table(size=SIZE)
    classic = _frame_to_dict(
        _build_mam().attribution_shapley(size=SIZE, values_col="conv_rate")[1]
    )

    result_table, frame = mam.attribution_shapley_from_conversion_table(
        table, size=SIZE, values_col="conv_rate"
    )

    assert _frame_to_dict(frame) == pytest.approx(classic, rel=1e-9)
    assert (
        f"attribution_shapley_size{SIZE}_conv_rate_algorithmic" in result_table.columns
    )


def test_golden_values_for_prepared_table() -> None:
    """Ready conversion table -> pinned Shapley values for three value columns."""
    for values_col, expected, total in (
        ("conv_rate", EXPECTED_CONV_RATE, 34.0),
        ("conversion_value", EXPECTED_CONVERSION_VALUE, 34.0),
        ("conversions", EXPECTED_CONVERSIONS, 9.0),
    ):
        result_table, model_name = shapley_from_conversion_table(
            CONV_TABLE, sep=SEP, size=SIZE, order=False, values_col=values_col
        )
        values = _series_to_dict(
            channel_values_from_conversion_table(result_table, model_name, sep=SEP)
        )

        assert values == pytest.approx(expected, rel=1e-9, abs=1e-9), values_col
        # Эффективность: сумма весов = объём, который распределяет колонка
        assert sum(values.values()) == pytest.approx(total, rel=1e-9), values_col


def test_input_table_is_not_modified() -> None:
    """The caller's table stays intact; the result is a new table."""
    before = CONV_TABLE.copy(deep=True)

    result_table, model_name = shapley_from_conversion_table(
        CONV_TABLE, sep=SEP, size=SIZE, order=False, values_col="conv_rate"
    )

    pd.testing.assert_frame_equal(CONV_TABLE, before)
    assert list(result_table.columns) == [*CONV_TABLE.columns, model_name]


def test_size_is_not_reapplied_to_prepared_table() -> None:
    """Truncation is the table owner's job: size=2 must not cut a 3-channel key."""
    three_channels = pd.DataFrame(
        {
            "combinations": ["A > B > C"],
            "conversions": [1],
            "total_sequences": [1],
            "conversion_value": [9],
            "conv_rate": [1.0],
        }
    )

    result_table, model_name = shapley_from_conversion_table(
        three_channels, sep=SEP, size=2, order=False, values_col="conversion_value"
    )
    values = _series_to_dict(
        channel_values_from_conversion_table(result_table, model_name, sep=SEP)
    )

    assert set(values) == {"A", "B", "C"}
    assert values == pytest.approx({"A": 3.0, "B": 3.0, "C": 3.0}, rel=1e-9)


def test_group_by_channels_models_false() -> None:
    """The old return contract is kept when aggregation is disabled."""
    mam = _build_mam()
    table = mam.journey_conversion_table(size=SIZE)

    result_table, frame = mam.attribution_shapley_from_conversion_table(
        table,
        size=SIZE,
        values_col="conv_rate",
        group_by_channels_models=False,
    )

    assert frame == "group_by_channels_models=False"
    assert (
        f"attribution_shapley_size{SIZE}_conv_rate_algorithmic" in result_table.columns
    )


def test_custom_values_branch() -> None:
    """merge_custom_values keeps working through the table-based path."""
    # Значения заданы для ВСЕХ наборов: строки с нулевым custom-значением дают
    # sum(phi) == 0 и NaN в делении (предсуществующее поведение ветки custom_value).
    custom_values = pd.DataFrame(
        {
            "combination": ["A", "A > B", "A > B > C", "B", "B > C"],
            "value": [4.0, 10.0, 6.0, 4.0, 2.0],
        }
    )

    result_table, model_name = shapley_from_conversion_table(
        CONV_TABLE,
        sep=SEP,
        size=SIZE,
        order=False,
        merge_custom_values=custom_values.copy(),
    )

    assert model_name == f"attribution_shapley_size{SIZE}_custom_value_algorithmic"
    assert model_name in result_table.columns


@pytest.mark.parametrize(
    ("broken", "message"),
    [
        (CONV_TABLE.drop(columns=["conversion_value"]), "отсутствуют"),
        (
            pd.concat([CONV_TABLE, CONV_TABLE.head(1)], ignore_index=True),
            "уникальным",
        ),
        (
            CONV_TABLE.assign(combinations=["A", "B > A", "A > B > C", "B", "B > C"]),
            "каноничны",
        ),
        (
            CONV_TABLE.assign(
                conversions=[2, 3, 1, 1, 2],
                total_sequences=[1, 1, 1, 1, 1],
            ),
            "превышать",
        ),
        (CONV_TABLE.assign(total_sequences=[0, 1, 1, 1, 1]), "больше 0"),
    ],
)
def test_validation_rejects_broken_tables(broken: pd.DataFrame, message: str) -> None:
    """A broken external table fails loudly instead of turning v(S) into 0."""
    with pytest.raises(ValueError, match=message):
        shapley_from_conversion_table(broken, sep=SEP)


def test_coalitions_exact_membership_is_kept() -> None:
    """coalitions() is unchanged by the refactor, including the substring fix.

    Channel "p_43" must NOT be reported as a member of coalitions that contain
    only "p_430" (a substring-matching bug fixed in this fork).
    """
    channels = ["p_43", "p_430", "p_5"]
    coalitions = _build_mam().coalitions(msize=3, unique_channels=channels, order=False)

    actual = {
        tuple(row): (bool(flag_43), bool(flag_430), bool(flag_5))
        for row, flag_43, flag_430, flag_5 in zip(
            coalitions["combinations"],
            coalitions["p_43"],
            coalitions["p_430"],
            coalitions["p_5"],
            strict=True,
        )
    }

    assert actual == {
        (): (False, False, False),
        ("p_43",): (True, False, False),
        ("p_430",): (False, True, False),
        ("p_5",): (False, False, True),
        ("p_43", "p_430"): (True, True, False),
        ("p_43", "p_5"): (True, False, True),
        ("p_430", "p_5"): (False, True, True),
        ("p_43", "p_430", "p_5"): (True, True, True),
    }
