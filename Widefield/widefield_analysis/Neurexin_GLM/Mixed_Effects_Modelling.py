import pandas as pd
import statsmodels.formula.api as smf


def fit_jaw_mixedlm(group_a, group_b, group_names=("A", "B"), reml=True):

    """
    Mixed-effects model for nested-list data:
        jaw ~ group + (1 | mouse)

    Parameters
    ----------
    group_a, group_b : list[list[float]]
        Nested lists shaped like [mouse][session].
    group_names : tuple[str, str]
        Names for the two groups (A is the reference).
    reml : bool
        Use REML (True) or ML (False).

    Returns
    -------
    fit : statsmodels MixedLMResults
        Use fit.summary(), fit.params, fit.pvalues, etc.
    df : pandas.DataFrame
        Long-format data used for the model.
    """

    a_name, b_name = group_names

    rows = []
    for gname, nested, prefix in [(a_name, group_a, "A"), (b_name, group_b, "B")]:
        for m, sessions in enumerate(nested):
            mouse_id = f"{prefix}{m:02d}"
            for s, y in enumerate(sessions):
                rows.append({"mouse": mouse_id, "session": s, "group": gname, "jaw": y})

    df = pd.DataFrame(rows)
    df["jaw"] = pd.to_numeric(df["jaw"], errors="coerce")
    df = df.dropna(subset=["jaw"]).reset_index(drop=True)

    # Make group categorical so A is the reference level
    df["group"] = pd.Categorical(df["group"], categories=[a_name, b_name], ordered=True)

    model = smf.mixedlm("jaw ~ C(group)", data=df, groups=df["mouse"], re_formula="1")
    fit = model.fit(reml=reml, method="lbfgs", maxiter=2000, disp=False)
    print(fit.summary())
    return fit, df
