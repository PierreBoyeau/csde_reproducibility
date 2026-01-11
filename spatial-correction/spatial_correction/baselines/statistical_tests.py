import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


def extract_scanpy_de(uns_obj, key):
    """
    Extract DE results from scanpy rank_genes_groups
    """
    index = uns_obj["names"][key]
    lfc = uns_obj["logfoldchanges"][key]
    padj = uns_obj["pvals_adj"][key]
    pval = uns_obj["pvals"][key]

    lfc = np.clip(lfc, -5, 5)
    res = pd.DataFrame(
        {
            "lfc": lfc,
            "padj": padj,
            "pval": pval,
            "gene_name": index,
            "is_de": (padj < 0.05),  # & (np.abs(lfc) > 0.25)
        },
    ).set_index("gene_name")
    return res


def _glm_t_test(adata, idx, label_key, family="poisson"):
    if family == "poisson":
        family_ = sm.families.Poisson()
    elif family == "gaussian":
        family_ = sm.families.Gaussian()
    elif family == "nb":
        family_ = sm.families.NegativeBinomial()
    else:
        raise ValueError(f"Family {family} not supported")

    y = adata.X[:, idx]
    gene_name = adata.var_names[idx]
    x = adata.obs[label_key].astype(float).values.reshape(-1, 1)
    x_ = sm.add_constant(x)
    res = sm.GLM(y, x_, family=family_).fit()
    R = np.array([0, 1])
    res_sum = res.t_test(R)
    return {
        "lfc": res_sum.statistic.item(),
        "pval": res_sum.pvalue.item(),
        "gene_name": gene_name,
        "beta0": res.params[0],
        "beta1": res.params[1],
        "e0": np.exp(res.params[0]),
        "e1": np.exp(res.params[1] + res.params[0]),
    }


def glm_test(adata, label_key, subset_expressed=False, family="poisson"):
    res = []
    for idx in tqdm(range(adata.X.shape[1])):
        try:
            res_ = _glm_t_test(adata, idx, label_key, family)
        except Exception as e:
            res_ = {
                "lfc": 0,
                "pval": 1,
                "gene_name": adata.var_names[idx],
                "beta0": 0,
                "beta1": 0,
                "e0": 0,
                "e1": 0,
            }
        res.append(res_)
    res = pd.DataFrame(res)

    if subset_expressed:
        gene_expressed = (res["e0"] > 1.0) | (res["e1"] > 1.0)
        print(f"Subset expressed: {gene_expressed.sum()}")
        subset_res = res[gene_expressed]
        padjs = multipletests(subset_res["pval"].values, method="fdr_bh")[1]
        res.loc[:, "padj"] = 1.0
        res.loc[subset_res.index, "padj"] = padjs
    else:
        res.loc[:, "padj"] = multipletests(res["pval"], method="fdr_bh")[1]
    res.loc[:, "is_de"] = res["padj"] < 0.1
    return res
