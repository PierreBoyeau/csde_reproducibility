"""
    Simulations on negative control data.
"""

import json

import click
import numpy as np
import pandas as pd

from spatial_correction.ppi import PoissonIntercept


def store_args(output_file, **kwargs):
    with open(output_file, "w") as f:
        json.dump(kwargs, f, indent=4)
    return output_file, kwargs


def generate_data(
    n_total, d, generate_nulldata=False, misassignment_rate=0.0, de_rate=0.0, seed=None
):
    np.random.seed(seed)
    beta0 = np.random.normal(1, 1, d)
    beta1 = np.random.normal(1, 1, d)
    beta2 = np.random.normal(1, 1, d)

    if generate_nulldata:
        is_de1 = np.zeros(d)
        is_de2 = np.zeros(d)
        beta1 = np.zeros(d)
        beta2 = np.zeros(d)
    else:
        is_de1 = np.random.random(d) < de_rate
        beta1[is_de1 == 0] = 0
        is_de2 = np.random.random(d) < de_rate
        beta2[is_de2 == 0] = 0
    print(beta1)
    print(beta2)

    betas = np.vstack([beta0, beta0 + beta1, beta0 + beta2])
    y = np.random.randint(0, 3, n_total)
    betas_ = betas[y]
    x = np.random.poisson(np.exp(betas_))

    is_pred_noisy = np.random.random(n_total) < misassignment_rate
    y_pred = y.copy()
    y_pred[is_pred_noisy == 1] = np.random.randint(0, 3, n_total)[is_pred_noisy == 1]
    return {
        "beta0": beta0,
        "beta1": beta1,
        "is_de1": is_de1,
        "beta2": beta2,
        "is_de2": is_de2,
        "x": x,
        "y": y,
        "y_pred": y_pred,
        "feature": np.arange(d),
    }


@click.command()
@click.option("--n", type=int, default=300)
@click.option("--n_total", type=int, default=3000)
@click.option("--d", type=int, default=100)
@click.option("--misassignment_rate", type=float, default=0.0)
@click.option("--generate_nulldata", type=bool, default=False)
@click.option("--de_rate", type=float, default=0.0)
@click.option("--output", type=str)
@click.option("--seed", type=int, default=None)
@click.option("--tag", type=str)
def main(
    n, n_total, d, misassignment_rate, generate_nulldata, de_rate, output, seed, tag
):
    args_output = output.replace(".tsv", ".json")
    _, args = store_args(
        args_output,
        n=n,
        n_total=n_total,
        d=d,
        misassignment_rate=misassignment_rate,
        de_rate=de_rate,
    )

    data = generate_data(
        n_total=n_total,
        d=d,
        misassignment_rate=misassignment_rate,
        de_rate=de_rate,
        seed=seed,
        generate_nulldata=generate_nulldata,
    )
    metadata = pd.DataFrame(
        {key: value for key, value in data.items() if key not in ["x", "y", "y_pred"]}
    )

    x_gt_t = data["x"][:n]
    y_gt_t = data["y"][:n]
    x_gt_pred_t = data["x"][:n]
    y_gt_pred_t = data["y_pred"][:n]
    x_pred = data["x"][n:]
    y_pred = data["y_pred"][n:]

    config = dict(
        inputs_gt=(x_gt_t, y_gt_t),
        inputs_hat=(x_gt_pred_t, y_gt_pred_t),
        inputs_unl=(x_pred, y_pred),
        mu_prior_std=None,
        optimizer="gd",
        weight_init="zero",
        optimizer_kwargs={
            "tol": 1e-12,
            "optimizer": "adam",
            "learning_rate": 1e-2,
            "n_iter": int(5e3),
        },
    )

    ppi_model_nopred = PoissonIntercept(**config)
    ppi_model_nopred.get_asymptotic_distribution(lambd_=0.0, mode="overall")
    de_res = (
        ppi_model_nopred.differential_expression_ew(1, mode="overall")
        .assign(feature=data["feature"], model="baseline")
        .merge(metadata, on="feature")
    )

    ppi_model_half = PoissonIntercept(**config)
    ppi_model_half.get_asymptotic_distribution(lambd_=0.5, mode="overall")
    de_res_half = (
        ppi_model_half.differential_expression_ew(1, mode="overall")
        .assign(feature=data["feature"], model="PPI($\\lambda=0.5$)")
        .merge(metadata, on="feature")
    )

    ppi_model_auto = PoissonIntercept(**config)
    ppi_model_auto.get_asymptotic_distribution(lambd_=None, mode="overall")
    de_res_auto = (
        ppi_model_auto.differential_expression_ew(1, mode="overall")
        .assign(feature=data["feature"], model="PPI($\\lambda^\\star$)")
        .merge(metadata, on="feature")
    )

    ppi_model_auto2 = PoissonIntercept(**config)
    ppi_model_auto2.get_asymptotic_distribution(
        lambd_=None, mode="overall", idx_to_optimize=np.arange(d)
    )
    de_res_auto2 = (
        ppi_model_auto2.differential_expression_ew(1, mode="overall")
        .assign(feature=data["feature"], model="PPI($\\lambda^\\star$B)")
        .merge(metadata, on="feature")
    )

    all_res = [
        de_res,
        de_res_half,
        de_res_auto,
        de_res_auto2,
    ]
    all_res = pd.concat(all_res).assign(tag=tag, **args)
    all_res.to_csv(output, index=False, sep="\t")


if __name__ == "__main__":
    main()
