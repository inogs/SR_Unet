import json
import matplotlib.pyplot as plt


def read_conf(conf_path):
    with open(conf_path, "r") as f:
        return json.load(f)


def main():

    conf = read_conf("conf_rmse_hist_blue.json")

    labels = conf["labels"]
    values = conf["values_chl"]

    if len(labels) != len(values):
        raise ValueError("labels and values must have the same length")

    plt.figure(figsize=(8, 5))

    colors = [conf["color"]] * len(labels)

    # cambio colore solo di alcune barre
    # colors[2] = conf["color2"]
    # colors[3] = conf["color2"]

    colors[4] = conf["color2"]
    colors[5] = conf["color2"]

    plt.bar(
        labels,
        values,
        color=colors
    )

    plt.xticks(rotation=70)

    plt.title(conf.get("title", ""))
    plt.xlabel(conf.get("xlabel", ""))
    plt.ylabel(conf.get("ylabel", ""))

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(conf["output_path"], dpi=300)
    plt.show()


if __name__ == "__main__":
    main()